use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine_llm::apply_chat_template;
use my_nn_engine_llm::build_llama;
use my_nn_engine_llm::llama::build_llama_prefill;
use my_nn_engine_llm::ChatMessage;
use my_nn_engine_llm::ChatTemplateError;
use my_nn_engine_llm::HfConfig;
use my_nn_engine_llm::HfWeights;
use my_nn_engine_llm::LlamaWeights;
use my_nn_engine_llm::LlmError;
use my_nn_engine_llm::LlmSession;
use serde::Deserialize;
use tokenizers::Tokenizer;
use uuid::Uuid;

const SESSION_TTL: Duration = Duration::from_secs(30 * 60);
const MAX_SEQ_LEN: usize = 1024;
const PREFILL_LEN: usize = 16;
const DEFAULT_MAX_TOKENS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LlmModelId {
    TinyLlama,
}

impl LlmModelId {
    pub fn dir_name(&self) -> &'static str {
        match self {
            LlmModelId::TinyLlama => "tinyllama",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            LlmModelId::TinyLlama => "TinyLlama",
        }
    }
}

pub type SessionId = String;

pub struct ChatSession {
    #[allow(dead_code)] // exposed for future eviction / introspection
    pub model_id: LlmModelId,
    #[allow(dead_code)]
    pub target: Target,
    llm: LlmSession,
    tokenizer: Tokenizer,
    history: Vec<ChatMessage>,
    chat_template: String,
    eos_token_str: String,
    bos_token_str: String,
    eos_token_id: u32,
    last_used: Instant,
    /// True iff the previous turn ended at the model emitting EOS. When false
    /// (i.e. we hit max_tokens), the KV cache state doesn't include a clean
    /// turn boundary and we must reset before the next turn.
    last_turn_eos_emitted: bool,
}

impl ChatSession {
    fn render_full(&self, add_gen_prompt: bool) -> Result<String, ChatTemplateError> {
        apply_chat_template(
            &self.chat_template,
            &self.history,
            &self.eos_token_str,
            &self.bos_token_str,
            add_gen_prompt,
        )
    }
}

pub struct ChatRegistry {
    sessions: Mutex<HashMap<SessionId, ChatSession>>,
    models_dir: PathBuf,
}

#[derive(Debug)]
pub enum ChatError {
    Llm(LlmError),
    Template(ChatTemplateError),
    UnknownSession,
    LoadConfig(String),
    LoadTokenizer(String),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Llm(e) => write!(f, "{e}"),
            ChatError::Template(e) => write!(f, "chat template: {e}"),
            ChatError::UnknownSession => write!(f, "unknown chat session"),
            ChatError::LoadConfig(s) => write!(f, "config load failed: {s}"),
            ChatError::LoadTokenizer(s) => write!(f, "tokenizer load failed: {s}"),
        }
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    chat_template: String,
    eos_token: TokenSpec,
    bos_token: TokenSpec,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenSpec {
    Str(String),
    Obj { content: String },
}

impl TokenSpec {
    fn as_str(&self) -> &str {
        match self {
            TokenSpec::Str(s) => s,
            TokenSpec::Obj { content } => content,
        }
    }
}

pub struct ChatTurnResult {
    pub assistant_message: String,
    pub tokens_generated: u32,
    pub generation_time: Duration,
    pub eos_emitted: bool,
}

impl ChatRegistry {
    pub fn new(models_dir: impl AsRef<Path>) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            models_dir: models_dir.as_ref().to_path_buf(),
        }
    }

    pub fn create(
        &self,
        model_id: LlmModelId,
        target: Target,
    ) -> Result<(SessionId, Duration), ChatError> {
        let started = Instant::now();
        let model_dir = self.models_dir.join("hf").join(model_id.dir_name());

        log::info!(
            "Loading {} from {} for chat session",
            model_id.display_name(),
            model_dir.display()
        );

        let config = HfConfig::from_path(model_dir.join("config.json"))
            .map_err(|e| ChatError::LoadConfig(format!("{e}")))?;
        let hf = HfWeights::from_dir(&model_dir)
            .map_err(|e| ChatError::LoadConfig(format!("hf weights: {e}")))?;
        let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers)
            .map_err(|e| ChatError::LoadConfig(format!("llama weights: {e}")))?;

        let tok_cfg_str = std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
            .map_err(|e| ChatError::LoadConfig(format!("tokenizer_config.json: {e}")))?;
        let tok_cfg: TokenizerConfig = serde_json::from_str(&tok_cfg_str)
            .map_err(|e| ChatError::LoadConfig(format!("tokenizer_config.json parse: {e}")))?;

        let r = build_llama(&config, &weights, MAX_SEQ_LEN);
        let p = build_llama_prefill(&config, &weights, MAX_SEQ_LEN, PREFILL_LEN);

        let opts = Options::builder().target(target).build();
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| ChatError::LoadTokenizer(format!("{e}")))?;
        let tokenizer_for_decode = tokenizer.clone();

        let llm = LlmSession::for_llama_with_prefill(
            r.graph,
            p.graph,
            r.kv_cache_names,
            PREFILL_LEN,
            tokenizer,
            &opts,
            MAX_SEQ_LEN,
            config.eos_token_id,
        )
        .map_err(ChatError::Llm)?;

        let session_id = Uuid::new_v4().to_string();
        let session = ChatSession {
            model_id,
            target,
            llm,
            tokenizer: tokenizer_for_decode,
            history: Vec::new(),
            chat_template: tok_cfg.chat_template,
            eos_token_str: tok_cfg.eos_token.as_str().to_string(),
            bos_token_str: tok_cfg.bos_token.as_str().to_string(),
            eos_token_id: config.eos_token_id,
            last_used: Instant::now(),
            last_turn_eos_emitted: true,
        };
        self.sessions
            .lock()
            .unwrap()
            .insert(session_id.clone(), session);
        log::info!("Created chat session {session_id}");
        Ok((session_id, started.elapsed()))
    }

    pub fn destroy(&self, id: &SessionId) {
        if self.sessions.lock().unwrap().remove(id).is_some() {
            log::info!("Destroyed chat session {id}");
        }
    }

    pub fn chat(
        &self,
        id: &SessionId,
        user_message: String,
        max_tokens: u32,
    ) -> Result<ChatTurnResult, ChatError> {
        let max_tokens = if max_tokens == 0 {
            DEFAULT_MAX_TOKENS
        } else {
            max_tokens
        };

        self.evict_idle();

        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions.get_mut(id).ok_or(ChatError::UnknownSession)?;

        // KV-cache reuse strategy:
        //   prev = render(history, add_gen_prompt=false)
        //   with_new = render(history + new_user, add_gen_prompt=true)
        // Feed only `with_new[len(prev)..]`. If the previous turn was truncated
        // (no EOS), the KV cache is missing the closing tokens the template
        // expects, so reset and re-prefill from scratch.
        let prev = if session.history.is_empty() || !session.last_turn_eos_emitted {
            if !session.last_turn_eos_emitted {
                log::info!("Resetting KV cache for session {id}: previous turn hit max_tokens");
            }
            session.llm.reset();
            String::new()
        } else {
            session.render_full(false).map_err(ChatError::Template)?
        };

        session.history.push(ChatMessage::user(user_message));
        let with_new = session.render_full(true).map_err(ChatError::Template)?;

        let delta = if !prev.is_empty() && with_new.starts_with(&prev) {
            with_new[prev.len()..].to_string()
        } else {
            // Template prefix mismatch (rare; most templates are append-only).
            log::warn!("Template prefix mismatch for session {id}; resetting KV cache");
            session.llm.reset();
            with_new
        };

        let started = Instant::now();
        let new_ids = session
            .llm
            .generate_ids(&delta, max_tokens as usize)
            .map_err(ChatError::Llm)?;
        let elapsed = started.elapsed();

        let eos_emitted = new_ids.last().copied() == Some(session.eos_token_id);
        let assistant_text = session
            .tokenizer
            .decode(&new_ids, true)
            .map_err(|e| ChatError::LoadTokenizer(format!("decode: {e}")))?;

        let tokens_generated = new_ids.len() as u32;
        session
            .history
            .push(ChatMessage::assistant(assistant_text.clone()));
        session.last_used = Instant::now();
        session.last_turn_eos_emitted = eos_emitted;

        Ok(ChatTurnResult {
            assistant_message: assistant_text,
            tokens_generated,
            generation_time: elapsed,
            eos_emitted,
        })
    }

    fn evict_idle(&self) {
        let now = Instant::now();
        let mut sessions = self.sessions.lock().unwrap();
        sessions.retain(|id, s| {
            let alive = now.duration_since(s.last_used) < SESSION_TTL;
            if !alive {
                log::info!("Evicting idle chat session {id}");
            }
            alive
        });
    }
}
