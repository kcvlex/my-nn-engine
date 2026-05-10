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
use my_nn_engine_llm::GenerateOptions;
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

pub type SessionId = String;

pub struct ChatSession {
    #[allow(dead_code)] // exposed for future eviction / introspection
    pub model_dir: String,
    #[allow(dead_code)]
    pub target: Target,
    llm: LlmSession,
    tokenizer: Tokenizer,
    history: Vec<ChatMessage>,
    chat_template: String,
    eos_token_str: String,
    bos_token_str: String,
    eos_token_id: u32,
    stop_strings: Vec<String>,
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
    models_root: PathBuf,
}

#[derive(Debug)]
pub enum ChatError {
    Llm(LlmError),
    Template(ChatTemplateError),
    UnknownSession,
    InvalidModelDir(String),
    LoadConfig(String),
    LoadTokenizer(String),
    MissingChatTemplate(String),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Llm(e) => write!(f, "{e}"),
            ChatError::Template(e) => write!(f, "chat template: {e}"),
            ChatError::UnknownSession => write!(f, "unknown chat session"),
            ChatError::InvalidModelDir(s) => write!(f, "invalid model_dir: {s}"),
            ChatError::LoadConfig(s) => write!(f, "config load failed: {s}"),
            ChatError::LoadTokenizer(s) => write!(f, "tokenizer load failed: {s}"),
            ChatError::MissingChatTemplate(s) => write!(f, "model has no chat_template: {s}"),
        }
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
    eos_token: Option<TokenSpec>,
    bos_token: Option<TokenSpec>,
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

/// One event in a streaming chat turn. The producer emits zero or more
/// `Chunk`s as text becomes safe to flush past any pending stop-string match,
/// then exactly one `Done` carrying terminal stats.
pub enum ChatStreamEvent {
    Chunk { delta: String },
    Done(ChatTurnDone),
}

pub struct ChatTurnDone {
    pub tokens_generated: u32,
    pub generation_time: Duration,
    pub eos_emitted: bool,
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    if s.len() <= idx {
        return s.len();
    }
    while 0 < idx && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Returns true if `name` is a safe single-segment directory name (no `/`, no
/// `..`, no leading dot, non-empty).
fn validate_model_dir(name: &str) -> Result<(), ChatError> {
    if name.is_empty() {
        return Err(ChatError::InvalidModelDir("empty".into()));
    }
    if name.contains('/') || name.contains('\\') || name.contains('\0') {
        return Err(ChatError::InvalidModelDir("contains path separator".into()));
    }
    if name == "." || name == ".." || name.starts_with("..") {
        return Err(ChatError::InvalidModelDir("traversal".into()));
    }
    Ok(())
}

/// Whether to compile a separate prefill graph alongside the decode graph.
/// We default to no-prefill (the Llama2 example's path); models small enough
/// that the extra graph is cheap can opt in via this allowlist.
fn dir_uses_prefill(dir_name: &str) -> bool {
    matches!(dir_name, "tinyllama" | "tiny-llama-random")
}

impl ChatRegistry {
    pub fn new(models_root: impl AsRef<Path>) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            models_root: models_root.as_ref().to_path_buf(),
        }
    }

    pub fn list(&self) -> Vec<String> {
        let dir = &self.models_root;
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut out: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type()
                    .map(|t| t.is_dir() || t.is_symlink())
                    .unwrap_or(false)
            })
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|name| !name.starts_with('.'))
            .filter(|name| has_chat_template(&dir.join(name)))
            .collect();
        out.sort();
        out
    }

    pub fn create(
        &self,
        model_dir: &str,
        target: Target,
    ) -> Result<(SessionId, Duration), ChatError> {
        validate_model_dir(model_dir)?;
        let started = Instant::now();
        let path = self.models_root.join(model_dir);

        log::info!(
            "Loading {} from {} for chat session",
            model_dir,
            path.display()
        );

        let config = HfConfig::from_path(path.join("config.json"))
            .map_err(|e| ChatError::LoadConfig(format!("config.json: {e}")))?;
        let hf = HfWeights::from_dir(&path)
            .map_err(|e| ChatError::LoadConfig(format!("hf weights: {e}")))?;
        let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers)
            .map_err(|e| ChatError::LoadConfig(format!("llama weights: {e}")))?;

        let (chat_template, eos_token_str, bos_token_str) = load_chat_meta(&path)?;

        let r = build_llama(&config, &weights, MAX_SEQ_LEN);

        let opts = Options::builder().target(target).build();
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| ChatError::LoadTokenizer(format!("{e}")))?;
        let tokenizer_for_decode = tokenizer.clone();
        let stop_strings = load_stop_strings(&path, &eos_token_str, &tokenizer_for_decode)?;
        log::info!("Stop strings for {model_dir}: {stop_strings:?}");

        let llm = if dir_uses_prefill(model_dir) {
            let p = build_llama_prefill(&config, &weights, MAX_SEQ_LEN, PREFILL_LEN);
            LlmSession::for_llama_with_prefill(
                r.graph,
                p.graph,
                r.kv_cache_names,
                PREFILL_LEN,
                tokenizer,
                &opts,
                MAX_SEQ_LEN,
                config.eos_token_id,
            )
        } else {
            LlmSession::for_llama(
                r.graph,
                r.kv_cache_names,
                tokenizer,
                &opts,
                MAX_SEQ_LEN,
                config.eos_token_id,
            )
        }
        .map_err(ChatError::Llm)?;

        let session_id = Uuid::new_v4().to_string();
        let session = ChatSession {
            model_dir: model_dir.to_string(),
            target,
            llm,
            tokenizer: tokenizer_for_decode,
            history: Vec::new(),
            chat_template,
            eos_token_str,
            bos_token_str,
            eos_token_id: config.eos_token_id,
            stop_strings,
            last_used: Instant::now(),
            last_turn_eos_emitted: true,
        };
        self.sessions
            .lock()
            .unwrap()
            .insert(session_id.clone(), session);
        log::info!("Created chat session {session_id} ({model_dir})");
        Ok((session_id, started.elapsed()))
    }

    pub fn destroy(&self, id: &SessionId) {
        if self.sessions.lock().unwrap().remove(id).is_some() {
            log::info!("Destroyed chat session {id}");
        }
    }

    /// `on_event` fires synchronously
    /// from the generation loop: zero or more [`ChatStreamEvent::Chunk`]
    /// events, followed by exactly one [`ChatStreamEvent::Done`] (also
    /// returned). Chunks hold back the trailing `max(stop_string.len())`
    /// bytes of decoded text so a stop string can never appear in an emitted
    /// chunk.
    pub fn chat_stream(
        &self,
        id: &SessionId,
        user_message: String,
        max_tokens: u32,
        on_event: &mut dyn FnMut(ChatStreamEvent),
    ) -> Result<ChatTurnDone, ChatError> {
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
        let opts = GenerateOptions {
            max_new_tokens: max_tokens as usize,
            stop_strings: session.stop_strings.clone(),
        };

        // Hold back at least max(stop_string byte length) from the tail of
        // every chunk. Any in-flight stop match must lie entirely within this
        // window, so an emitted chunk can never contain a stop string.
        let lookahead = session
            .stop_strings
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);

        let mut accumulated_text = String::new();
        let mut emitted_len: usize = 0;
        let new_ids = {
            let mut decode_stream = session.tokenizer.decode_stream(true);
            session
                .llm
                .generate_ids_with_callback(&delta, &opts, &mut |tok| {
                    // decode_stream buffers incomplete byte-fallback sequences
                    // internally and only yields complete UTF-8 once the bytes
                    // form valid chars, so emitted text is always byte-safe.
                    if let Ok(Some(new_text)) = decode_stream.step(tok) {
                        accumulated_text.push_str(&new_text);
                    }
                    let safe_end = floor_char_boundary(
                        &accumulated_text,
                        accumulated_text.len().saturating_sub(lookahead),
                    );
                    if emitted_len < safe_end {
                        let chunk = accumulated_text[emitted_len..safe_end].to_string();
                        emitted_len = safe_end;
                        on_event(ChatStreamEvent::Chunk { delta: chunk });
                    }
                })
        }
        .map_err(ChatError::Llm)?;
        let elapsed = started.elapsed();

        let eos_emitted = (new_ids.len() as u32) < max_tokens ||
            new_ids.last().copied() == Some(session.eos_token_id);
        let assistant_text = session
            .tokenizer
            .decode(&new_ids, true)
            .map_err(|e| ChatError::LoadTokenizer(format!("decode: {e}")))?;

        // Flush whatever's left after the lookahead window (and after any
        // truncate_at_stop rollback inside the LLM session). Snap emitted_len
        // to a boundary in the post-truncation text since post-processing
        // can shift bytes between the in-loop `full` and `assistant_text`.
        let start = floor_char_boundary(&assistant_text, emitted_len.min(assistant_text.len()));
        if start < assistant_text.len() {
            let tail = assistant_text[start..].to_string();
            on_event(ChatStreamEvent::Chunk { delta: tail });
        }

        let tokens_generated = new_ids.len() as u32;
        session.history.push(ChatMessage::assistant(assistant_text));
        session.last_used = Instant::now();
        session.last_turn_eos_emitted = eos_emitted;

        let done = ChatTurnDone {
            tokens_generated,
            generation_time: elapsed,
            eos_emitted,
        };
        on_event(ChatStreamEvent::Done(ChatTurnDone {
            tokens_generated: done.tokens_generated,
            generation_time: done.generation_time,
            eos_emitted: done.eos_emitted,
        }));
        Ok(done)
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

fn load_chat_meta(path: &Path) -> Result<(String, String, String), ChatError> {
    let cfg_path = path.join("tokenizer_config.json");
    if !cfg_path.exists() {
        return Err(ChatError::MissingChatTemplate(format!(
            "{} not found",
            cfg_path.display()
        )));
    }
    let s = std::fs::read_to_string(&cfg_path)
        .map_err(|e| ChatError::LoadConfig(format!("tokenizer_config.json: {e}")))?;
    let parsed: TokenizerConfig = serde_json::from_str(&s)
        .map_err(|e| ChatError::LoadConfig(format!("tokenizer_config.json parse: {e}")))?;

    let chat_template = parsed.chat_template.ok_or_else(|| {
        ChatError::MissingChatTemplate(format!(
            "chat_template field missing in {}",
            cfg_path.display()
        ))
    })?;
    let eos_token_str = parsed
        .eos_token
        .as_ref()
        .map(|t| t.as_str().to_string())
        .unwrap_or_else(|| "</s>".to_string());
    let bos_token_str = parsed
        .bos_token
        .as_ref()
        .map(|t| t.as_str().to_string())
        .unwrap_or_else(|| "<s>".to_string());
    Ok((chat_template, eos_token_str, bos_token_str))
}

fn has_chat_template(model_path: &Path) -> bool {
    let cfg_path = model_path.join("tokenizer_config.json");
    let Ok(s) = std::fs::read_to_string(&cfg_path) else {
        return false;
    };
    serde_json::from_str::<TokenizerConfig>(&s)
        .map(|c| c.chat_template.is_some())
        .unwrap_or(false)
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosIds {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    eos_token_id: Option<EosIds>,
}

fn load_stop_strings(
    path: &Path,
    eos_token_str: &str,
    tokenizer: &Tokenizer,
) -> Result<Vec<String>, ChatError> {
    let mut out: Vec<String> = vec![eos_token_str.to_string()];

    let gen_path = path.join("generation_config.json");
    if gen_path.exists() {
        let s = std::fs::read_to_string(&gen_path)
            .map_err(|e| ChatError::LoadConfig(format!("generation_config.json: {e}")))?;
        let parsed: GenerationConfig = serde_json::from_str(&s)
            .map_err(|e| ChatError::LoadConfig(format!("generation_config.json parse: {e}")))?;
        let ids: Vec<u32> = match parsed.eos_token_id {
            Some(EosIds::One(x)) => vec![x],
            Some(EosIds::Many(xs)) => xs,
            None => vec![],
        };
        for id in ids {
            if let Some(tok) = tokenizer.id_to_token(id) {
                out.push(tok);
            }
        }
    }

    out.sort();
    out.dedup();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA2_CHAT_TEMPLATE: &str = "{% for message in messages %}\n\
{% if message['role'] == 'user' %}\
{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}\n\
{% elif message['role'] == 'assistant' %}\
{{ ' ' + message['content'] + ' ' + eos_token }}\n\
{% endif %}\n\
{% endfor %}";

    const TINYLLAMA_TEMPLATE: &str = r#"{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>
' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ '<|system|>
' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>
'  + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}
{% endfor %}"#;

    #[test]
    fn token_spec_deserializes_string_form() {
        let spec: TokenSpec = serde_json::from_str(r#""</s>""#).unwrap();
        assert_eq!(spec.as_str(), "</s>");
    }

    #[test]
    fn token_spec_deserializes_object_form() {
        let spec: TokenSpec = serde_json::from_str(
            r#"{"content": "</s>", "lstrip": false, "normalized": false, "rstrip": false, "single_word": false}"#,
        )
        .unwrap();
        assert_eq!(spec.as_str(), "</s>");
    }

    #[test]
    fn validate_model_dir_accepts_simple_name() {
        assert!(validate_model_dir("tinyllama").is_ok());
        assert!(validate_model_dir("llama2-7b-sft").is_ok());
    }

    #[test]
    fn validate_model_dir_rejects_traversal_and_separators() {
        assert!(validate_model_dir("").is_err());
        assert!(validate_model_dir("..").is_err());
        assert!(validate_model_dir("../escape").is_err());
        assert!(validate_model_dir("a/b").is_err());
        assert!(validate_model_dir("a\\b").is_err());
    }

    #[test]
    fn chat_error_display_strips_debug_wrappers() {
        let e = ChatError::UnknownSession;
        assert_eq!(e.to_string(), "unknown chat session");

        let e = ChatError::LoadConfig("missing config.json".to_string());
        assert_eq!(e.to_string(), "config load failed: missing config.json");

        let e = ChatError::InvalidModelDir("traversal".to_string());
        assert_eq!(e.to_string(), "invalid model_dir: traversal");

        let e = ChatError::MissingChatTemplate("chat_template field missing in foo".to_string());
        assert_eq!(
            e.to_string(),
            "model has no chat_template: chat_template field missing in foo"
        );
    }

    #[test]
    fn load_chat_meta_rejects_missing_tokenizer_config() {
        let tmp = tempfile::tempdir().unwrap();
        let err = load_chat_meta(tmp.path()).unwrap_err();
        assert!(
            matches!(err, ChatError::MissingChatTemplate(_)),
            "got {err}"
        );
    }

    #[test]
    fn load_chat_meta_rejects_tokenizer_config_without_chat_template() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            r#"{"eos_token": "</s>"}"#,
        )
        .unwrap();
        let err = load_chat_meta(tmp.path()).unwrap_err();
        assert!(
            matches!(err, ChatError::MissingChatTemplate(_)),
            "got {err}"
        );
    }

    #[test]
    fn load_chat_meta_accepts_valid_config() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            r#"{"chat_template": "TPL", "eos_token": "<eos>", "bos_token": "<bos>"}"#,
        )
        .unwrap();
        let (tpl, eos, bos) = load_chat_meta(tmp.path()).unwrap();
        assert_eq!(tpl, "TPL");
        assert_eq!(eos, "<eos>");
        assert_eq!(bos, "<bos>");
    }

    #[test]
    fn has_chat_template_detects_presence() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!has_chat_template(tmp.path()));
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            r#"{"eos_token": "</s>"}"#,
        )
        .unwrap();
        assert!(!has_chat_template(tmp.path()));
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            r#"{"chat_template": "TPL"}"#,
        )
        .unwrap();
        assert!(has_chat_template(tmp.path()));
    }

    /// KV-cache reuse correctness invariant: rendering the conversation
    /// without the new turn must be a prefix of rendering it with the new
    /// turn appended. If this breaks, the chat() delta extraction would feed
    /// wrong tokens to the model.
    #[test]
    fn template_prefix_invariant_holds_for_tinyllama() {
        let history = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];
        let mut with_new = history.clone();
        with_new.push(ChatMessage::user("How are you?"));

        let prev = apply_chat_template(TINYLLAMA_TEMPLATE, &history, "</s>", "<s>", false).unwrap();
        let with_new_rendered =
            apply_chat_template(TINYLLAMA_TEMPLATE, &with_new, "</s>", "<s>", true).unwrap();

        assert!(
            with_new_rendered.starts_with(&prev),
            "template not append-only:\n--- prev ---\n{prev}\n--- with_new ---\n{with_new_rendered}"
        );
        let suffix = &with_new_rendered[prev.len()..];
        assert!(
            suffix.contains("How are you?"),
            "suffix missing new user content: {suffix:?}"
        );
    }

    #[test]
    fn template_prefix_invariant_holds_for_llama2_fallback() {
        let history = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];
        let mut with_new = history.clone();
        with_new.push(ChatMessage::user("How are you?"));

        let prev =
            apply_chat_template(LLAMA2_CHAT_TEMPLATE, &history, "</s>", "<s>", false).unwrap();
        let with_new_rendered =
            apply_chat_template(LLAMA2_CHAT_TEMPLATE, &with_new, "</s>", "<s>", true).unwrap();

        assert!(
            with_new_rendered.starts_with(&prev),
            "llama2 template not append-only:\n--- prev ---\n{prev}\n--- with_new ---\n{with_new_rendered}"
        );
        let suffix = &with_new_rendered[prev.len()..];
        assert!(
            suffix.contains("How are you?"),
            "llama2 suffix missing new user content: {suffix:?}"
        );
    }

    #[test]
    fn template_first_turn_has_no_prior_state() {
        let history: Vec<ChatMessage> = vec![];
        let mut with_first = history.clone();
        with_first.push(ChatMessage::user("Hello"));
        let with_first_rendered =
            apply_chat_template(TINYLLAMA_TEMPLATE, &with_first, "</s>", "<s>", true).unwrap();
        assert!(with_first_rendered.contains("Hello"));
        assert!(with_first_rendered.contains("<|assistant|>"));
    }
}
