use std::path::Path;
use std::sync::Arc;

use log::info;
use my_onnx::onnx::load::ModelLoadError;
use my_onnx::onnx::model::Graph;
use my_onnx::options::Options;
use my_onnx::session::DeviceBuffer;
use my_onnx::session::Session;
use my_onnx::session::SessionConfig;
use my_onnx::session::SessionError;
use my_onnx::session::SessionStateSpec;
use my_onnx::tensor::data::TensorData;
use my_onnx::tensor::types::FloatType;
use my_onnx::tensor::types::ResolvedTensorDims;
use my_onnx::tensor::types::SIntType;
use my_onnx::tensor::Tensor;
use tokenizers::Tokenizer;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("session error: {0:?}")]
    Session(SessionError),
    #[error("model load error: {0:?}")]
    ModelLoad(ModelLoadError),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("invalid output: {0}")]
    InvalidOutput(&'static str),
    #[error("not implemented yet: {0}")]
    NotImplemented(&'static str),
}

impl From<SessionError> for LlmError {
    fn from(e: SessionError) -> Self {
        LlmError::Session(e)
    }
}

#[derive(Debug)]
pub struct KVCache {
    pub k_name: String,
    pub v_name: String,
    pub bytes_per_buffer: usize,
}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub max_seq_len: usize,
    pub eos_token_id: u32,
    pub session_states: Vec<SessionStateSpec>,
}

/// Per-step input convention. Selected by which constructor is used.
enum DecodeKind {
    /// `[input_ids: i64[1], past_len: i64[]]` — used by simple test fixtures.
    Mock,
    /// `[input_ids: i64[1,1], position_id: i64[1], past_len: i64[], active_seq_kv: i64[]]`
    /// + SessionState K/V caches — produced by [`crate::build_llama`].
    Llama,
}

pub struct LlmSession {
    session: Session,
    tokenizer: Tokenizer,
    config: LlmConfig,
    past_len: usize,
    kind: DecodeKind,
}

impl LlmSession {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        tokenizer_path: P,
        opts: &Options,
        config: LlmConfig,
    ) -> Result<Self, LlmError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| LlmError::Tokenizer(format!("{:?}", e)))?;
        Self::with_tokenizer(model_path, tokenizer, opts, config)
    }

    pub fn with_tokenizer<P: AsRef<Path>>(
        model_path: P,
        tokenizer: Tokenizer,
        opts: &Options,
        config: LlmConfig,
    ) -> Result<Self, LlmError> {
        info!("LlmSession: loading {:?}", model_path.as_ref());
        let session_config = SessionConfig {
            session_states: config.session_states.clone(),
        };
        let session = Session::new(model_path.as_ref(), None, opts, &session_config)?;
        Ok(Self {
            session,
            tokenizer,
            config,
            past_len: 0,
            kind: DecodeKind::Mock,
        })
    }

    /// Construct a session for a LLaMA-family graph built via [`crate::build_llama`].
    /// Wires the K/V cache inputs as zero-initialized `SessionState` buffers and
    /// sets the per-step input convention to `[input_ids, position_id, past_len, active_seq_kv]`.
    pub fn for_llama(
        graph: Graph,
        kv_cache_names: Vec<KVCache>,
        tokenizer: Tokenizer,
        opts: &Options,
        max_seq_len: usize,
        eos_token_id: u32,
    ) -> Result<Self, LlmError> {
        let session_config = SessionConfig {
            session_states: kv_cache_names
                .into_iter()
                .flat_map(
                    |KVCache {
                         k_name,
                         v_name,
                         bytes_per_buffer,
                     }| {
                        let k_buf = Arc::new(
                            DeviceBuffer::alloc_zeroed(bytes_per_buffer).expect("alloc K cache"),
                        );
                        let v_buf = Arc::new(
                            DeviceBuffer::alloc_zeroed(bytes_per_buffer).expect("alloc V cache"),
                        );
                        [
                            SessionStateSpec {
                                name: k_name,
                                buffer: k_buf,
                            },
                            SessionStateSpec {
                                name: v_name,
                                buffer: v_buf,
                            },
                        ]
                    },
                )
                .collect(),
        };
        let session = Session::from_graph(graph, opts, &session_config)?;
        Ok(Self {
            session,
            tokenizer,
            config: LlmConfig {
                max_seq_len,
                eos_token_id,
                session_states: vec![],
            },
            past_len: 0,
            kind: DecodeKind::Llama,
        })
    }

    pub fn reset(&mut self) {
        self.past_len = 0;
    }

    pub fn past_len(&self) -> usize {
        self.past_len
    }

    pub fn generate_ids(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, LlmError> {
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| LlmError::Tokenizer(format!("{:?}", e)))?;
        let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();

        let mut new_ids: Vec<u32> = Vec::new();
        let Some(&last_prompt) = prompt_ids.last() else {
            return Ok(new_ids);
        };
        for &tok in &prompt_ids[..prompt_ids.len() - 1] {
            self.decode_step(tok)?;
        }
        let mut last_token = last_prompt;

        for _ in 0..max_new_tokens {
            if self.past_len + 1 > self.config.max_seq_len {
                break;
            }
            let next = self.decode_step(last_token)?;
            new_ids.push(next);
            last_token = next;
            if next == self.config.eos_token_id {
                break;
            }
        }

        Ok(new_ids)
    }

    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> Result<String, LlmError> {
        let new_ids = self.generate_ids(prompt, max_new_tokens)?;
        self.tokenizer
            .decode(&new_ids, false)
            .map_err(|e| LlmError::Tokenizer(format!("{:?}", e)))
    }

    fn decode_step(&mut self, token: u32) -> Result<u32, LlmError> {
        let inputs = match self.kind {
            DecodeKind::Mock => vec![
                make_i64(&[1], vec![token as i64]),
                make_i64(&[], vec![self.past_len as i64]),
            ],
            DecodeKind::Llama => vec![
                make_i64(&[1, 1], vec![token as i64]),
                make_i64(&[1], vec![self.past_len as i64]),
                make_i64(&[], vec![self.past_len as i64]),
                make_i64(&[], vec![self.past_len as i64 + 1]),
            ],
        };
        let outputs = self.session.run(&inputs)?;
        let logits = outputs
            .first()
            .ok_or(LlmError::InvalidOutput("model returned no outputs"))?;
        let next = argmax_logits(logits)?;
        self.past_len += 1;
        Ok(next)
    }
}

fn make_i64(dims: &[usize], values: Vec<i64>) -> Tensor {
    Tensor::new(
        ResolvedTensorDims::new(dims),
        TensorData::SInt(SIntType::I64, values),
    )
    .unwrap()
}

fn argmax_logits(logits: &Tensor) -> Result<u32, LlmError> {
    let TensorData::Float(FloatType::F32, ref data) = logits.data else {
        return Err(LlmError::InvalidOutput("logits must be f32"));
    };
    if data.is_empty() {
        return Err(LlmError::InvalidOutput("empty logits"));
    }

    let (idx, _) = data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    Ok(idx as u32)
}
