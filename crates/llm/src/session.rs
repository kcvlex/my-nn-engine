use std::path::Path;
use std::sync::Arc;

use itertools::Itertools;
use log::info;
use my_nn_engine::graph::Graph;
use my_nn_engine::onnx::load::ModelLoadError;
use my_nn_engine::options::Options;
use my_nn_engine::session::DeviceBuffer;
use my_nn_engine::session::InitializerBuffers;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::session::SessionError;
use my_nn_engine::session::SessionStateSpec;
use my_nn_engine::tensor::data::TensorData;
use my_nn_engine::tensor::types::ResolvedTensorDims;
use my_nn_engine::tensor::types::SIntType;
use my_nn_engine::tensor::Tensor;
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
    prefill: Option<PrefillSession>,
    tokenizer: Tokenizer,
    config: LlmConfig,
    past_len: usize,
    kind: DecodeKind,
}

struct PrefillSession {
    session: Session,
    prefill_len: usize,
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
            ..SessionConfig::default()
        };
        let session = Session::new(model_path.as_ref(), None, opts, &session_config)?;
        Ok(Self {
            session,
            prefill: None,
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
            ..SessionConfig::default()
        };
        let session = Session::from_graph(graph, opts, &session_config)?;
        Ok(Self {
            session,
            prefill: None,
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

    pub fn for_llama_with_prefill(
        decode_graph: Graph,
        prefill_graph: Graph,
        kv_cache_names: Vec<KVCache>,
        prefill_len: usize,
        tokenizer: Tokenizer,
        opts: &Options,
        max_seq_len: usize,
        eos_token_id: u32,
    ) -> Result<Self, LlmError> {
        let kv_buffers: Vec<(String, String, Arc<DeviceBuffer>, Arc<DeviceBuffer>)> =
            kv_cache_names
                .into_iter()
                .map(
                    |KVCache {
                         k_name,
                         v_name,
                         bytes_per_buffer,
                     }| {
                        let k = Arc::new(
                            DeviceBuffer::alloc_zeroed(bytes_per_buffer).expect("alloc K"),
                        );
                        let v = Arc::new(
                            DeviceBuffer::alloc_zeroed(bytes_per_buffer).expect("alloc V"),
                        );
                        (k_name, v_name, k, v)
                    },
                )
                .collect();

        let make_specs = || -> Vec<SessionStateSpec> {
            kv_buffers
                .iter()
                .flat_map(|(k_name, v_name, k_buf, v_buf)| {
                    [
                        SessionStateSpec {
                            name: k_name.clone(),
                            buffer: Arc::clone(k_buf),
                        },
                        SessionStateSpec {
                            name: v_name.clone(),
                            buffer: Arc::clone(v_buf),
                        },
                    ]
                })
                .collect()
        };

        let initializer_buffers = Arc::new(InitializerBuffers::new());
        let decode_session = Session::from_graph(
            decode_graph,
            opts,
            &SessionConfig {
                session_states: make_specs(),
                initializer_buffers: Some(Arc::clone(&initializer_buffers)),
            },
        )?;
        let prefill_session = Session::from_graph(
            prefill_graph,
            opts,
            &SessionConfig {
                session_states: make_specs(),
                initializer_buffers: Some(initializer_buffers),
            },
        )?;

        Ok(Self {
            session: decode_session,
            prefill: Some(PrefillSession {
                session: prefill_session,
                prefill_len,
            }),
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
        if prompt_ids.is_empty() {
            return Ok(new_ids);
        }

        let prefill_runs = self.prefill.is_some() && self.past_len == 0;

        let last_token = if prefill_runs {
            self.run_prefill(&prompt_ids)?
        } else {
            for &tok in &prompt_ids[..prompt_ids.len() - 1] {
                self.decode_step(tok)?;
            }
            let last_prompt = *prompt_ids.last().unwrap();
            self.decode_step(last_prompt)?
        };
        new_ids.push(last_token);
        if last_token == self.config.eos_token_id {
            return Ok(new_ids);
        }

        let mut last_token = last_token;
        for _ in 1..max_new_tokens {
            if self.config.max_seq_len < self.past_len + 1 {
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

    fn run_prefill_rec(&mut self, prompt_ids: &[u32]) -> Result<Vec<f64>, LlmError> {
        let prefill_len = self.prefill.as_ref().unwrap().prefill_len;
        let total_m = prompt_ids.len();
        assert!(0 < total_m);

        let chunk_size = total_m.min(prefill_len);
        let is_last = chunk_size == total_m;

        let mut padded: Vec<i64> = vec![0; prefill_len];
        for i in 0..chunk_size {
            padded[i] = prompt_ids[i] as i64;
        }
        let positions = (0..prefill_len as i64)
            .map(|i| self.past_len as i64 + i)
            .collect_vec();
        let active_seq_kv = (self.past_len + prefill_len) as i64;
        let inputs = vec![
            make_i64(&[1, prefill_len], padded),
            make_i64(&[prefill_len], positions),
            make_i64(&[], vec![self.past_len as i64]),
            make_i64(&[], vec![active_seq_kv]),
        ];
        let outputs = self.prefill.as_mut().unwrap().session.run(&inputs)?;
        self.past_len += chunk_size;

        if is_last {
            let logits = outputs
                .into_iter()
                .next()
                .ok_or(LlmError::InvalidOutput("model returned no outputs"))?;
            let TensorData::Float(_, data) = logits.data else {
                return Err(LlmError::InvalidOutput("logits must be float"));
            };
            let vocab_size = logits
                .dims
                .last()
                .copied()
                .ok_or(LlmError::InvalidOutput("logits have no dims"))?;
            Ok(data
                .iter()
                .skip((chunk_size - 1) * vocab_size)
                .take(vocab_size)
                .copied()
                .collect_vec())
        } else {
            self.run_prefill_rec(&prompt_ids[chunk_size..])
        }
    }

    fn run_prefill(&mut self, prompt_ids: &[u32]) -> Result<u32, LlmError> {
        let logits = self.run_prefill_rec(prompt_ids)?;
        argmax_logits(&logits)
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
        let TensorData::Float(_, ref logits) = logits.data else {
            return Err(LlmError::InvalidOutput("logits must be float"));
        };
        if logits.is_empty() {
            return Err(LlmError::InvalidOutput("empty logits"));
        }
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

fn argmax_logits(logits: &[f64]) -> Result<u32, LlmError> {
    let (idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    Ok(idx as u32)
}
