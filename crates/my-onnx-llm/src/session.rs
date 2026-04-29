use std::path::Path;

use log::info;
use my_onnx::onnx::load::ModelLoadError;
use my_onnx::options::Options;
use my_onnx::session::Session;
use my_onnx::session::SessionConfig;
use my_onnx::session::SessionError;
use my_onnx::tensor::types::ResolvedTensorType;
use my_onnx::tensor::Tensor;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("session error: {0:?}")]
    Session(SessionError),
    #[error("model load error: {0:?}")]
    ModelLoad(ModelLoadError),
    #[error("not implemented yet: {0}")]
    NotImplemented(&'static str),
}

impl From<SessionError> for LlmError {
    fn from(e: SessionError) -> Self {
        LlmError::Session(e)
    }
}

pub struct LlmSession {
    session: Session,
    past_len: usize,
}

impl LlmSession {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        input_types: &[ResolvedTensorType],
        opts: &Options,
    ) -> Result<Self, LlmError> {
        info!("LlmSession: loading {:?}", model_path.as_ref());
        let session = Session::new(
            model_path,
            Some(input_types),
            opts,
            &SessionConfig::default(),
        )?;
        Ok(Self {
            session,
            past_len: 0,
        })
    }

    pub fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, LlmError> {
        Ok(self.session.run(inputs)?)
    }

    pub fn decode(&mut self, _last_token: i64) -> Result<i64, LlmError> {
        Err(LlmError::NotImplemented(
            "decode-graph plumbing pending (issue #24/#25)",
        ))
    }

    pub fn reset(&mut self) {
        self.past_len = 0;
    }

    pub fn past_len(&self) -> usize {
        self.past_len
    }
}
