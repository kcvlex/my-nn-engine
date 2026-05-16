pub mod cpu;
pub mod cuda;

#[derive(Debug, thiserror::Error)]
pub enum CodeGenError {
    #[error("builder error: {0:?}")]
    BuilderError(#[from] inkwell::builder::BuilderError),
    #[error("LLVM error: {0}")]
    LLVMError(inkwell::support::LLVMString),
    #[error("target machine error: {0}")]
    TargetMachineError(String),
    #[error("intrinsic not found: {0}")]
    IntrinsicNotFound(String),
    #[error("cuda build error: {0}")]
    CudaBuildError(#[from] cuda::BuildError),
}
