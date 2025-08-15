pub mod cpu;
pub mod cuda;

#[derive(Debug)]
pub enum CodeGenError {
    BuilderError(inkwell::builder::BuilderError),
    LLVMError(inkwell::support::LLVMString),
    TargetMachineError(String),
    IntrinsicNotFound(String),
}
