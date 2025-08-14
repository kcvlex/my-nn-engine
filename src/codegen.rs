pub mod cpu;

#[derive(Debug)]
pub enum CodeGenError {
    BuilderError(inkwell::builder::BuilderError),
    LLVMError(inkwell::support::LLVMString),
    TargetMachineError(String),
    IntrinsicNotFound(String),
}
