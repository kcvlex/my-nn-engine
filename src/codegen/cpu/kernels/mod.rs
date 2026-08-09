//! Hand-written CPU compute kernels invoked from JIT-generated code via
//! `add_global_mapping` (see `session/cpu.rs`). One kernel per file. Each is
//! `#[no_mangle] extern "C"` so its symbol name matches the external
//! declaration the codegen lowering emits.

mod all_reduce;
mod dynquant_i8;
mod ftype;
mod qgemv_i8i8;

pub use all_reduce::mynn_all_reduce;
pub use dynquant_i8::mynn_dynquant_i8;
pub use ftype::FType;
pub use qgemv_i8i8::mynn_qgemv_i8i8;
