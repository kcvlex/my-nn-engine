use my_onnx::codegen::jit;
use my_onnx::load;
use std::env;
use std::io::{Error, Result};

fn main() -> Result<()> {
    if false {
        let args: Vec<_> = env::args().collect();
        let model = load::load_from_path(&args[1]).map_err(|e| Error::other(format!("{:?}", e)))?;
        println!("{:?}", model);
    } else {
        let mut jit = jit::JIT::default();
        let code = jit::sample(&mut jit)?;
        let code = unsafe { core::mem::transmute::<*const u8, fn(()) -> isize>(code) };
        let res = code(());
        println!("res={}", res);
    }
    Ok(())
}
