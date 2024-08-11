use std::env;
use my_onnx::load;
use std::io::{Result, Error};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    let model = load::load_from_path(&args[1]).map_err(|e| Error::other(format!("{:?}", e)))?;
    println!("{:?}", model);
    Ok(())
}
