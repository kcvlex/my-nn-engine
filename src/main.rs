use my_onnx::model::Model;
use my_onnx::optimize::{matmul_a_tb::MatMulAxTB, optimizer::Optimizer};
use std::env;
use std::fs::File;
use std::io::{Error, Result, Write};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    let mut model =
        Model::load_from_path(&args[1]).map_err(|e| Error::other(format!("{:?}", e)))?;
    model
        .graph
        .infer()
        .map_err(|e| Error::other(format!("{:?}", e)))?;
    let mut optimizer = Optimizer::new(String::from("test pass"));
    optimizer.passes.push(Box::new(MatMulAxTB::default()));
    optimizer.run(&mut model.graph);
    let file = File::options()
        .truncate(true)
        .create(true)
        .write(true)
        .open("graph.dot")?;
    let mut writer = std::io::BufWriter::new(file);
    writer.write_all(model.graph.to_dot().as_bytes())?;
    Ok(())
}
