use my_onnx::codegen::jit;
use my_onnx::{
    load,
    tensor::{
        resolved_dimensions::ResolvedTensorDims,
        tensor::{Tensor, TensorData},
    },
};
use std::env;
use std::fs::File;
use std::io::{Error, Result, Write};

fn main() -> Result<()> {
    if true {
        let args: Vec<_> = env::args().collect();
        let mut model =
            load::load_from_path(&args[1]).map_err(|e| Error::other(format!("{:?}", e)))?;
        model
            .graph
            .infer()
            .map_err(|e| Error::other(format!("{:?}", e)))?;
        let file = File::options()
            .truncate(true)
            .create(true)
            .write(true)
            .open("graph.dot")?;
        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(model.graph.to_dot().as_bytes())?;
    } else {
        let mut jit = jit::JIT::default();
        let code = jit::sample(&mut jit)?;
        let code = unsafe { core::mem::transmute::<*const u8, fn(()) -> isize>(code) };
        let res = code(());
        println!("res={}", res);

        let (code, output_id) = jit::sample2(&mut jit)?;
        let code = unsafe { core::mem::transmute::<*const u8, fn(()) -> isize>(code) };
        code(());

        let dim = ResolvedTensorDims::new(vec![2, 2, 2]);
        let buffer = jit.get_finalized_data(output_id);
        let mut data = Vec::new();
        for i in 0..dim.size() {
            let offset = i * 4;
            let value = f32::from_le_bytes([
                buffer[offset],
                buffer[offset + 1],
                buffer[offset + 2],
                buffer[offset + 3],
            ]);
            data.push(value);
        }
        let output = Tensor::new(dim, TensorData::F32(data))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
        println!("{:?}", output);
    }
    Ok(())
}
