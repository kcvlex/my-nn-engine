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
    let args: Vec<_> = env::args().collect();
    if false {
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
        if false {
            let code = jit::sample(&mut jit)?;
            let code = unsafe { core::mem::transmute::<*const u8, fn(()) -> isize>(code) };
            let res = code(());
            println!("res={}", res);

            let (code, output_id) = jit::sample3(&mut jit)?;
            let code = unsafe { core::mem::transmute::<*const u8, fn(()) -> isize>(code) };
            code(());

            let dim = ResolvedTensorDims::new(vec![2, 2, 2]);
            let buffer = jit.get_finalized_data(output_id);
            let mut data = Vec::new();
            for i in 0..dim.size() {
                let offset = i * 8;
                let value = f64::from_le_bytes([
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                    buffer[offset + 3],
                    buffer[offset + 4],
                    buffer[offset + 5],
                    buffer[offset + 6],
                    buffer[offset + 7],
                ]);
                data.push(value);
            }
            let output = Tensor::new(dim, TensorData::F64(data))
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
            println!("{:?}", output);
        } else {
            let mut model =
                load::load_from_path(&args[1]).map_err(|e| Error::other(format!("{:?}", e)))?;
            model
                .graph
                .infer()
                .map_err(|e| Error::other(format!("{:?}", e)))?;
            let code = jit::GraphCompiler::compile(&mut jit, &model.graph)
                .map_err(|e| Error::other(format!("{:?}", e)))?;
            {
                let file = File::options()
                    .truncate(true)
                    .create(true)
                    .write(true)
                    .open("graph.dot")?;
                let mut writer = std::io::BufWriter::new(file);
                writer.write_all(model.graph.to_dot().as_bytes())?;
            }
            let code = unsafe { core::mem::transmute::<*const u8, fn(*const u8, *const u8)>(code) };
            let input: Tensor =
                ndarray::array!([[[1.0, -2.0], [42.0, 4.0]], [[-5.0, 6.0], [-7.0, -8.0]],])
                    .try_into()
                    .map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))
                    })?;
            let output = input.clone();
            println!("input={:?}", input);
            let ty = output.ty.clone();
            let input = input.to_bytes();
            let output = output.to_bytes();
            code(input.as_ptr(), output.as_ptr());
            let output = Tensor::from_bytes(ty, &output)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
            println!("output={:?}", output);
        }
    }
    Ok(())
}
