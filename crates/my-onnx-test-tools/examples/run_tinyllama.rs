use std::path::PathBuf;

use my_onnx::onnx::load::*;
use my_onnx::options::*;
use my_onnx::session::Session;
use my_onnx::session::SessionConfig;
use my_onnx::tensor::data::CompPolicy;
use my_onnx::tensor::Tensor;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from("models/hf/tinyllama");
    let model_path = dir.join("model.onnx");
    let data_dir = dir.join("test_data_set_0");

    let mut inputs = Vec::new();
    for i in 0.. {
        let p = data_dir.join(format!("input_{}.pb", i));
        if !p.exists() {
            break;
        }
        inputs.push(Tensor::load_from_path(&p).map_err(|e| format!("load {:?}: {:?}", p, e))?);
    }
    let mut expected = Vec::new();
    for i in 0.. {
        let p = data_dir.join(format!("output_{}.pb", i));
        if !p.exists() {
            break;
        }
        expected.push(Tensor::load_from_path(&p).map_err(|e| format!("load {:?}: {:?}", p, e))?);
    }
    eprintln!(
        "Inputs: {}  Expected outputs: {}",
        inputs.len(),
        expected.len()
    );

    let input_types: Vec<_> = inputs.iter().map(|t| t.tensor_type()).collect();
    let target = match std::env::var("TARGET").as_deref() {
        Ok("CUDA") | Ok("cuda") => Target::CUDA,
        _ => Target::CPU,
    };
    eprintln!("Target: {:?}", target);
    let t0 = std::time::Instant::now();
    let mut session = Session::new(
        &model_path,
        Some(&input_types),
        &Options::builder().target(target).build(),
        &SessionConfig::default(),
    )
    .map_err(|e| format!("session: {:?}", e))?;
    eprintln!("Session::new: {:?}", t0.elapsed());

    // warmup
    eprintln!("About to call run() warmup");
    let _ = session.run(&inputs).map_err(|e| format!("run: {:?}", e))?;
    eprintln!("Warmup run completed");
    let mut ts = Vec::new();
    for _ in 0..5 {
        let t = std::time::Instant::now();
        let _ = session.run(&inputs).map_err(|e| format!("run: {:?}", e))?;
        ts.push(t.elapsed());
    }
    eprintln!("run() x5: {:?}", ts);
    let outputs = session.run(&inputs).map_err(|e| format!("run: {:?}", e))?;
    eprintln!("Got {} outputs", outputs.len());

    for (i, (out, exp)) in outputs.iter().zip(expected.iter()).enumerate() {
        let ok = out.eq_with_epsilon(exp, 0.05, CompPolicy::Either);
        eprintln!(
            "  out[{}] shape={:?} expected={:?} match={}",
            i, out.dims, exp.dims, ok
        );
    }
    Ok(())
}
