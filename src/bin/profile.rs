use std::path::PathBuf;

use my_onnx::onnx::load::LoadProto;
use my_onnx::options::*;
use my_onnx::session::Session;
use my_onnx::tensor::Tensor;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir> [num_runs] [num_streams]", args[0]);
        eprintln!("  model_dir: e.g. models/validated/resnet18-v2-7");
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let num_runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let num_streams: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);

    let model_name = model_dir.file_name().unwrap().to_str().unwrap().to_string();
    let model_path = model_dir.join(format!("{model_name}.onnx"));
    if !model_path.exists() {
        // Try model.onnx (e.g. GPT2)
        let alt = model_dir.join("model.onnx");
        if alt.exists() {
            run(&alt, &model_dir, num_runs, num_streams);
            return;
        }
        eprintln!("Model not found: {:?}", model_path);
        std::process::exit(1);
    }
    run(&model_path, &model_dir, num_runs, num_streams);
}

fn run(
    model_path: &std::path::Path,
    model_dir: &std::path::Path,
    num_runs: usize,
    num_streams: usize,
) {
    let data_dir = model_dir.join("test_data_set_0");

    let mut i = 0;
    let mut inputs = Vec::new();
    loop {
        let path = data_dir.join(format!("input_{i}.pb"));
        if !path.exists() {
            break;
        }
        inputs.push(Tensor::load_from_path(path).unwrap());
        i += 1;
    }
    let input_types: Vec<_> = inputs.iter().map(|t| t.tensor_type()).collect();

    let session = Session::new(
        model_path,
        Some(&input_types),
        &Options::builder()
            .target(Target::CUDA)
            .num_cuda_streams(num_streams)
            .build(),
    )
    .unwrap();

    // Warmup
    let _ = session.run(&inputs).unwrap();

    eprintln!("Running {num_runs} iterations...");
    let start = std::time::Instant::now();
    for _ in 0..num_runs {
        let _ = session.run(&inputs).unwrap();
    }
    let elapsed = start.elapsed();
    eprintln!(
        "{num_runs} runs in {:.2?}, avg {:.2?}/run",
        elapsed,
        elapsed / num_runs as u32
    );
}
