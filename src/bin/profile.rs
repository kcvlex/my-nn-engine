use std::path::PathBuf;

use my_nn_engine::onnx::load::LoadProto;
use my_nn_engine::options::*;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::tensor::Tensor;

fn cuda_profiler_start() {
    unsafe {
        let lib = libloading::Library::new("libcudart.so").expect("failed to load libcudart.so");
        let func: libloading::Symbol<unsafe extern "C" fn() -> i32> = lib
            .get(b"cudaProfilerStart")
            .expect("cudaProfilerStart not found");
        func();
        std::mem::forget(lib);
    }
}

fn cuda_profiler_stop() {
    unsafe {
        let lib = libloading::Library::new("libcudart.so").expect("failed to load libcudart.so");
        let func: libloading::Symbol<unsafe extern "C" fn() -> i32> = lib
            .get(b"cudaProfilerStop")
            .expect("cudaProfilerStop not found");
        func();
        std::mem::forget(lib);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <cpu|cuda> <model_dir> [num_runs] [num_streams] [--profile]",
            args[0]
        );
        eprintln!("  e.g. {} cuda models/validated/resnet18-v2-7 5 2", args[0]);
        std::process::exit(1);
    }

    let enable_profile = args.iter().any(|a| a == "--profile");
    let enable_nhwc = !args.iter().any(|a| a == "--no-nhwc");
    let target = match args[1].as_str() {
        "cpu" => Target::CPU,
        "cuda" => Target::CUDA,
        other => {
            eprintln!("Unknown target: {other}. Use 'cpu' or 'cuda'.");
            std::process::exit(1);
        }
    };
    let model_dir = PathBuf::from(&args[2]);
    let num_runs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let num_streams: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);

    let model_name = model_dir.file_name().unwrap().to_str().unwrap().to_string();
    let model_path = model_dir.join(format!("{model_name}.onnx"));
    if !model_path.exists() {
        let alt = model_dir.join("model.onnx");
        if alt.exists() {
            run(
                &alt,
                &model_dir,
                target,
                num_runs,
                num_streams,
                enable_profile,
                enable_nhwc,
            );
            return;
        }
        eprintln!("Model not found: {:?}", model_path);
        std::process::exit(1);
    }
    run(
        &model_path,
        &model_dir,
        target,
        num_runs,
        num_streams,
        enable_profile,
        enable_nhwc,
    );
}

fn run(
    model_path: &std::path::Path,
    model_dir: &std::path::Path,
    target: Target,
    num_runs: usize,
    num_streams: usize,
    enable_profile: bool,
    enable_nhwc: bool,
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

    let options = match target {
        Target::CPU => Options::builder()
            .target(Target::CPU)
            .profile(enable_profile)
            .enable_nhwc_optimization(Some(enable_nhwc))
            .build(),
        Target::CUDA => Options::builder()
            .target(Target::CUDA)
            .num_cuda_streams(num_streams)
            .profile(enable_profile)
            .enable_nhwc_optimization(Some(enable_nhwc))
            .build(),
    };

    let mut session = Session::new(
        model_path,
        Some(&input_types),
        &options,
        &SessionConfig::default(),
    )
    .unwrap();

    let _ = session.run(&inputs).unwrap();

    if matches!(target, Target::CUDA) {
        cuda_profiler_start();
    }

    eprintln!("Running {num_runs} iterations...");
    let start = std::time::Instant::now();
    for _ in 0..num_runs {
        let _ = session.run(&inputs).unwrap();
    }
    let elapsed = start.elapsed();

    if matches!(target, Target::CUDA) {
        cuda_profiler_stop();
    }
    eprintln!(
        "{num_runs} runs in {:.2?}, avg {:.2?}/run",
        elapsed,
        elapsed / num_runs as u32
    );
}
