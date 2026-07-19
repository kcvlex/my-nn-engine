//! Minimal inference example
//!
//! Usage:
//!   cargo run --release --example run_model -- [cpu|cuda]

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use my_nn_engine::onnx::load::*;
use my_nn_engine::options::Options;
use my_nn_engine::options::PrefetchPolicy;
use my_nn_engine::options::Target;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::tensor::Tensor;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: {} [cpu|cuda]", args[0]);
        return ExitCode::FAILURE;
    }
    let target = match args[1].as_str() {
        "cuda" => Target::CUDA(PrefetchPolicy::Disabled),
        "cpu" => Target::CPU,
        other => {
            eprintln!("unknown target {other:?} (expected `cpu` or `cuda`)");
            return ExitCode::FAILURE;
        }
    };
    match run(target) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(target: Target) -> Result<(), String> {
    let model_name = "resnet18-v2-7";
    let resnet_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("validated")
        .join(model_name);
    let model_path = resnet_dir.join(format!("{model_name}.onnx"));
    let data = Tensor::load_from_path(resnet_dir.join("test_data_set_0").join("input_0.pb"))
        .map_err(|e| format!("load input: {e:?}"))?;
    let inputs = HashMap::from([(String::from("data"), data)]);
    let options = Options::builder().target(target).build();
    let mut session =
        Session::new_with_inputs(&model_path, &inputs, &options, &SessionConfig::default())
            .map_err(|e| format!("session init: {e:?}"))?;
    let outputs = session
        .run_named(inputs)
        .map_err(|e| format!("run: {e:?}"))?;
    for (name, tensor) in &outputs {
        println!("=== output: {name} ===");
        println!("{}", tensor.to_ndarray());
    }
    Ok(())
}
