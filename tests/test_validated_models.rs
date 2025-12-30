use std::path::PathBuf;

use my_onnx::onnx::load::*;
use my_onnx::options::*;
use my_onnx::session::Session;
use my_onnx::session::SessionError;
use my_onnx::tensor::data::CompPolicy;
use my_onnx::tensor::Tensor;

type Result = std::result::Result<(), SessionError>;

fn run_test(model: &str, epsilon: f64, target: Target) -> Result {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/validated")
        .join(model);
    let data_dir = root_dir.join("test_data_set_0");
    let model_path = root_dir.join(format!("{model}.onnx"));
    let input_path = data_dir.join("input_0.pb");
    let output_path = data_dir.join("output_0.pb");

    let input = Tensor::load_from_path(input_path).map_err(SessionError::ModelLoadError)?;
    let session = Session::new(
        &model_path,
        Some(&[input.tensor_type()]),
        &Options::builder().target(target).build(),
    )?;
    let output = session.run(&[input])?;
    let expected = Tensor::load_from_path(output_path).map_err(SessionError::ModelLoadError)?;
    if !output[0].eq_with_epsilon(&expected, epsilon, CompPolicy::Either) {
        // For pretty printing
        assert_eq!(output[0], expected);
    }
    Ok(())
}

#[test]
fn test_mnist12_cpu() -> Result {
    run_test("mnist-12", 1e-3, Target::CPU)
}

#[test]
fn test_resnet18_cpu() -> Result {
    run_test("resnet18-v2-7", 1e-3, Target::CPU)
}

#[test]
fn test_resnet152_cpu() -> Result {
    run_test("resnet152-v2-7", 1e-3, Target::CPU)
}

#[test]
fn test_yolov4_cpu() -> Result {
    run_test("yolov4", 1e-3, Target::CPU)
}

#[test]
fn test_mnist12_cuda() -> Result {
    run_test("mnist-12", 1e-2, Target::CUDA)
}

#[test]
fn test_resnet18_cuda() -> Result {
    run_test("resnet18-v2-7", 1e-2, Target::CUDA)
}

#[test]
fn test_resnet152_cuda() -> Result {
    run_test("resnet152-v2-7", 1e-1, Target::CUDA)
}

