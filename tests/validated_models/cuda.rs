use my_onnx::options::Target;
use my_onnx::session::SessionError;

use super::common::run_validated_model;

type Result = std::result::Result<(), SessionError>;

fn run_test(
    model: &str,
    epsilon: f64,
    nums: (usize, usize),
    model_filename: Option<&str>,
) -> Result {
    run_validated_model(model, epsilon, nums, model_filename, Target::CUDA)
}

#[test]
fn test_mnist12() -> Result {
    run_test("mnist-12", 1e-2, (1, 1), None)
}

#[test]
fn test_resnet18() -> Result {
    run_test("resnet18-v2-7", 1e-2, (1, 1), None)
}

#[test]
fn test_resnet152() -> Result {
    run_test("resnet152-v2-7", 1e-1, (1, 1), None)
}

#[test]
fn test_yolov4() -> Result {
    run_test("yolov4", 1.0, (1, 1), None)
}

#[test]
fn test_bertsquad12() -> Result {
    run_test("bertsquad-12", 1e-2, (4, 3), None)
}

#[test]
fn test_gpt2() -> Result {
    run_test("GPT2", 1e-2, (1, 13), Some("model.onnx"))
}

#[test]
fn test_mobilenetv2() -> Result {
    run_test("mobilenetv2-12", 1e-2, (1, 1), None)
}

#[test]
fn test_efficientnet_lite4_11() -> Result {
    run_test(
        "efficientnet-lite4-11",
        1e-2,
        (1, 1),
        Some("efficientnet-lite4.onnx"),
    )
}
