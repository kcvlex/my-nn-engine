use my_onnx::codegen::session::{Session, SessionError};
use my_onnx::onnx::load::*;
use my_onnx::tensor::Tensor;
use std::path::PathBuf;

type Result = std::result::Result<(), SessionError>;

fn run_test(model: &str, epsilon: f64) -> Result {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/validated")
        .join(model);
    let data_dir = root_dir.join("test_data_set_0");
    let model_path = root_dir.join(format!("{model}.onnx"));
    let input_path = data_dir.join("input_0.pb");
    let output_path = data_dir.join("output_0.pb");

    let input = Tensor::load_from_path(input_path).map_err(SessionError::ModelLoadError)?;
    let session = Session::new(&model_path, Some(&[&input.ty.dims]), 100)?;
    let output = session.run(&[input])?;
    let expected = Tensor::load_from_path(output_path).map_err(SessionError::ModelLoadError)?;
    if !output[0].eq_with_epsilon(&expected, epsilon) {
        // For pretty printing
        assert_eq!(output[0], expected);
    }
    Ok(())
}

#[test]
fn test_mnist12() -> Result {
    run_test("mnist-12", 1e-3)
}

#[test]
fn test_resnet18() -> Result {
    run_test("resnet18-v2-7", 1e-3)
}

#[test]
fn test_resnet152() -> Result {
    run_test("resnet152-v2-7", 1e-3)
}
