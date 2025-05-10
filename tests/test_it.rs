use my_onnx::onnx::load::*;
use my_onnx::session::{Session, SessionError};
use my_onnx::tensor::Tensor;
use my_onnx::tensor::data::CompPolicy;
use std::path::PathBuf;

type Result = std::result::Result<(), SessionError>;

fn run_test(dir: &str, epsilon: f64) -> Result {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/it")
        .join(dir);
    let model_path = dir.join("model.onnx");
    let input_path = dir.join("input_0.pb");
    let output_path = dir.join("output_0.pb");

    let input = Tensor::load_from_path(input_path).map_err(SessionError::ModelLoadError)?;
    let session = Session::new(&model_path, Some(&[&input.tensor_type()]), 100)?;
    let output = session.run(&[input])?;
    let expected = Tensor::load_from_path(output_path).map_err(SessionError::ModelLoadError)?;
    if !output[0].eq_with_epsilon(&expected, epsilon, CompPolicy::Either) {
        // For pretty printing
        assert_eq!(output[0], expected);
    }
    Ok(())
}

#[test]
fn test_transpose_conv2d() -> Result {
    // TODO: Disable optimizations after they are implemented
    run_test("transpose_conv2d", 1e-3)
}
