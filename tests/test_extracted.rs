use my_onnx::onnx::load::*;
use my_onnx::session::{Session, SessionError};
use my_onnx::tensor::Tensor;
use std::path::PathBuf;

type Result = std::result::Result<(), SessionError>;

fn run_test(path: &str, epsilon: f64) -> Result {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/extracted")
        .join(path);
    let input_path = root_dir.join("input.pb");
    let model_path = root_dir.join("model.onnx");
    let output_path = root_dir.join("output.pb");

    let input = Tensor::load_from_path(input_path).map_err(SessionError::ModelLoadError)?;
    let session = Session::new(&model_path, Some(&[&input.tensor_type()]), 100)?;
    let output = session.run(&[input])?;
    let expected = Tensor::load_from_path(output_path).map_err(SessionError::ModelLoadError)?;
    if !output[0].eq_with_epsilon(&expected, epsilon) {
        let starts = [0, 0, 0, 0];
        let ends = [1, 1, 1, 10];
        // For pretty printing
        assert_eq!(
            output[0].slices(&starts, &ends),
            expected.slices(&starts, &ends)
        );
    }
    Ok(())
}

#[test]
fn test_yolov4_until_conv2d() -> Result {
    run_test("yolov4/until_conv2d", 1e-3)
}

#[ignore]
#[test]
fn test_yolov4_until_lambda_1_mul() -> Result {
    run_test("yolov4/until_lambda_1_mul", 1e-3)
}
