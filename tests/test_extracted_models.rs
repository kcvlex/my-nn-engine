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
        .join("models/extracted")
        .join(model);
    let model_path = root_dir.join("model.onnx");
    let input_path = root_dir.join("input_0.pb");
    let output_path = root_dir.join("output_0.pb");

    let input = Tensor::load_from_path(input_path).map_err(SessionError::ModelLoadError)?;
    let session = Session::new(
        &model_path,
        Some(&[input.tensor_type()]),
        &Options::builder().target(target).build(),
    )?;
    let output = session.run(&[input])?;
    let expected = Tensor::load_from_path(output_path).map_err(SessionError::ModelLoadError)?;
    if !output[0].eq_with_epsilon(&expected, epsilon, CompPolicy::Either) {
        if true {
            assert_eq!(output[0], expected);
        } else {
            let left = output[0]
                .clone()
                .data
                .into_1d_tensor()
                .to_1d_floats()
                .unwrap();
            let right = expected
                .clone()
                .data
                .into_1d_tensor()
                .to_1d_floats()
                .unwrap();
            // For pretty printing
            assert_eq!(&left[..10], &right[..10]);
        }
    }
    Ok(())
}

#[ignore]
#[test]
fn test_resnet18_until() -> Result {
    run_test("resnet18-v2-7/until_pool1_fwd", 1e-2, Target::CUDA)
}

#[ignore]
#[test]
fn test_yolov4_until() -> Result {
    run_test("yolov4/until_lambda_5_add", 5e-1, Target::CUDA)
}
