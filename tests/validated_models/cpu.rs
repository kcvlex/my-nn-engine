use std::path::PathBuf;

use my_onnx::onnx::load::*;
use my_onnx::options::*;
use my_onnx::session::Session;
use my_onnx::session::SessionError;
use my_onnx::tensor::data::CompPolicy;
use my_onnx::tensor::Tensor;
use serial_test::serial;

type Result = std::result::Result<(), SessionError>;

fn run_test(
    model: &str,
    epsilon: f64,
    nums: (usize, usize),
    model_filename: Option<&str>,
) -> Result {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/validated")
        .join(model);
    let data_dir = root_dir.join("test_data_set_0");
    let model_path = root_dir.join(model_filename.unwrap_or(format!("{model}.onnx").as_str()));

    let (num_inputs, num_outputs) = nums;
    let inputs = (0..num_inputs)
        .map(|i| {
            Tensor::load_from_path(data_dir.join(format!("input_{}.pb", i)))
                .map_err(SessionError::ModelLoadError)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let input_types = inputs
        .iter()
        .map(|input| input.tensor_type())
        .collect::<Vec<_>>();
    let session = Session::new(
        &model_path,
        Some(&input_types),
        &Options::builder().target(Target::CPU).build(),
    )?;
    let outputs = session.run(&inputs)?;
    let expected = (0..num_outputs)
        .map(|i| {
            Tensor::load_from_path(data_dir.join(format!("output_{}.pb", i)))
                .map_err(SessionError::ModelLoadError)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for (i, (output, expected)) in outputs.iter().zip(expected.iter()).enumerate() {
        if !output.eq_with_epsilon(expected, epsilon, CompPolicy::Either) {
            dbg!(i);
            assert_eq!(output, expected);
        }
    }
    Ok(())
}

#[test]
#[serial]
fn test_mnist12() -> Result {
    run_test("mnist-12", 1e-3, (1, 1), None)
}

#[test]
#[serial]
fn test_resnet18() -> Result {
    run_test("resnet18-v2-7", 1e-3, (1, 1), None)
}

#[test]
#[serial]
fn test_resnet152() -> Result {
    run_test("resnet152-v2-7", 1e-3, (1, 1), None)
}

#[test]
#[serial]
fn test_yolov4() -> Result {
    run_test("yolov4", 1e-3, (1, 1), None)
}

#[test]
#[serial]
fn test_bertsquad12() -> Result {
    run_test("bertsquad-12", 1e-2, (4, 3), None)
}

#[test]
#[serial]
fn test_gpt2() -> Result {
    run_test("GPT2", 1e-2, (1, 13), Some("model.onnx"))
}
