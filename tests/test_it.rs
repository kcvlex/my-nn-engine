use my_onnx::onnx::load::*;
use my_onnx::session::{Session, SessionError, Target};
use my_onnx::tensor::data::CompPolicy;
use my_onnx::tensor::Tensor;
use my_onnx::transform::Options;
use std::path::PathBuf;

type Result = std::result::Result<(), SessionError>;

fn run_test(dir: &str, epsilon: f64, options: &Options, nums: (usize, usize)) -> Result {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/it")
        .join(dir);
    let model_path = dir.join("model.onnx");
    let (num_inputs, num_outputs) = nums;
    let inputs = (0..num_inputs)
        .map(|i| {
            Tensor::load_from_path(dir.join(format!("input_{}.pb", i)))
                .map_err(SessionError::ModelLoadError)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let input_types = inputs
        .iter()
        .map(|input| input.tensor_type())
        .collect::<Vec<_>>();
    let session = Session::new(&model_path, Some(&input_types), options, Target::CPU)?;
    let outputs = session.run(&inputs)?;
    let expected = (0..num_outputs)
        .map(|i| {
            Tensor::load_from_path(dir.join(format!("output_{}.pb", i)))
                .map_err(SessionError::ModelLoadError)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    assert_eq!(outputs.len(), expected.len());
    for (output, expected) in outputs.iter().zip(expected.iter()) {
        if !output.eq_with_epsilon(expected, epsilon, CompPolicy::Either) {
            // For pretty printing
            assert_eq!(output, expected);
        }
    }
    Ok(())
}

fn run_test_default(dir: &str, epsilon: f64) -> Result {
    run_test(dir, epsilon, &Options::builder().build(), (1, 1))
}

#[test]
fn test_transpose_conv2d() -> Result {
    run_test_default("transpose_conv2d", 1e-3)
}

#[test]
fn test_add_same_tensor() -> Result {
    run_test(
        "add_same_tensor",
        1e-5,
        &Options::builder().enable_fuse_ops(false).build(),
        (1, 1),
    )
}

#[test]
fn test_elementwise_chain_single() -> Result {
    run_test_default("elementwise_chain_single", 1e-3)
}

#[test]
fn test_elementwise_chain_branch() -> Result {
    run_test(
        "elementwise_chain_branch",
        1e-3,
        &Options::builder().build(),
        (1, 2),
    )
}

#[test]
fn test_elementwise_complex() -> Result {
    run_test_default("elementwise_complex", 1e-3)
}
