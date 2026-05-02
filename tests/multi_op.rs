mod common;

use std::path::PathBuf;

use my_nn_engine::onnx::load::*;
use my_nn_engine::options::*;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::session::SessionError;
use my_nn_engine::tensor::data::CompPolicy;
use my_nn_engine::tensor::Tensor;

type Result = std::result::Result<(), SessionError>;

fn run_test(dir: &str, epsilon: f64, options: &Options, nums: (usize, usize)) -> Result {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/test/multi_op")
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
    let mut session = Session::new(
        &model_path,
        Some(&input_types),
        options,
        &SessionConfig::default(),
    )?;
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
fn test_transpose_conv2d_cpu() -> Result {
    run_test_default("transpose_conv2d", 1e-3)
}

#[test]
#[cfg(feature = "cuda")]
fn test_transpose_conv2d_cuda() -> Result {
    run_test(
        "transpose_conv2d",
        1e-3,
        &Options::builder().target(Target::CUDA).build(),
        (1, 1),
    )
}

#[test]
#[cfg(feature = "cuda")]
fn test_add_same_tensor_cpu() -> Result {
    run_test(
        "add_same_tensor",
        1e-5,
        &Options::builder().enable_fuse_ops(false).build(),
        (1, 1),
    )
}

#[test]
#[cfg(feature = "cuda")]
fn test_add_same_tensor_cuda() -> Result {
    run_test(
        "add_same_tensor",
        1e-5,
        &Options::builder()
            .target(Target::CUDA)
            .enable_fuse_ops(false)
            .build(),
        (1, 1),
    )
}

#[test]
fn test_elementwise_chain_single_cpu() -> Result {
    run_test_default("elementwise_chain_single", 1e-3)
}

#[test]
#[cfg(feature = "cuda")]
fn test_elementwise_chain_single_cuda() -> Result {
    run_test(
        "elementwise_chain_single",
        1e-3,
        &Options::builder().target(Target::CUDA).build(),
        (1, 1),
    )
}

#[test]
fn test_elementwise_chain_branch_cpu() -> Result {
    run_test(
        "elementwise_chain_branch",
        1e-3,
        &Options::builder().build(),
        (1, 2),
    )
}

#[test]
#[cfg(feature = "cuda")]
fn test_elementwise_chain_branch_cuda() -> Result {
    run_test(
        "elementwise_chain_branch",
        1e-3,
        &Options::builder().target(Target::CUDA).build(),
        (1, 2),
    )
}

#[test]
fn test_elementwise_complex_cpu() -> Result {
    run_test_default("elementwise_complex", 1e-3)
}

#[test]
#[cfg(feature = "cuda")]
fn test_elementwise_complex_cuda() -> Result {
    run_test(
        "elementwise_complex",
        1e-3,
        &Options::builder().target(Target::CUDA).build(),
        (1, 1),
    )
}

#[test]
fn test_transpose_split_cpu() -> Result {
    run_test("transpose_split", 0.0, &Options::builder().build(), (1, 3))
}

#[test]
#[cfg(feature = "cuda")]
fn test_transpose_split_cuda() -> Result {
    run_test(
        "transpose_split",
        0.0,
        &Options::builder().target(Target::CUDA).build(),
        (1, 3),
    )
}

#[test]
fn test_transpose_concat_cpu() -> Result {
    run_test("transpose_concat", 0.0, &Options::builder().build(), (2, 1))
}

#[test]
#[cfg(feature = "cuda")]
fn test_transpose_concat_cuda() -> Result {
    run_test(
        "transpose_concat",
        0.0,
        &Options::builder().target(Target::CUDA).build(),
        (2, 1),
    )
}

#[test]
fn test_transpose_matmul_and_someone_cpu() -> Result {
    run_test(
        "transpose_matmul_and_someone",
        1e-2,
        &Options::builder().target(Target::CPU).build(),
        (2, 2),
    )
}

#[test]
#[cfg(feature = "cuda")]
fn test_transpose_matmul_and_someone_cuda() -> Result {
    run_test(
        "transpose_matmul_and_someone",
        1e-2,
        &Options::builder().target(Target::CUDA).build(),
        (2, 2),
    )
}

#[test]
fn test_div_broadcast_scalar_cpu() -> Result {
    run_test_default("div_broadcast_scalar", 1e-5)
}

#[test]
fn test_gpt2_attention_cpu() -> Result {
    run_test(
        "gpt2_attention",
        1e-4,
        &Options::builder().target(Target::CPU).build(),
        (3, 1),
    )
}

#[test]
#[cfg(feature = "cuda")]
fn test_gpt2_attention_cuda() -> Result {
    run_test(
        "gpt2_attention",
        1e-4,
        &Options::builder().target(Target::CUDA).build(),
        (3, 1),
    )
}
