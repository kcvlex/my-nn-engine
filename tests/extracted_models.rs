use std::path::PathBuf;

use my_onnx::onnx::load::*;
use my_onnx::options::*;
use my_onnx::session::Session;
use my_onnx::session::SessionError;
use my_onnx::tensor::data::CompPolicy;
use my_onnx::tensor::Tensor;

type Result = std::result::Result<(), SessionError>;

/// Run a test on an extracted model from an absolute directory path.
/// This is useful for testing models extracted by the binary search tool.
///
/// # Arguments
/// * `extract_dir` - Absolute path to directory containing model.onnx and test data
/// * `epsilon` - Maximum allowed difference between outputs
/// * `target` - Target device (CPU or CUDA)
///
/// # Example
/// ```ignore
/// run_test_from_dir("/tmp/bertsquad_debug/node_416", 1e-2, Target::CPU)?;
/// ```
fn run_test_from_dir(extract_dir: &str, epsilon: f64, target: Target) -> Result {
    let root_dir = PathBuf::from(extract_dir);
    let model_path = root_dir.join("model.onnx");

    // Auto-detect number of inputs and outputs
    let mut num_inputs = 0;
    while root_dir.join(format!("input_{}.pb", num_inputs)).exists() {
        num_inputs += 1;
    }

    let mut num_outputs = 0;
    while root_dir.join(format!("output_{}.pb", num_outputs)).exists() {
        num_outputs += 1;
    }

    if num_inputs == 0 {
        panic!("No input files found in {}", extract_dir);
    }
    if num_outputs == 0 {
        panic!("No output files found in {}", extract_dir);
    }

    let inputs = (0..num_inputs)
        .map(|i| {
            Tensor::load_from_path(root_dir.join(format!("input_{}.pb", i)))
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
        &Options::builder().target(target).build(),
    )?;
    let outputs = session.run(&inputs)?;
    let expected = (0..num_outputs)
        .map(|i| {
            Tensor::load_from_path(root_dir.join(format!("output_{}.pb", i)))
                .map_err(SessionError::ModelLoadError)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    assert_eq!(outputs.len(), expected.len());
    for (i, (output, expected)) in outputs.iter().zip(expected.iter()).enumerate() {
        if !output.eq_with_epsilon(expected, epsilon, CompPolicy::Either) {
            eprintln!("Output {} mismatch", i);
            eprintln!("Expected shape: {:?}", expected.dims);
            eprintln!("Got shape: {:?}", output.dims);
            // For pretty printing
            assert_eq!(output, expected);
        }
    }
    Ok(())
}

fn run_test(model: &str, epsilon: f64, target: Target, nums: (usize, usize)) -> Result {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/extracted")
        .join(model);
    let model_path = root_dir.join("model.onnx");

    let (num_inputs, num_outputs) = nums;
    let inputs = (0..num_inputs)
        .map(|i| {
            Tensor::load_from_path(root_dir.join(format!("input_{}.pb", i)))
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
        &Options::builder().target(target).build(),
    )?;
    let outputs = session.run(&inputs)?;
    let expected = (0..num_outputs)
        .map(|i| {
            Tensor::load_from_path(root_dir.join(format!("output_{}.pb", i)))
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
    // if !output[0].eq_with_epsilon(&expected, epsilon, CompPolicy::Either) {
    //     if true {
    //         assert_eq!(output[0], expected);
    //     } else {
    //         let left = output[0]
    //             .clone()
    //             .data
    //             .into_1d_tensor()
    //             .to_1d_floats()
    //             .unwrap();
    //         let right = expected
    //             .clone()
    //             .data
    //             .into_1d_tensor()
    //             .to_1d_floats()
    //             .unwrap();
    //         // For pretty printing
    //         assert_eq!(&left[..10], &right[..10]);
    //     }
    // }
    // Ok(())
}

#[ignore]
#[test]
#[cfg(feature = "cuda")]
fn test_resnet18_until() -> Result {
    run_test("resnet18-v2-7/until_pool1_fwd", 1e-2, Target::CUDA, (1, 1))
}

#[ignore]
#[test]
#[cfg(feature = "cuda")]
fn test_yolov4_until() -> Result {
    run_test("yolov4/until_lambda_5_add", 5e-1, Target::CUDA, (1, 1))
}

#[test]
fn test_bert_gather_cpu() -> Result {
    run_test(
        "bertsquad-12/until_embeddings_gatherv2",
        1e-2,
        Target::CPU,
        (4, 1),
    )
}

#[test]
fn test_bert_add_0_cpu() -> Result {
    run_test(
        "bertsquad-12/until_embeddings_add_0",
        1e-2,
        Target::CPU,
        (4, 1),
    )
}

#[test]
fn test_bert_reshape_3_cpu() -> Result {
    run_test(
        "bertsquad-12/until_embeddings_reshape_3",
        1e-2,
        Target::CPU,
        (4, 1),
    )
}

#[test]
fn test_bert_node_416_softmax_cpu() -> Result {
    // Node 416: First failing node found by binary search
    // bert/encoder/layer_0/attention/self/Softmax
    run_test("bertsquad-12/node_416_softmax", 1e-2, Target::CPU, (4, 1))
}

#[test]
fn test_bert_node_1148_squeeze_cpu() -> Result {
    // Node 1148: Second failing node found by binary search (after Softmax fix)
    // strided_slice_1__476 - Squeeze operation
    // Output shape: () - scalar
    run_test("bertsquad-12/node_1148_squeeze", 1e-2, Target::CPU, (4, 1))
}

#[test]
fn test_bert_node_1163_squeeze_unstack_cpu() -> Result {
    // Node 1163: Third failing node (after Softmax and Squeeze scalar fixes)
    // unstack__490 - Squeeze operation in unstack context
    // Output shape: (1, 256)
    // Bug: Produces incorrect values (-5.05 vs expected -6.10)
    run_test(
        "bertsquad-12/node_1163_squeeze_unstack",
        1e-2,
        Target::CPU,
        (4, 1),
    )
}

#[ignore]
#[test]
#[cfg(feature = "cuda")]
fn test_bert_until() -> Result {
    run_test(
        "bertsquad-12/until_embeddings_batchnorm_add_1",
        1e-2,
        Target::CUDA,
        (4, 1),
    )
}

#[ignore]
#[test]
#[cfg(feature = "cuda")]
fn test_gpt_until() -> Result {
    run_test("GPT2/until_output2_277", 1e-2, Target::CUDA, (1, 2))
}
