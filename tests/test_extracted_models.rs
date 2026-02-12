use std::path::PathBuf;

use my_onnx::onnx::load::*;
use my_onnx::options::*;
use my_onnx::session::Session;
use my_onnx::session::SessionError;
use my_onnx::tensor::data::CompPolicy;
use my_onnx::tensor::Tensor;

type Result = std::result::Result<(), SessionError>;

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
