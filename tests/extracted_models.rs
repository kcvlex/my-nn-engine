use std::path::Path;
use std::path::PathBuf;

use my_nn_engine::onnx::load::*;
use my_nn_engine::options::*;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::session::SessionError;
use my_nn_engine::tensor::data::CompPolicy;
use my_nn_engine::tensor::Tensor;

type Result = std::result::Result<(), SessionError>;

fn count_pb_files(dir: &Path, prefix: &str) -> usize {
    (0..)
        .take_while(|i| dir.join(format!("{}_{}.pb", prefix, i)).exists())
        .count()
}

fn run_test(root_dir: &Path, epsilon: f64, target: Target) -> Result {
    let model_path = root_dir.join("model.onnx");

    let num_inputs = count_pb_files(root_dir, "input");
    let num_outputs = count_pb_files(root_dir, "output");

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
    let mut session = Session::new(
        &model_path,
        Some(&input_types),
        &Options::builder().target(target).build(),
        &SessionConfig::default(),
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
            assert_eq!(output, expected);
        }
    }
    Ok(())
}

/// Run via: EXTRACTED_MODEL_DIR=path/to/dir cargo test --test extracted_models -- --ignored
///
/// Optional env vars:
///   EPSILON  - comparison tolerance (default: 1e-2)
///   TARGET   - "cpu" or "cuda" (default: "cpu")
#[ignore]
#[test]
fn test_extracted_model() -> Result {
    let dir = PathBuf::from(
        std::env::var("EXTRACTED_MODEL_DIR").expect("EXTRACTED_MODEL_DIR must be set"),
    );
    let epsilon: f64 = std::env::var("EPSILON")
        .unwrap_or_else(|_| "1e-2".to_string())
        .parse()
        .expect("invalid EPSILON value");
    let target = match std::env::var("TARGET").as_deref() {
        Ok("cuda") => Target::CUDA,
        _ => Target::CPU,
    };
    run_test(&dir, epsilon, target)
}
