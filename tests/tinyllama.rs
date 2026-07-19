#![cfg(feature = "local")]

mod common;

use std::path::PathBuf;

use my_nn_engine::onnx::load::LoadProto;
use my_nn_engine::options::Options;
use my_nn_engine::options::PrefetchPolicy;
use my_nn_engine::options::Target;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::session::SessionError;
use my_nn_engine::tensor::data::CompPolicy;
use my_nn_engine::tensor::Tensor;

type TestResult = Result<(), SessionError>;

fn run_tinyllama(target: Target) -> TestResult {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/hf/tinyllama");
    let model_path = dir.join("model.onnx");
    let data_dir = dir.join("test_data_set_0");

    let mut inputs = Vec::new();
    for i in 0.. {
        let p = data_dir.join(format!("input_{}.pb", i));
        if !p.exists() {
            break;
        }
        inputs.push(Tensor::load_from_path(&p).map_err(SessionError::ModelLoadError)?);
    }
    let mut expected = Vec::new();
    for i in 0.. {
        let p = data_dir.join(format!("output_{}.pb", i));
        if !p.exists() {
            break;
        }
        expected.push(Tensor::load_from_path(&p).map_err(SessionError::ModelLoadError)?);
    }

    let input_types: Vec<_> = inputs.iter().map(|t| t.tensor_type()).collect();
    let opt = Options::builder().target(target).build();
    let mut session = Session::new(
        &model_path,
        Some(&input_types),
        &opt,
        &SessionConfig::default(),
    )?;
    let outputs = session.run(&inputs)?;

    assert_eq!(outputs.len(), expected.len(), "output count mismatch");
    for (i, (got, exp)) in outputs.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.eq_with_epsilon(exp, 0.05, CompPolicy::Either),
            "tinyllama output[{}] mismatch (shape {:?})",
            i,
            got.dims
        );
    }
    Ok(())
}

#[cfg(feature = "local")]
#[test]
fn tinyllama_cpu() -> TestResult {
    run_tinyllama(Target::CPU)
}

#[cfg(all(feature = "local", feature = "cuda"))]
#[test]
fn tinyllama_cuda() -> TestResult {
    run_tinyllama(Target::CUDA(PrefetchPolicy::Disabled))
}
