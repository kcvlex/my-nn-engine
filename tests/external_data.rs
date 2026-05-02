mod common;

use std::path::PathBuf;

use my_nn_engine::onnx::load::LoadProto;
use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::session::SessionError;
use my_nn_engine::tensor::data::CompPolicy;
use my_nn_engine::tensor::Tensor;

type TestResult = Result<(), SessionError>;

fn run_external(target: Target) -> TestResult {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/test/single_op/conv_external");
    let model_path = dir.join("model.onnx");
    let input =
        Tensor::load_from_path(dir.join("input_0.pb")).map_err(SessionError::ModelLoadError)?;
    let expected =
        Tensor::load_from_path(dir.join("output_0.pb")).map_err(SessionError::ModelLoadError)?;

    let opt = Options::builder().target(target).build();
    let mut session = Session::new(&model_path, None, &opt, &SessionConfig::default())?;
    let outputs = session.run(&[input])?;

    assert!(
        outputs[0].eq_with_epsilon(&expected, 1e-4, CompPolicy::Either),
        "conv_external output mismatch"
    );
    Ok(())
}

#[test]
fn conv_external_cpu() -> TestResult {
    run_external(Target::CPU)
}

#[cfg(feature = "cuda")]
#[test]
fn conv_external_cuda() -> TestResult {
    run_external(Target::CUDA)
}
