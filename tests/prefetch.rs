#![cfg(feature = "cuda")]

use my_nn_engine::onnx::load::LoadProto;
use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine::schedule::scheduler::PrefetchPolicy;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::session::SessionError;
use my_nn_engine::tensor::data::CompPolicy;
use my_nn_engine::tensor::Tensor;

fn model_dir(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/test/single_op")
        .join(name)
}

fn run(
    opt: &Options,
    dir: &std::path::Path,
    inputs: &[Tensor],
) -> Result<Vec<Tensor>, SessionError> {
    let mut session = Session::new(dir.join("model.onnx"), None, opt, &SessionConfig::default())?;
    session.run(inputs)
}

#[test]
fn conv_bias_streamed_weight_matches_resident() -> Result<(), SessionError> {
    let dir = model_dir("conv_bias");
    let input = Tensor::load_from_path(dir.join("input_0.pb"))
        .map_err(|e| SessionError::OtherError(format!("load input: {e:?}")))?;
    let expected = Tensor::load_from_path(dir.join("output_0.pb"))
        .map_err(|e| SessionError::OtherError(format!("load output: {e:?}")))?;
    let inputs = [input];

    // Baseline: all weights resident in VRAM.
    let resident = Options::builder().target(Target::CUDA).build();
    let resident_out = run(&resident, &dir, &inputs)?;

    // Prefetch: threshold between bias (20 B) and weight (540 B), so only
    // `layer.weight` is HostStreamed; bias stays resident.
    let prefetch = Options::builder()
        .target(Target::CUDA)
        .prefetch_policy(Some(PrefetchPolicy::SizeThreshold { min_bytes: 100 }))
        .build();
    let prefetch_out = run(&prefetch, &dir, &inputs)?;

    assert!(
        resident_out[0].eq_with_epsilon(&expected, 1e-4, CompPolicy::Either),
        "resident output diverged from expected"
    );
    assert!(
        prefetch_out[0].eq_with_epsilon(&resident_out[0], 1e-6, CompPolicy::Either),
        "prefetch (streamed weight) output diverged from resident output"
    );
    Ok(())
}
