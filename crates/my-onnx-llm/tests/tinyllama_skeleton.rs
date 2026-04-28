#![cfg(feature = "local")]

use std::path::PathBuf;

use my_onnx::onnx::load::LoadProto;
use my_onnx::options::Options;
use my_onnx::options::Target;
use my_onnx::tensor::Tensor;
use my_onnx_llm::LlmSession;

#[test]
fn tinyllama_skeleton_loads_and_runs_prefill() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("models/hf/tinyllama");
    let model_path = dir.join("model.onnx");
    let data_dir = dir.join("test_data_set_0");

    let mut inputs = Vec::new();
    for i in 0.. {
        let p = data_dir.join(format!("input_{i}.pb"));
        if !p.exists() {
            break;
        }
        inputs.push(Tensor::load_from_path(p).unwrap());
    }
    let input_types: Vec<_> = inputs.iter().map(|t| t.tensor_type()).collect();

    let target = if cfg!(feature = "cuda") {
        Target::CUDA
    } else {
        Target::CPU
    };
    let opts = Options::builder().target(target).build();

    let mut session = LlmSession::new(&model_path, &input_types, &opts).unwrap();
    assert_eq!(session.past_len(), 0);

    let outputs = session.run(&inputs).unwrap();
    assert!(
        !outputs.is_empty(),
        "prefill should return at least one output"
    );

    assert!(matches!(
        session.decode(0),
        Err(my_onnx_llm::LlmError::NotImplemented(_))
    ));

    session.reset();
    assert_eq!(session.past_len(), 0);
}
