#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine::schedule::scheduler::PlacementStrategy;
use my_nn_engine::session::DeviceBuffer;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::session::SessionError;
use my_nn_engine::session::SessionStateSpec;
use my_nn_engine::tensor::data::CompPolicy;
use my_nn_engine::tensor::data::TensorData;
use my_nn_engine::tensor::types::FloatType;
use my_nn_engine::tensor::types::ResolvedTensorDims;
use my_nn_engine::tensor::types::SIntType;
use my_nn_engine::tensor::Tensor;

type TestResult = Result<(), SessionError>;

const B: usize = 1;
const H: usize = 2;
const S_MAX: usize = 8;
const D: usize = 4;
const S_NEW: usize = 1;

fn make_f32(dims: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::new(
        ResolvedTensorDims::from(dims),
        TensorData::Float(FloatType::F32, data),
    )
    .unwrap()
}

fn make_i64_scalar(v: i64) -> Tensor {
    Tensor::new(
        ResolvedTensorDims::from(&[][..]),
        TensorData::SInt(SIntType::I64, vec![v]),
    )
    .unwrap()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// kv_cache_update + Sigmoid splits cleanly under StructuralKvTouch:
//   KVCacheUpdate touches the SessionState buffer -> CUDA
//   Sigmoid touches cache_new only -> CPU
// Plan therefore inserts a GPU->Host Transfer for cache_new, exercising both
// CUDA per-step wrapper dispatch and direct cudaMemcpy in HybridSession.
#[test]
fn kv_cache_update_then_sigmoid_hybrid() -> TestResult {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/test/single_op/kv_cache_update/model.onnx");

    let cache_buf = Arc::new(DeviceBuffer::alloc_zeroed(B * H * S_MAX * D * 4).unwrap());
    let config = SessionConfig {
        session_states: vec![SessionStateSpec {
            name: "cache".to_string(),
            buffer: cache_buf,
        }],
        ..SessionConfig::default()
    };

    let opts = Options::builder()
        .target(Target::CUDA)
        .placement_strategy(Some(PlacementStrategy::StructuralKvTouch))
        .build();

    let mut session = Session::new(&model_path, None, &opts, &config)?;

    let src_values: Vec<f64> = (0..(B * H * S_NEW * D))
        .map(|i| (i as f64) * 0.1 + 1.0)
        .collect();
    let offset = 0i64;
    let mut ref_cache = vec![0.0f64; B * H * S_MAX * D];
    for b in 0..B {
        for h in 0..H {
            for d in 0..D {
                let cache_idx = b * (H * S_MAX * D) + h * (S_MAX * D) + (offset as usize) * D + d;
                let src_idx = b * (H * S_NEW * D) + h * (S_NEW * D) + d;
                ref_cache[cache_idx] = src_values[src_idx];
            }
        }
    }
    let expected_data: Vec<f64> = ref_cache.iter().map(|&x| sigmoid(x)).collect();
    let expected = make_f32(&[B, H, S_MAX, D], expected_data);

    let src = make_f32(&[B, H, S_NEW, D], src_values);
    let offset_t = make_i64_scalar(offset);
    let outputs = session.run(&[src, offset_t])?;
    assert_eq!(outputs.len(), 1);
    assert!(
        outputs[0].eq_with_epsilon(&expected, 1e-5, CompPolicy::Either),
        "hybrid output != sigmoid(reference cache)"
    );
    Ok(())
}
