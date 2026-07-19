#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;

use my_nn_engine::options::Options;
use my_nn_engine::options::PrefetchPolicy;
use my_nn_engine::options::Target;
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
        ResolvedTensorDims::new(dims),
        TensorData::Float(FloatType::F32, data),
    )
    .unwrap()
}

fn make_i64_scalar(v: i64) -> Tensor {
    Tensor::new(
        ResolvedTensorDims::new(&[]),
        TensorData::SInt(SIntType::I64, vec![v]),
    )
    .unwrap()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Verify that:
// 1. A graph input named "cache" is converted to SessionState by the rewrite
//    pass driven by SessionConfig (run() takes the remaining inputs only).
// 2. The state buffer is zero-initialized at session creation.
// 3. KVCacheUpdate writes into the state buffer in place (mandatory in-place
//    aliases through SessionState).
// 4. The state buffer persists across run() calls — successive scatters
//    accumulate into the same buffer.
//
// Graph (reused from single_op/kv_cache_update fixture):
//   inputs:  cache [1,2,8,4] f32, src [1,2,1,4] f32, offset [] i64
//   ops:     cache_new = KVCacheUpdate(cache, src, offset)
//            out       = Sigmoid(cache_new)
//   output:  out [1,2,8,4] f32
#[test]
fn kv_cache_state_persistence() -> TestResult {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/test/single_op/kv_cache_update/model.onnx");

    let config = SessionConfig {
        session_states: vec![SessionStateSpec {
            name: "cache".to_string(),
            bytes: B * H * S_MAX * D * 4,
        }],
        ..SessionConfig::default()
    };

    let opts = Options::builder()
        .target(Target::CUDA(PrefetchPolicy::Disabled))
        .build();

    let mut session = Session::new(&model_path, None, &opts, &config)?;

    // Host-side reference cache, mirroring the device session-state buffer.
    let mut ref_cache = vec![0.0f64; B * H * S_MAX * D];

    for step in 0..3usize {
        // Distinct values per step so state mixing is observable.
        let src_values: Vec<f64> = (0..(B * H * S_NEW * D))
            .map(|i| (i as f64) * 0.1 + (step as f64) + 1.0)
            .collect();
        let offset = step as i64;

        // Reference: scatter src at row=offset along the seq axis.
        for b in 0..B {
            for h in 0..H {
                for d in 0..D {
                    let cache_idx =
                        b * (H * S_MAX * D) + h * (S_MAX * D) + (offset as usize) * D + d;
                    let src_idx = b * (H * S_NEW * D) + h * (S_NEW * D) + d;
                    ref_cache[cache_idx] = src_values[src_idx];
                }
            }
        }

        let src = make_f32(&[B, H, S_NEW, D], src_values);
        let offset_t = make_i64_scalar(offset);

        let outputs = session.run(&[src, offset_t])?;
        assert_eq!(outputs.len(), 1);

        let expected_data: Vec<f64> = ref_cache.iter().map(|&x| sigmoid(x)).collect();
        let expected = make_f32(&[B, H, S_MAX, D], expected_data);

        assert!(
            outputs[0].eq_with_epsilon(&expected, 1e-5, CompPolicy::Either),
            "step {step}: output != sigmoid(reference cache)"
        );
    }

    Ok(())
}

fn ref_attention(q: &[f64], k: &[f64], v: &[f64], active: usize) -> Vec<f64> {
    let scale = 1.0 / (D as f64).sqrt();
    let mut out = vec![0.0f64; (B * H) * D];
    for b in 0..B {
        for h in 0..H {
            let mut logits = vec![0.0f64; active];
            for i in 0..active {
                let mut dot = 0.0f64;
                for d in 0..D {
                    dot += q[b * (H * D) + h * D + d] *
                        k[b * (H * S_MAX * D) + h * (S_MAX * D) + i * D + d];
                }
                logits[i] = dot * scale;
            }
            let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = logits.iter().map(|x| (x - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();
            for d in 0..D {
                let mut acc = 0.0f64;
                for i in 0..active {
                    acc += probs[i] * v[b * (H * S_MAX * D) + h * (S_MAX * D) + i * D + d];
                }
                out[b * (H * D) + h * D + d] = acc;
            }
        }
    }
    out
}

#[test]
fn kv_cache_attention_decode_e2e() -> TestResult {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/test/session_state/kv_cache_attention_decode/model.onnx");

    let cache_size = B * H * S_MAX * D * 4;
    let config = SessionConfig {
        session_states: vec![
            SessionStateSpec {
                name: "K_cache".to_string(),
                bytes: cache_size,
            },
            SessionStateSpec {
                name: "V_cache".to_string(),
                bytes: cache_size,
            },
        ],
        ..SessionConfig::default()
    };

    let opts = Options::builder()
        .target(Target::CUDA(PrefetchPolicy::Disabled))
        .build();

    let mut session = Session::new(&model_path, None, &opts, &config)?;

    let mut k_ref = vec![0.0f64; B * H * S_MAX * D];
    let mut v_ref = vec![0.0f64; B * H * S_MAX * D];

    let make_data = |step: usize, kind: usize| -> Vec<f64> {
        (0..(B * H * D))
            .map(|i| {
                let s = step as f64;
                let f = i as f64;
                let k = kind as f64;
                ((f * 0.31 + s * 1.7 + k * 2.4).sin() * 0.5).clamp(-1.0, 1.0)
            })
            .collect()
    };

    for step in 0..3usize {
        let q_data = make_data(step, 0);
        let src_k = make_data(step, 1);
        let src_v = make_data(step, 2);
        let offset = step as i64;
        let active = (step + 1) as i64;

        for b in 0..B {
            for h in 0..H {
                for d in 0..D {
                    let cache_idx =
                        b * (H * S_MAX * D) + h * (S_MAX * D) + (offset as usize) * D + d;
                    let src_idx = b * (H * D) + h * D + d;
                    k_ref[cache_idx] = src_k[src_idx];
                    v_ref[cache_idx] = src_v[src_idx];
                }
            }
        }
        let expected_data = ref_attention(&q_data, &k_ref, &v_ref, active as usize);
        let expected = make_f32(&[B, H, 1, D], expected_data);

        let q_t = make_f32(&[B, H, 1, D], q_data);
        let src_k_t = make_f32(&[B, H, 1, D], src_k);
        let src_v_t = make_f32(&[B, H, 1, D], src_v);
        let offset_t = make_i64_scalar(offset);
        let active_t = make_i64_scalar(active);

        let outputs = session.run(&[q_t, src_k_t, src_v_t, offset_t, active_t])?;
        assert_eq!(outputs.len(), 1);

        assert!(
            outputs[0].eq_with_epsilon(&expected, 1e-4, CompPolicy::Either),
            "step {step}: attention output mismatch"
        );
    }

    Ok(())
}
