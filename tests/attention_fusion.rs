mod common;

use std::collections::HashMap;

use common::create_value;
use my_onnx::onnx::model::Graph;
use my_onnx::onnx::model::Node;
use my_onnx::onnx::model::ValueId;
use my_onnx::onnx::model::ValueInfo;
use my_onnx::onnx::operator::*;
use my_onnx::onnx::utils::compare_graphs_structural;
use my_onnx::tensor::data::ScalarData;
use my_onnx::tensor::data::TensorData;
use my_onnx::tensor::types::DataType;
use my_onnx::tensor::types::FloatType;
use my_onnx::tensor::types::ResolvedTensorDims;
use my_onnx::tensor::types::TensorType;
use my_onnx::tensor::Tensor;
use my_onnx::transform::modify::NodeDelete;
use my_onnx::transform::modify::SimpleGraphOp;
use my_onnx::transform::optimize::attention_fusion::AttentionFusion;
use my_onnx::transform::optimize::canonicalization::Canonicalization;
use my_onnx::transform::optimize::const_folding::ConstantFolding;
use my_onnx::transform::Pass;

fn build_unfused_attention_graph(scale: f64) -> Graph {
    let scale_tensor = ScalarData::Float(FloatType::F32, scale).to_tensor_data(1);
    let scale_tensor = Tensor::new(ResolvedTensorDims::new(&[]), scale_tensor).unwrap();

    build_graph! {
        name: "attention_no_causal_test",

        inputs: {
            q: (FloatType::F32, &[2, 4, 8, 16]),
            k_t: (FloatType::F32, &[2, 4, 16, 8]),
            v: (FloatType::F32, &[2, 4, 8, 16]),
        },

        outputs: {
            y: (FloatType::F32, &[2, 4, 8, 16]),
        },

        initializers: {
            scale_val = scale_tensor,
        },

        nodes: [
            { "QK", Operator::MatMul,
              [q, k_t] => qk: &[2, 4, 8, 8] },

            { "QK_Scaled", Operator::Mul,
              [qk, scale_val] => qk_scaled: &[2, 4, 8, 8] },

            { "QK_Softmax", Operator::Softmax(Softmax { axis: TensorIndex::new(-1) }),
              [qk_scaled] => qk_softmax: &[2, 4, 8, 8] },

            { "Y", Operator::MatMul,
              [qk_softmax, v] => y: &[2, 4, 8, 16] },
        ]
    }
}

fn build_unfused_causal_attention_graph(scale: f64, penalty: f64) -> Graph {
    let seq = 4;

    let scale_tensor = ScalarData::Float(FloatType::F32, scale).to_tensor_data(1);
    let scale_tensor = Tensor::new(ResolvedTensorDims::new(&[]), scale_tensor).unwrap();

    let mut causal_data = vec![0.0f64; seq * seq];
    for r in 0..seq {
        for c in 0..=r {
            causal_data[r * seq + c] = 1.0;
        }
    }
    let causal_tensor = Tensor::new(
        ResolvedTensorDims::new(&[seq, seq]),
        TensorData::Float(FloatType::F32, causal_data),
    )
    .unwrap();

    let mut penalty_data = vec![0.0f64; seq * seq];
    for r in 0..seq {
        for c in (r + 1)..seq {
            penalty_data[r * seq + c] = penalty;
        }
    }
    let penalty_tensor = Tensor::new(
        ResolvedTensorDims::new(&[seq, seq]),
        TensorData::Float(FloatType::F32, penalty_data),
    )
    .unwrap();

    build_graph! {
        name: "attention_causal_test",

        inputs: {
            q: (FloatType::F32, &[2, 4, 4, 16]),
            k_t: (FloatType::F32, &[2, 4, 16, 4]),
            v: (FloatType::F32, &[2, 4, 4, 16]),
        },

        outputs: {
            y: (FloatType::F32, &[2, 4, 4, 16]),
        },

        initializers: {
            scale_val = scale_tensor,
            causal_mask = causal_tensor,
            penalty_mask = penalty_tensor,
        },

        nodes: [
            { "QK", Operator::MatMul,
              [q, k_t] => qk: &[2, 4, 4, 4] },

            { "QK_Scaled", Operator::Mul,
              [qk, scale_val] => qk_scaled: &[2, 4, 4, 4] },

            { "QK_Causal", Operator::Mul,
              [qk_scaled, causal_mask] => qk_causal: &[2, 4, 4, 4] },

            { "QK_Masked", Operator::Sub,
              [qk_causal, penalty_mask] => qk_masked: &[2, 4, 4, 4] },

            { "QK_Softmax", Operator::Softmax(Softmax { axis: TensorIndex::new(-1) }),
              [qk_masked] => qk_softmax: &[2, 4, 4, 4] },

            { "Y", Operator::MatMul,
              [qk_softmax, v] => y: &[2, 4, 4, 16] },
        ]
    }
}

#[test]
fn test_no_causal_is_fused() {
    let scale = 0.25;
    let mut graph = build_unfused_attention_graph(scale);

    let mut modifier = SimpleGraphOp::new(&graph);
    let canonicalize = Canonicalization::default();
    canonicalize.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);
    let pass = AttentionFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let expected = build_graph! {
        name: "expected",
        inputs: {
            q: (FloatType::F32, &[2, 4, 8, 16]),
            k_t: (FloatType::F32, &[2, 4, 16, 8]),
            v: (FloatType::F32, &[2, 4, 8, 16]),
        },
        outputs: {
            y: (FloatType::F32, &[2, 4, 8, 16]),
        },
        initializers: {},
        nodes: [
            { "Transpose", Operator::Transpose(Transpose { perm: Some(vec![0, 1, 3, 2]) }),
              [k_t] => k: &[2, 4, 8, 16] },
            { "Attention", Operator::Attention(Attention { scale: scale as f32, is_causal: false }),
              [q, k, v] => y: &[2, 4, 8, 16] },
        ]
    };
    compare_graphs_structural(&graph, &expected).unwrap();
}

#[test]
fn test_causal_is_fused() {
    let scale = 0.25;
    let penalty = 10000.0;
    let seq = 4;
    let mut graph = build_unfused_causal_attention_graph(scale, penalty);

    let mut modifier = SimpleGraphOp::new(&graph);
    let canonicalize = Canonicalization::default();
    canonicalize.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);
    let const_fold = ConstantFolding {
        check_strides: false,
    };
    const_fold.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);
    let pass = AttentionFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    // Build expected causal mask: 0 for allowed, -penalty for masked
    let mut mask_data = vec![0.0f64; seq * seq];
    for r in 0..seq {
        for c in (r + 1)..seq {
            mask_data[r * seq + c] = -penalty;
        }
    }
    let mask_tensor = Tensor::new(
        ResolvedTensorDims::new(&[seq, seq]),
        TensorData::Float(FloatType::F32, mask_data),
    )
    .unwrap();

    let expected = build_graph! {
        name: "expected",
        inputs: {
            q: (FloatType::F32, &[2, 4, 4, 16]),
            k_t: (FloatType::F32, &[2, 4, 16, 4]),
            v: (FloatType::F32, &[2, 4, 4, 16]),
        },
        outputs: {
            y: (FloatType::F32, &[2, 4, 4, 16]),
        },
        initializers: {
            mask = mask_tensor,
        },
        nodes: [
            { "Transpose", Operator::Transpose(Transpose { perm: Some(vec![0, 1, 3, 2]) }),
              [k_t] => k: &[2, 4, 4, 16] },
            { "Attention", Operator::Attention(Attention { scale: scale as f32, is_causal: true }),
              [q, k, v, mask] => y: &[2, 4, 4, 16] },
        ]
    };
    compare_graphs_structural(&graph, &expected).unwrap();
}

// Extracted from GPT-2
fn build_gpt2_attention_subgraph() -> Graph {
    let num_heads = 12;
    let seq = 5;
    let d_k = 64;
    let sqrt_dk = (d_k as f64).sqrt(); // 8.0
    let penalty = 10000.0;

    let sqrt_dk_tensor = ScalarData::Float(FloatType::F32, sqrt_dk).to_tensor_data(1);
    let sqrt_dk_tensor = Tensor::new(ResolvedTensorDims::new(&[]), sqrt_dk_tensor).unwrap();

    let mut causal_data = vec![0.0f64; seq * seq];
    for r in 0..seq {
        for c in 0..=r {
            causal_data[r * seq + c] = 1.0;
        }
    }
    let causal_tensor = Tensor::new(
        ResolvedTensorDims::new(&[seq, seq]),
        TensorData::Float(FloatType::F32, causal_data),
    )
    .unwrap();

    let mut penalty_data = vec![0.0f64; seq * seq];
    for r in 0..seq {
        for c in (r + 1)..seq {
            penalty_data[r * seq + c] = penalty;
        }
    }
    let penalty_tensor = Tensor::new(
        ResolvedTensorDims::new(&[seq, seq]),
        TensorData::Float(FloatType::F32, penalty_data),
    )
    .unwrap();

    build_graph! {
        name: "gpt2_attention",

        inputs: {
            q: (FloatType::F32, &[1, num_heads, seq, d_k]),
            k_t: (FloatType::F32, &[1, num_heads, d_k, seq]),
            v: (FloatType::F32, &[1, num_heads, seq, d_k]),
        },

        outputs: {
            y: (FloatType::F32, &[1, num_heads, seq, d_k]),
        },

        initializers: {
            sqrt_dk_val = sqrt_dk_tensor,
            causal_mask = causal_tensor,
            penalty_mask = penalty_tensor,
        },

        nodes: [
            { "QK", Operator::MatMul,
              [q, k_t] => qk: &[1, num_heads, seq, seq] },

            { "QK_Scaled", Operator::Div,
              [qk, sqrt_dk_val] => qk_scaled: &[1, num_heads, seq, seq] },

            { "QK_Causal", Operator::Mul,
              [qk_scaled, causal_mask] => qk_causal: &[1, num_heads, seq, seq] },

            { "QK_Masked", Operator::Sub,
              [qk_causal, penalty_mask] => qk_masked: &[1, num_heads, seq, seq] },

            { "QK_Softmax", Operator::Softmax(Softmax { axis: TensorIndex::new(-1) }),
              [qk_masked] => qk_softmax: &[1, num_heads, seq, seq] },

            { "Y", Operator::MatMul,
              [qk_softmax, v] => y: &[1, num_heads, seq, d_k] },
        ]
    }
}

#[test]
fn test_gpt2_attention_is_fused() {
    let num_heads = 12;
    let seq = 5;
    let d_k = 64;
    let penalty = 10000.0;
    let expected_scale = 1.0 / (d_k as f64).sqrt();

    let mut graph = build_gpt2_attention_subgraph();

    let mut modifier = SimpleGraphOp::new(&graph);
    let canonicalize = Canonicalization::default();
    canonicalize.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);
    let const_fold = ConstantFolding {
        check_strides: false,
    };
    const_fold.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);
    let pass = AttentionFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let mut mask_data = vec![0.0f64; seq * seq];
    for r in 0..seq {
        for c in (r + 1)..seq {
            mask_data[r * seq + c] = -penalty;
        }
    }
    let mask_tensor = Tensor::new(
        ResolvedTensorDims::new(&[seq, seq]),
        TensorData::Float(FloatType::F32, mask_data),
    )
    .unwrap();

    let expected = build_graph! {
        name: "expected",
        inputs: {
            q: (FloatType::F32, &[1, num_heads, seq, d_k]),
            k_t: (FloatType::F32, &[1, num_heads, d_k, seq]),
            v: (FloatType::F32, &[1, num_heads, seq, d_k]),
        },
        outputs: {
            y: (FloatType::F32, &[1, num_heads, seq, d_k]),
        },
        initializers: {
            mask = mask_tensor,
        },
        nodes: [
            { "Transpose", Operator::Transpose(Transpose { perm: Some(vec![0, 1, 3, 2]) }),
              [k_t] => k: &[1, num_heads, seq, d_k] },
            { "Attention", Operator::Attention(Attention { scale: expected_scale as f32, is_causal: true }),
              [q, k, v, mask] => y: &[1, num_heads, seq, d_k] },
        ]
    };
    compare_graphs_structural(&graph, &expected).unwrap();
}

// Extracted from BERT (bertsquad-12)
//
// BERT's attention uses Add for the mask: Add(scaled_QK, attention_mask).
// The attention_mask is an input (not an initializer) since it depends on padding.
fn build_bert_attention_subgraph() -> Graph {
    let num_heads = 12;
    let seq = 8;
    let d_k = 64;
    let scale = 1.0 / (d_k as f64).sqrt(); // 0.125

    let scale_tensor = ScalarData::Float(FloatType::F32, scale).to_tensor_data(1);
    let scale_tensor = Tensor::new(ResolvedTensorDims::new(&[]), scale_tensor).unwrap();

    build_graph! {
        name: "bert_attention",

        inputs: {
            q: (FloatType::F32, &[1, num_heads, seq, d_k]),
            k_t: (FloatType::F32, &[1, num_heads, d_k, seq]),
            v: (FloatType::F32, &[1, num_heads, seq, d_k]),
            attention_mask: (FloatType::F32, &[1, 1, 1, seq]),
        },

        outputs: {
            y: (FloatType::F32, &[1, num_heads, seq, d_k]),
        },

        initializers: {
            scale_val = scale_tensor,
        },

        nodes: [
            { "QK", Operator::MatMul,
              [q, k_t] => qk: &[1, num_heads, seq, seq] },

            { "QK_Scaled", Operator::Mul,
              [qk, scale_val] => qk_scaled: &[1, num_heads, seq, seq] },

            { "QK_Masked", Operator::Add,
              [qk_scaled, attention_mask] => qk_masked: &[1, num_heads, seq, seq] },

            { "QK_Softmax", Operator::Softmax(Softmax { axis: TensorIndex::new(-1) }),
              [qk_masked] => qk_softmax: &[1, num_heads, seq, seq] },

            { "Y", Operator::MatMul,
              [qk_softmax, v] => y: &[1, num_heads, seq, d_k] },
        ]
    }
}

// Extracted from TinyLlama: Q and K are pre-scaled (no Mul after MatMul),
// the mask comes from a Where (runtime-built), and a NaN-guard
// Where(IsNaN(softmax), 0, softmax) sits between Softmax and the V-MatMul.
fn build_tinyllama_attention_subgraph() -> Graph {
    let num_heads = 4;
    let seq = 6;
    let d_k = 8;
    let s_q = 0.5f64;
    let s_k = 0.5f64;

    let s_q_tensor = Tensor::new(
        ResolvedTensorDims::new(&[]),
        ScalarData::Float(FloatType::F32, s_q).to_tensor_data(1),
    )
    .unwrap();
    let s_k_tensor = Tensor::new(
        ResolvedTensorDims::new(&[]),
        ScalarData::Float(FloatType::F32, s_k).to_tensor_data(1),
    )
    .unwrap();
    let neg_inf_tensor = Tensor::new(
        ResolvedTensorDims::new(&[]),
        ScalarData::Float(FloatType::F32, -1.0e9).to_tensor_data(1),
    )
    .unwrap();
    let zero_tensor = Tensor::new(
        ResolvedTensorDims::new(&[]),
        ScalarData::Float(FloatType::F32, 0.0).to_tensor_data(1),
    )
    .unwrap();

    build_graph! {
        name: "tinyllama_attention",

        inputs: {
            q_raw: (FloatType::F32, &[1, num_heads, seq, d_k]),
            k_raw_t: (FloatType::F32, &[1, num_heads, d_k, seq]),
            v: (FloatType::F32, &[1, num_heads, seq, d_k]),
            mask_cond: (DataType::Bool, &[1, 1, seq, seq]),
        },

        outputs: {
            y: (FloatType::F32, &[1, num_heads, seq, d_k]),
        },

        initializers: {
            s_q_val = s_q_tensor,
            s_k_val = s_k_tensor,
            mask_fill = neg_inf_tensor,
            mask_zero = zero_tensor.clone(),
            nan_fill = zero_tensor,
        },

        nodes: [
            { "ScaleQ", Operator::Mul,
              [q_raw, s_q_val] => q_scaled: &[1, num_heads, seq, d_k] },

            { "ScaleK", Operator::Mul,
              [k_raw_t, s_k_val] => k_t_scaled: &[1, num_heads, d_k, seq] },

            { "QK", Operator::MatMul,
              [q_scaled, k_t_scaled] => qk: &[1, num_heads, seq, seq] },

            { "MaskWhere", Operator::Where,
              [mask_cond, mask_fill, mask_zero] => mask: &[1, 1, seq, seq] },

            { "QK_Masked", Operator::Add,
              [qk, mask] => qk_masked: &[1, num_heads, seq, seq] },

            { "QK_Softmax", Operator::Softmax(Softmax { axis: TensorIndex::new(-1) }),
              [qk_masked] => qk_softmax: &[1, num_heads, seq, seq] },

            { "IsNaN", Operator::IsNaN,
              [qk_softmax] => nan_cond: &[1, num_heads, seq, seq] },

            { "NaNGuard", Operator::Where,
              [nan_cond, nan_fill, qk_softmax] => qk_safe: &[1, num_heads, seq, seq] },

            { "Y", Operator::MatMul,
              [qk_safe, v] => y: &[1, num_heads, seq, d_k] },
        ]
    }
}

#[test]
fn test_tinyllama_attention_is_fused() {
    let num_heads = 4;
    let seq = 6;
    let d_k = 8;

    let mut graph = build_tinyllama_attention_subgraph();

    let mut modifier = SimpleGraphOp::new(&graph);
    let canonicalize = Canonicalization::default();
    canonicalize.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);
    let const_fold = ConstantFolding {
        check_strides: false,
    };
    const_fold.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);
    let pass = AttentionFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let attention_nodes: Vec<_> = graph
        .nodes
        .iter()
        .filter(|(_, n)| matches!(n.op, Operator::Attention(_)))
        .collect();
    assert_eq!(
        attention_nodes.len(),
        1,
        "expected exactly one Attention op"
    );
    let (_, attn) = &attention_nodes[0];
    let Operator::Attention(Attention { scale, is_causal }) = attn.op else {
        unreachable!()
    };
    assert!(
        (scale - 0.25).abs() < 1e-6,
        "expected s_q*s_k=0.25, got {scale}"
    );
    assert!(!is_causal, "runtime mask should not be flagged as causal");
    assert_eq!(attn.inputs.len(), 4, "Attention should have mask input");
    let output_ty = graph.get_resolved_tensor_type(attn.outputs[0]).unwrap();
    assert_eq!(output_ty.dims[..], [1, num_heads, seq, d_k]);

    // NaN guard should have been subsumed by the fused op.
    let remaining_isnan = graph
        .nodes
        .iter()
        .filter(|(_, n)| matches!(n.op, Operator::IsNaN))
        .count();
    assert_eq!(remaining_isnan, 0, "IsNaN should be subsumed by Attention");
}

#[test]
fn test_bert_attention_is_fused() {
    let num_heads = 12;
    let seq = 8;
    let d_k = 64;
    let expected_scale = 1.0 / (d_k as f64).sqrt();

    let mut graph = build_bert_attention_subgraph();

    let mut modifier = SimpleGraphOp::new(&graph);
    let canonicalize = Canonicalization::default();
    canonicalize.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);
    let const_fold = ConstantFolding {
        check_strides: false,
    };
    const_fold.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);
    let pass = AttentionFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let expected = build_graph! {
        name: "expected",
        inputs: {
            q: (FloatType::F32, &[1, num_heads, seq, d_k]),
            k_t: (FloatType::F32, &[1, num_heads, d_k, seq]),
            v: (FloatType::F32, &[1, num_heads, seq, d_k]),
            attention_mask: (FloatType::F32, &[1, 1, 1, seq]),
        },
        outputs: {
            y: (FloatType::F32, &[1, num_heads, seq, d_k]),
        },
        initializers: {},
        nodes: [
            { "Transpose", Operator::Transpose(Transpose { perm: Some(vec![0, 1, 3, 2]) }),
              [k_t] => k: &[1, num_heads, seq, d_k] },
            { "Attention", Operator::Attention(Attention { scale: expected_scale as f32, is_causal: false }),
              [q, k, v, attention_mask] => y: &[1, num_heads, seq, d_k] },
        ]
    };
    compare_graphs_structural(&graph, &expected).unwrap();
}
