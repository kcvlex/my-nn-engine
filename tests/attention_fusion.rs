mod common;

use std::collections::HashMap;

use common::create_value;
use my_onnx::onnx::model::Graph;
use my_onnx::onnx::model::Node;
use my_onnx::onnx::model::NodeId;
use my_onnx::onnx::model::ValueId;
use my_onnx::onnx::model::ValueInfo;
use my_onnx::onnx::operator::args;
use my_onnx::onnx::operator::*;
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
use my_onnx::transform::optimize::canonicalize::Canonicalize;
use my_onnx::transform::optimize::const_fold::ConstantFold;
use my_onnx::transform::Pass;

fn find_nodes<F>(graph: &Graph, predicate: F) -> Vec<NodeId>
where
    F: Fn(&Node) -> bool,
{
    graph
        .nodes
        .iter()
        .filter(|(_, node)| predicate(node))
        .map(|(id, _)| id)
        .collect()
}

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
    let inputs = graph.inputs.clone();
    let outputs = graph.outputs.clone();

    macro_rules! match_input {
        ($value_id:expr, $node_id:expr) => {{
            let node = &graph.nodes[$node_id];
            matches!(node.op, Operator::Input(id) if id == $value_id)
        }}
    }

    macro_rules! match_output {
        ($value_id:expr, $node_id:expr) => {{
            let node = &graph.nodes[$node_id];
            matches!(node.op, Operator::Output(id) if id == $value_id)
        }}
    }

    let mut modifier = SimpleGraphOp::new(&graph);
    let canonicalize = Canonicalize::default();
    canonicalize.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);
    let pass = AttentionFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let attentions = find_nodes(&graph, |node| {
        let attn = match &node.op {
            Operator::Attention(attn) => attn,
            _ => return false,
        };

        attn.scale == scale as f32 &&
            !attn.is_causal &&
            node.inputs.len() == 3 &&
            match_input!(node.inputs[args::ATTENTION_Q], inputs[0]) &&
            match_input!(node.inputs[args::ATTENTION_V], inputs[2]) &&
            match_output!(node.outputs[0], outputs[0])
    });
    assert_eq!(attentions.len(), 1);

    let transposes = find_nodes(&graph, |node| matches!(&node.op, Operator::Transpose(_)));
    assert_eq!(transposes.len(), 1);
    let transpose_node = &graph.nodes[transposes[0]];
    assert!(match_input!(transpose_node.inputs[0], inputs[1]));

    // Input(q) + Input(k_t) + Input(v) + Output(y) + Transpose + Attention = 6
    assert_eq!(graph.nodes.iter().count(), 6);
}

#[test]
fn test_causal_is_fused() {
    let scale = 0.25;
    let penalty = 10000.0;
    let mut graph = build_unfused_causal_attention_graph(scale, penalty);
    let inputs = graph.inputs.clone();
    let outputs = graph.outputs.clone();

    macro_rules! match_input {
        ($value_id:expr, $node_id:expr) => {{
            let node = &graph.nodes[$node_id];
            matches!(node.op, Operator::Input(id) if id == $value_id)
        }}
    }

    macro_rules! match_output {
        ($value_id:expr, $node_id:expr) => {{
            let node = &graph.nodes[$node_id];
            matches!(node.op, Operator::Output(id) if id == $value_id)
        }}
    }

    let mut modifier = SimpleGraphOp::new(&graph);
    let canonicalize = Canonicalize::default();
    canonicalize.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);
    let pass = AttentionFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let attentions = find_nodes(&graph, |node| {
        let attn = match &node.op {
            Operator::Attention(attn) => attn,
            _ => return false,
        };

        attn.scale == scale as f32 &&
            attn.is_causal &&
            node.inputs.len() == 4 &&
            match_input!(node.inputs[args::ATTENTION_Q], inputs[0]) &&
            match_input!(node.inputs[args::ATTENTION_V], inputs[2]) &&
            match_output!(node.outputs[0], outputs[0])
    });
    assert_eq!(attentions.len(), 1);

    // Verify the mask is an additive mask initializer with correct values
    let attn_node = &graph.nodes[attentions[0]];
    let mask_value = attn_node.inputs[args::ATTENTION_MASK];
    let mask_tensor = graph.initializer.get(&mask_value).unwrap();
    let TensorData::Float(_, ref data) = mask_tensor.data else {
        panic!("Expected float mask");
    };
    let seq = 4;
    for r in 0..seq {
        for c in 0..seq {
            let val = data[r * seq + c];
            if c <= r {
                assert_eq!(val, 0.0, "allowed position ({r}, {c}) should be 0");
            } else {
                assert_eq!(
                    val, -penalty,
                    "masked position ({r}, {c}) should be -{penalty}"
                );
            }
        }
    }

    let transposes = find_nodes(&graph, |node| matches!(&node.op, Operator::Transpose(_)));
    assert_eq!(transposes.len(), 1);
    let transpose_node = &graph.nodes[transposes[0]];
    assert!(match_input!(transpose_node.inputs[0], inputs[1]));

    // Input(q) + Input(k_t) + Input(v) + Output(y) + Transpose + Attention = 6
    // (mask is an initializer, not a separate node)
    assert_eq!(graph.nodes.iter().count(), 6);
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
    let mut graph = build_gpt2_attention_subgraph();
    let inputs = graph.inputs.clone();
    let outputs = graph.outputs.clone();

    macro_rules! match_input {
        ($value_id:expr, $node_id:expr) => {{
            let node = &graph.nodes[$node_id];
            matches!(node.op, Operator::Input(id) if id == $value_id)
        }}
    }

    macro_rules! match_output {
        ($value_id:expr, $node_id:expr) => {{
            let node = &graph.nodes[$node_id];
            matches!(node.op, Operator::Output(id) if id == $value_id)
        }}
    }

    // Canonicalize (Div -> Reciprocal + Mul) -> ConstantFold (Reciprocal(8.0) -> 0.125) -> AttentionFusion
    let mut modifier = SimpleGraphOp::new(&graph);

    let canonicalize = Canonicalize::default();
    canonicalize.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let const_fold = ConstantFold {
        check_strides: false,
    };
    const_fold.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let pass = AttentionFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let expected_scale = 1.0 / (64.0f64).sqrt();
    let attentions = find_nodes(&graph, |node| {
        let attn = match &node.op {
            Operator::Attention(attn) => attn,
            _ => return false,
        };

        attn.scale == expected_scale as f32 &&
            attn.is_causal &&
            node.inputs.len() == 4 &&
            match_input!(node.inputs[args::ATTENTION_Q], inputs[0]) &&
            match_input!(node.inputs[args::ATTENTION_V], inputs[2]) &&
            match_output!(node.outputs[0], outputs[0])
    });
    assert_eq!(attentions.len(), 1);

    let transposes = find_nodes(&graph, |node| matches!(&node.op, Operator::Transpose(_)));
    assert_eq!(transposes.len(), 1);
    let transpose_node = &graph.nodes[transposes[0]];
    assert!(match_input!(transpose_node.inputs[0], inputs[1]));

    // Input(q) + Input(k_t) + Input(v) + Output(y) + Transpose + Attention = 6
    assert_eq!(graph.nodes.iter().count(), 6);
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

#[test]
fn test_bert_attention_is_fused() {
    let mut graph = build_bert_attention_subgraph();
    let inputs = graph.inputs.clone();
    let outputs = graph.outputs.clone();

    macro_rules! match_input {
        ($value_id:expr, $node_id:expr) => {{
            let node = &graph.nodes[$node_id];
            matches!(node.op, Operator::Input(id) if id == $value_id)
        }}
    }

    macro_rules! match_output {
        ($value_id:expr, $node_id:expr) => {{
            let node = &graph.nodes[$node_id];
            matches!(node.op, Operator::Output(id) if id == $value_id)
        }}
    }

    let mut modifier = SimpleGraphOp::new(&graph);

    let canonicalize = Canonicalize::default();
    canonicalize.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let const_fold = ConstantFold {
        check_strides: false,
    };
    const_fold.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let pass = AttentionFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let expected_scale = 1.0 / (64.0f64).sqrt();
    let attentions = find_nodes(&graph, |node| {
        let attn = match &node.op {
            Operator::Attention(attn) => attn,
            _ => return false,
        };

        attn.scale == expected_scale as f32 &&
            !attn.is_causal &&
            node.inputs.len() == 4 &&
            match_input!(node.inputs[args::ATTENTION_Q], inputs[0]) &&
            match_input!(node.inputs[args::ATTENTION_V], inputs[2]) &&
            match_output!(node.outputs[0], outputs[0])
    });
    assert_eq!(attentions.len(), 1);

    // The mask input should be the original attention_mask graph input
    let attn_node = &graph.nodes[attentions[0]];
    let mask_value = attn_node.inputs[args::ATTENTION_MASK];
    assert!(match_input!(mask_value, inputs[3]));

    let transposes = find_nodes(&graph, |node| matches!(&node.op, Operator::Transpose(_)));
    assert_eq!(transposes.len(), 1);
    let transpose_node = &graph.nodes[transposes[0]];
    assert!(match_input!(transpose_node.inputs[0], inputs[1]));

    // Input(q) + Input(k_t) + Input(v) + Input(attention_mask) + Output(y) + Transpose + Attention = 7
    assert_eq!(graph.nodes.iter().count(), 7);
}
