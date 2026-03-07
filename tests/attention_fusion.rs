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
            attn.penalty.is_none() &&
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
    let final_total_nodes = graph.nodes.iter().count();
    assert_eq!(final_total_nodes, 6);
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
            attn.penalty == Some(penalty as f32) &&
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
    let final_total_nodes = graph.nodes.iter().count();
    assert_eq!(final_total_nodes, 6);
}
