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
use my_onnx::tensor::types::DataType;
use my_onnx::tensor::types::FloatType;
use my_onnx::tensor::types::ResolvedTensorDims;
use my_onnx::tensor::types::TensorType;
use my_onnx::tensor::Tensor;
use my_onnx::transform::modify::NodeDelete;
use my_onnx::transform::modify::SimpleGraphOp;
use my_onnx::transform::optimize::canonicalize::Canonicalize;
use my_onnx::transform::optimize::layer_norm_fusion::LayerNormFusion;
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

fn build_unfused_layer_norm_graph(epsilon: f64) -> Graph {
    let epsilon = ScalarData::Float(FloatType::F32, epsilon).to_tensor_data(1);
    let epsilon = Tensor::new(ResolvedTensorDims::new(&[]), epsilon).unwrap();

    build_graph! {
        name: "layer_norm_test",

        inputs: {
            x: (FloatType::F32, &[2, 10, 768]),
            scale: (FloatType::F32, &[768]),
            bias: (FloatType::F32, &[768]),
        },

        outputs: {
            y: (FloatType::F32, &[2, 10, 768]),
        },

        initializers: {
            epsilon = epsilon,
        },

        nodes: [
            { "Mean", Operator::ReduceMean(Reduce { axes: vec![-1], keepdims: true }),
              [x] => mean_out: &[2, 10, 1] },

            { "D", Operator::Sub,
              [x, mean_out] => d_out: &[2, 10, 768] },

            { "DD", Operator::Mul,
              [d_out, d_out] => dd_out: &[2, 10, 768] },

            { "Var", Operator::ReduceMean(Reduce { axes: vec![-1], keepdims: true }),
              [dd_out] => var_out: &[2, 10, 1] },

            { "VarEps", Operator::Add,
              [var_out, epsilon] => var_eps_out: &[2, 10, 1] },

            { "StdDev", Operator::Sqrt,
              [var_eps_out] => std_out: &[2, 10, 1] },

            { "Normalized", Operator::Div,
              [d_out, std_out] => normalized_out: &[2, 10, 768] },

            { "NormalizedScaled", Operator::Mul,
              [normalized_out, scale] => scaled_out: &[2, 10, 768] },

            { "Y", Operator::Add,
              [scaled_out, bias] => y: &[2, 10, 768] },
        ]
    }
}

#[test]
fn test_valid_pattern_is_fused() {
    let epsilon = 1e-5;
    let mut graph = build_unfused_layer_norm_graph(epsilon);
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
    let pass = LayerNormFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let layer_norms = find_nodes(&graph, |node| {
        let eps = match &node.op {
            Operator::LayerNormalization(LayerNormalization { epsilon, .. }) => *epsilon,
            _ => return false,
        };

        eps == epsilon &&
            match_input!(node.inputs[args::LAYER_NORM_DATA], inputs[0]) &&
            match_input!(node.inputs[args::LAYER_NORM_SCALE], inputs[1]) &&
            match_input!(node.inputs[args::LAYER_NORM_BIAS], inputs[2]) &&
            match_output!(node.outputs[0], outputs[0])
    });
    assert_eq!(layer_norms.len(), 1);

    let final_total_nodes = graph.nodes.iter().count();
    assert!(final_total_nodes == 5);
}
