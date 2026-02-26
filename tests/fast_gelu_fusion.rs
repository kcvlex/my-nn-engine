mod common;

use std::collections::HashMap;

use common::create_value;
use my_onnx::onnx::model::Graph;
use my_onnx::onnx::model::Node;
use my_onnx::onnx::model::NodeId;
use my_onnx::onnx::model::ValueId;
use my_onnx::onnx::model::ValueInfo;
use my_onnx::onnx::operator::*;
use my_onnx::tensor::data::ScalarData;
use my_onnx::tensor::types::DataType;
use my_onnx::tensor::types::FloatType;
use my_onnx::tensor::types::ResolvedTensorDims;
use my_onnx::tensor::types::TensorType;
use my_onnx::tensor::Tensor;
use my_onnx::transform::modify::NodeDelete;
use my_onnx::transform::modify::SimpleGraphOp;
use my_onnx::transform::optimize::fast_gelu_fusion::FastGeLUFusion;
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

fn make_scalar(val: f64) -> Tensor {
    let data = ScalarData::Float(FloatType::F32, val).to_tensor_data(1);
    Tensor::new(ResolvedTensorDims::new(&[]), data).unwrap()
}

// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn build_unfused_fast_gelu_graph() -> Graph {
    let coeff = make_scalar(0.044715);
    let sqrt_2_over_pi = make_scalar((2.0f64 / std::f64::consts::PI).sqrt());
    let one = make_scalar(1.0);
    let half = make_scalar(0.5);
    let three = {
        let data = ScalarData::Float(FloatType::F32, 3.0).to_tensor_data(1);
        Tensor::new(ResolvedTensorDims::new(&[]), data).unwrap()
    };

    build_graph! {
        name: "fast_gelu_test",

        inputs: {
            x: (FloatType::F32, &[2, 10, 768]),
        },

        outputs: {
            y: (FloatType::F32, &[2, 10, 768]),
        },

        initializers: {
            coeff = coeff,
            sqrt_2_over_pi = sqrt_2_over_pi,
            one = one,
            half = half,
            three = three,
        },

        nodes: [
            // x^3
            { "Cube", Operator::Pow,
              [x, three] => cube_out: &[2, 10, 768] },

            // 0.044715 * x^3
            { "ScaledCube", Operator::Mul,
              [cube_out, coeff] => scaled_cube_out: &[2, 10, 768] },

            // x + 0.044715 * x^3
            { "InnerSum", Operator::Add,
              [x, scaled_cube_out] => inner_sum_out: &[2, 10, 768] },

            // sqrt(2/pi) * (x + 0.044715 * x^3)
            { "ScaledInner", Operator::Mul,
              [inner_sum_out, sqrt_2_over_pi] => scaled_inner_out: &[2, 10, 768] },

            // tanh(...)
            { "Tanh", Operator::Tanh,
              [scaled_inner_out] => tanh_out: &[2, 10, 768] },

            // 1 + tanh(...)
            { "OnePlusTanh", Operator::Add,
              [tanh_out, one] => one_plus_tanh_out: &[2, 10, 768] },

            // 0.5 * x
            { "HalfX", Operator::Mul,
              [x, half] => half_x_out: &[2, 10, 768] },

            // 0.5 * x * (1 + tanh(...))
            { "Y", Operator::Mul,
              [half_x_out, one_plus_tanh_out] => y: &[2, 10, 768] },
        ]
    }
}

#[test]
fn test_fast_gelu_is_fused() {
    let mut graph = build_unfused_fast_gelu_graph();
    let inputs = graph.inputs.clone();
    let outputs = graph.outputs.clone();

    let mut modifier = SimpleGraphOp::new(&graph);
    let pass = FastGeLUFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let gelu_nodes = find_nodes(&graph, |node| {
        let Operator::GeLU(GeLU { approximate }) = &node.op else {
            return false;
        };
        if !approximate {
            return false;
        }
        let x_input = node.inputs[0];
        let x_node_id = inputs[0];
        matches!(graph.nodes[x_node_id].op, Operator::Input(id) if id == x_input)
    });
    assert_eq!(gelu_nodes.len(), 1, "expected exactly one fused GeLU node");

    let gelu_node = &graph.nodes[gelu_nodes[0]];
    let gelu_output = gelu_node.outputs[0];
    let out_node_id = outputs[0];
    assert!(
        matches!(graph.nodes[out_node_id].op, Operator::Output(id) if id == gelu_output),
        "GeLU output should feed into the graph output"
    );
}
