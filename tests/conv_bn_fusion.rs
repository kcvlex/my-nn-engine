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
use my_onnx::tensor::data::TensorData;
use my_onnx::tensor::types::DataType;
use my_onnx::tensor::types::FloatType;
use my_onnx::tensor::types::ResolvedTensorDims;
use my_onnx::tensor::types::TensorType;
use my_onnx::tensor::Tensor;
use my_onnx::transform::modify::NodeDelete;
use my_onnx::transform::modify::SimpleGraphOp;
use my_onnx::transform::optimize::conv_bn_fusion::ConvBNFusion;
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

fn make_1d_tensor(data: Vec<f64>) -> Tensor {
    let len = data.len();
    Tensor::new(
        ResolvedTensorDims::new(&[len]),
        TensorData::Float(FloatType::F32, data),
    )
    .unwrap()
}

fn make_tensor(dims: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::new(
        ResolvedTensorDims::new(dims),
        TensorData::Float(FloatType::F32, data),
    )
    .unwrap()
}

// Conv(3 input channels, 2 output channels, 1x1 kernel) -> BatchNorm
fn build_conv_bn_graph_no_conv_bias() -> Graph {
    let c_out = 2;
    let c_in = 3;

    let weight = make_tensor(&[c_out, c_in, 1, 1], vec![1.0; c_out * c_in]);
    let bn_scale = make_1d_tensor(vec![2.0, 3.0]);
    let bn_bias = make_1d_tensor(vec![0.5, -0.5]);
    let bn_mean = make_1d_tensor(vec![1.0, 2.0]);
    let bn_var = make_1d_tensor(vec![4.0, 9.0]);

    build_graph! {
        name: "conv_bn_no_bias",

        inputs: {
            x: (FloatType::F32, &[1, c_in, 4, 4]),
        },

        outputs: {
            y: (FloatType::F32, &[1, c_out, 4, 4]),
        },

        initializers: {
            weight = weight,
            bn_scale = bn_scale,
            bn_bias = bn_bias,
            bn_mean = bn_mean,
            bn_var = bn_var,
        },

        nodes: [
            { "Conv", Operator::Conv(Conv {
                pad: ConvPad::NotSet(OptionalVec::new(None, (0, 0))),
                dilations: OptionalVec::new(None, 1),
                groups: 1,
                kernel_shape: ResolvedTensorDims::new(&[1, 1]),
                strides: OptionalVec::new(None, 1),
                layout: Layout::NCHW,
            }),
              [x, weight] => conv_out: &[1, c_out, 4, 4] },

            { "BN", Operator::BatchNormalization(BatchNormalization {
                epsilon: 0.0,
                momentum: 0.9,
            }),
              [conv_out, bn_scale, bn_bias, bn_mean, bn_var] => y: &[1, c_out, 4, 4] },
        ]
    }
}

// Conv(3 input channels, 2 output channels, 1x1 kernel, with bias) -> BatchNorm
fn build_conv_bn_graph_with_conv_bias() -> Graph {
    let c_out = 2;
    let c_in = 3;

    let weight = make_tensor(&[c_out, c_in, 1, 1], vec![1.0; c_out * c_in]);
    let conv_bias = make_1d_tensor(vec![0.1, 0.2]);
    let bn_scale = make_1d_tensor(vec![2.0, 3.0]);
    let bn_bias = make_1d_tensor(vec![0.5, -0.5]);
    let bn_mean = make_1d_tensor(vec![1.0, 2.0]);
    let bn_var = make_1d_tensor(vec![4.0, 9.0]);

    build_graph! {
        name: "conv_bn_with_bias",

        inputs: {
            x: (FloatType::F32, &[1, c_in, 4, 4]),
        },

        outputs: {
            y: (FloatType::F32, &[1, c_out, 4, 4]),
        },

        initializers: {
            weight = weight,
            conv_bias = conv_bias,
            bn_scale = bn_scale,
            bn_bias = bn_bias,
            bn_mean = bn_mean,
            bn_var = bn_var,
        },

        nodes: [
            { "Conv", Operator::Conv(Conv {
                pad: ConvPad::NotSet(OptionalVec::new(None, (0, 0))),
                dilations: OptionalVec::new(None, 1),
                groups: 1,
                kernel_shape: ResolvedTensorDims::new(&[1, 1]),
                strides: OptionalVec::new(None, 1),
                layout: Layout::NCHW,
            }),
              [x, weight, conv_bias] => conv_out: &[1, c_out, 4, 4] },

            { "BN", Operator::BatchNormalization(BatchNormalization {
                epsilon: 0.0,
                momentum: 0.9,
            }),
              [conv_out, bn_scale, bn_bias, bn_mean, bn_var] => y: &[1, c_out, 4, 4] },
        ]
    }
}

#[test]
fn test_conv_bn_fused_no_conv_bias() {
    let mut graph = build_conv_bn_graph_no_conv_bias();
    let mut modifier = SimpleGraphOp::new(&graph);
    let pass = ConvBNFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    // BN should be eliminated
    let bns = find_nodes(&graph, |node| {
        matches!(&node.op, Operator::BatchNormalization(_))
    });
    assert_eq!(bns.len(), 0);

    // Should have exactly one Conv
    let convs = find_nodes(&graph, |node| matches!(&node.op, Operator::Conv(_)));
    assert_eq!(convs.len(), 1);

    let conv_node = &graph.nodes[convs[0]];

    // Conv should now have 3 inputs (data, weight, bias)
    assert_eq!(conv_node.inputs.len(), 3);

    // Verify fused weight: original weight=1.0, multiplier = scale/sqrt(var)
    // Channel 0: multiplier = 2.0 / sqrt(4.0) = 1.0, new_weight = 1.0 * 1.0 = 1.0
    // Channel 1: multiplier = 3.0 / sqrt(9.0) = 1.0, new_weight = 1.0 * 1.0 = 1.0
    let fused_weight = graph
        .initializer
        .get(&conv_node.inputs[args::CONV_WEIGHT])
        .unwrap();
    let TensorData::Float(_, ref w) = fused_weight.data else {
        panic!()
    };
    for val in w {
        assert!((val - 1.0).abs() < 1e-6);
    }

    // Verify fused bias: bn_bias - bn_mean * multiplier
    // Channel 0: 0.5 - 1.0 * 1.0 = -0.5
    // Channel 1: -0.5 - 2.0 * 1.0 = -2.5
    let fused_bias = graph
        .initializer
        .get(&conv_node.inputs[args::CONV_BIAS])
        .unwrap();
    let TensorData::Float(_, ref b) = fused_bias.data else {
        panic!()
    };
    assert!((b[0] - (-0.5)).abs() < 1e-6);
    assert!((b[1] - (-2.5)).abs() < 1e-6);
}

#[test]
fn test_conv_bn_fused_with_conv_bias() {
    let mut graph = build_conv_bn_graph_with_conv_bias();
    let mut modifier = SimpleGraphOp::new(&graph);
    let pass = ConvBNFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let bns = find_nodes(&graph, |node| {
        matches!(&node.op, Operator::BatchNormalization(_))
    });
    assert_eq!(bns.len(), 0);

    let convs = find_nodes(&graph, |node| matches!(&node.op, Operator::Conv(_)));
    assert_eq!(convs.len(), 1);

    let conv_node = &graph.nodes[convs[0]];
    assert_eq!(conv_node.inputs.len(), 3);

    // Fused bias: bn_bias - bn_mean * multiplier + conv_bias * multiplier
    // Channel 0: 0.5 - 1.0 * 1.0 + 0.1 * 1.0 = -0.4
    // Channel 1: -0.5 - 2.0 * 1.0 + 0.2 * 1.0 = -2.3
    let fused_bias = graph
        .initializer
        .get(&conv_node.inputs[args::CONV_BIAS])
        .unwrap();
    let TensorData::Float(_, ref b) = fused_bias.data else {
        panic!()
    };
    assert!((b[0] - (-0.4)).abs() < 1e-6);
    assert!((b[1] - (-2.3)).abs() < 1e-6);
}

#[test]
fn test_conv_bn_not_fused_when_multiple_users() {
    // If Conv output is used by more than just BN, fusion should not apply
    let c_out = 2;
    let c_in = 3;

    let weight = make_tensor(&[c_out, c_in, 1, 1], vec![1.0; c_out * c_in]);
    let bn_scale = make_1d_tensor(vec![2.0, 3.0]);
    let bn_bias = make_1d_tensor(vec![0.5, -0.5]);
    let bn_mean = make_1d_tensor(vec![1.0, 2.0]);
    let bn_var = make_1d_tensor(vec![4.0, 9.0]);

    let mut graph = build_graph! {
        name: "conv_bn_multi_user",

        inputs: {
            x: (FloatType::F32, &[1, c_in, 4, 4]),
        },

        outputs: {
            y1: (FloatType::F32, &[1, c_out, 4, 4]),
            y2: (FloatType::F32, &[1, c_out, 4, 4]),
        },

        initializers: {
            weight = weight,
            bn_scale = bn_scale,
            bn_bias = bn_bias,
            bn_mean = bn_mean,
            bn_var = bn_var,
        },

        nodes: [
            { "Conv", Operator::Conv(Conv {
                pad: ConvPad::NotSet(OptionalVec::new(None, (0, 0))),
                dilations: OptionalVec::new(None, 1),
                groups: 1,
                kernel_shape: ResolvedTensorDims::new(&[1, 1]),
                strides: OptionalVec::new(None, 1),
                layout: Layout::NCHW,
            }),
              [x, weight] => conv_out: &[1, c_out, 4, 4] },

            { "BN", Operator::BatchNormalization(BatchNormalization {
                epsilon: 0.0,
                momentum: 0.9,
            }),
              [conv_out, bn_scale, bn_bias, bn_mean, bn_var] => y1: &[1, c_out, 4, 4] },

            { "Identity", Operator::Identity,
              [conv_out] => y2: &[1, c_out, 4, 4] },
        ]
    };

    let mut modifier = SimpleGraphOp::new(&graph);
    let pass = ConvBNFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    // BN should NOT be eliminated because conv_out has multiple users
    let bns = find_nodes(&graph, |node| {
        matches!(&node.op, Operator::BatchNormalization(_))
    });
    assert_eq!(bns.len(), 1);
}
