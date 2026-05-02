mod common;

use std::collections::HashMap;

use common::create_value;
use common::make_1d_tensor;
use common::make_tensor;
use my_nn_engine::graph::operator::*;
use my_nn_engine::graph::utils::compare_graphs_structural;
use my_nn_engine::graph::utils::compare_graphs_structural_with_epsilon;
use my_nn_engine::graph::Graph;
use my_nn_engine::graph::Node;
use my_nn_engine::graph::ValueId;
use my_nn_engine::graph::ValueInfo;
use my_nn_engine::tensor::types::DataType;
use my_nn_engine::tensor::types::FloatType;
use my_nn_engine::tensor::types::ResolvedTensorDims;
use my_nn_engine::tensor::types::TensorType;
use my_nn_engine::transform::modify::NodeDelete;
use my_nn_engine::transform::modify::SimpleGraphOp;
use my_nn_engine::transform::optimize::conv_bn_fusion::ConvBNFusion;
use my_nn_engine::transform::Pass;

const C_OUT: usize = 2;
const C_IN: usize = 3;
const EPSILON: f64 = 1e-6;

fn conv_op() -> Conv {
    Conv {
        pad: ConvPad::NotSet(OptionalVec::new(None, (0, 0))),
        dilations: OptionalVec::new(None, 1),
        group: 1,
        kernel_shape: ResolvedTensorDims::new(&[1, 1]),
        strides: OptionalVec::new(None, 1),
        input_layout: Layout::NCHW,
        output_layout: Layout::NCHW,
        activation: Activation::default(),
    }
}

fn bn_params() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let scale = vec![2.0, 3.0];
    let bias = vec![0.5, -0.5];
    let mean = vec![1.0, 2.0];
    let var = vec![4.0, 9.0];
    (scale, bias, mean, var)
}

fn compute_fused_weight(weight: &[f64], multiplier: &[f64]) -> Vec<f64> {
    let per_channel = weight.len() / C_OUT;
    let mut fused = weight.to_vec();
    for c in 0..C_OUT {
        for i in 0..per_channel {
            fused[c * per_channel + i] *= multiplier[c];
        }
    }
    fused
}

fn compute_fused_bias(
    conv_bias: Option<&[f64]>,
    bn_scale: &[f64],
    bn_bias: &[f64],
    bn_mean: &[f64],
    bn_var: &[f64],
    bn_epsilon: f64,
) -> Vec<f64> {
    (0..C_OUT)
        .map(|c| {
            let multiplier = bn_scale[c] / (bn_var[c] + bn_epsilon).sqrt();
            let cb = conv_bias.map_or(0.0, |b| b[c]);
            bn_bias[c] + (cb - bn_mean[c]) * multiplier
        })
        .collect()
}

fn compute_multiplier(bn_scale: &[f64], bn_var: &[f64], bn_epsilon: f64) -> Vec<f64> {
    (0..C_OUT)
        .map(|c| bn_scale[c] / (bn_var[c] + bn_epsilon).sqrt())
        .collect()
}

fn build_conv_bn_graph_no_conv_bias() -> Graph {
    let weight = make_tensor(&[C_OUT, C_IN, 1, 1], vec![1.0; C_OUT * C_IN]);
    let (scale, bias, mean, var) = bn_params();
    let bn_scale = make_1d_tensor(scale);
    let bn_bias = make_1d_tensor(bias);
    let bn_mean = make_1d_tensor(mean);
    let bn_var = make_1d_tensor(var);

    build_graph! {
        name: "conv_bn_no_bias",

        inputs: {
            x: (FloatType::F32, &[1, C_IN, 4, 4]),
        },

        outputs: {
            y: (FloatType::F32, &[1, C_OUT, 4, 4]),
        },

        initializers: {
            weight = weight,
            bn_scale = bn_scale,
            bn_bias = bn_bias,
            bn_mean = bn_mean,
            bn_var = bn_var,
        },

        nodes: [
            { "Conv", Operator::Conv(conv_op()),
              [x, weight] => conv_out: &[1, C_OUT, 4, 4] },

            { "BN", Operator::BatchNormalization(BatchNormalization {
                epsilon: 0.0,
                momentum: 0.9,
            }),
              [conv_out, bn_scale, bn_bias, bn_mean, bn_var] => y: &[1, C_OUT, 4, 4] },
        ]
    }
}

fn build_conv_bn_graph_with_conv_bias() -> Graph {
    let weight = make_tensor(&[C_OUT, C_IN, 1, 1], vec![1.0; C_OUT * C_IN]);
    let conv_bias = make_1d_tensor(vec![0.1, 0.2]);
    let (scale, bias, mean, var) = bn_params();
    let bn_scale = make_1d_tensor(scale);
    let bn_bias = make_1d_tensor(bias);
    let bn_mean = make_1d_tensor(mean);
    let bn_var = make_1d_tensor(var);

    build_graph! {
        name: "conv_bn_with_bias",

        inputs: {
            x: (FloatType::F32, &[1, C_IN, 4, 4]),
        },

        outputs: {
            y: (FloatType::F32, &[1, C_OUT, 4, 4]),
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
            { "Conv", Operator::Conv(conv_op()),
              [x, weight, conv_bias] => conv_out: &[1, C_OUT, 4, 4] },

            { "BN", Operator::BatchNormalization(BatchNormalization {
                epsilon: 0.0,
                momentum: 0.9,
            }),
              [conv_out, bn_scale, bn_bias, bn_mean, bn_var] => y: &[1, C_OUT, 4, 4] },
        ]
    }
}

fn run_conv_bn_fusion(graph: &mut Graph) {
    let mut modifier = SimpleGraphOp::new(graph);
    ConvBNFusion::default().run(graph, &mut modifier);
    modifier.update_deleted_nodes(graph);
}

#[test]
fn test_conv_bn_fused_no_conv_bias() {
    let mut graph = build_conv_bn_graph_no_conv_bias();
    run_conv_bn_fusion(&mut graph);

    let weight_data = vec![1.0; C_OUT * C_IN];
    let (scale, bias, mean, var) = bn_params();
    let multiplier = compute_multiplier(&scale, &var, 0.0);
    let fused_w = compute_fused_weight(&weight_data, &multiplier);
    let fused_b = compute_fused_bias(None, &scale, &bias, &mean, &var, 0.0);

    let fused_weight = make_tensor(&[C_OUT, C_IN, 1, 1], fused_w);
    let fused_bias = make_1d_tensor(fused_b);

    let expected = build_graph! {
        name: "expected",

        inputs: {
            x: (FloatType::F32, &[1, C_IN, 4, 4]),
        },

        outputs: {
            y: (FloatType::F32, &[1, C_OUT, 4, 4]),
        },

        initializers: {
            fused_weight = fused_weight,
            fused_bias = fused_bias,
        },

        nodes: [
            { "Conv", Operator::Conv(conv_op()),
              [x, fused_weight, fused_bias] => y: &[1, C_OUT, 4, 4] },
        ]
    };

    compare_graphs_structural_with_epsilon(&graph, &expected, EPSILON).unwrap();
}

#[test]
fn test_conv_bn_fused_with_conv_bias() {
    let mut graph = build_conv_bn_graph_with_conv_bias();
    run_conv_bn_fusion(&mut graph);

    let weight_data = vec![1.0; C_OUT * C_IN];
    let conv_bias_data = vec![0.1, 0.2];
    let (scale, bias, mean, var) = bn_params();
    let multiplier = compute_multiplier(&scale, &var, 0.0);
    let fused_w = compute_fused_weight(&weight_data, &multiplier);
    let fused_b = compute_fused_bias(Some(&conv_bias_data), &scale, &bias, &mean, &var, 0.0);

    let fused_weight = make_tensor(&[C_OUT, C_IN, 1, 1], fused_w);
    let fused_bias = make_1d_tensor(fused_b);

    let expected = build_graph! {
        name: "expected",

        inputs: {
            x: (FloatType::F32, &[1, C_IN, 4, 4]),
        },

        outputs: {
            y: (FloatType::F32, &[1, C_OUT, 4, 4]),
        },

        initializers: {
            fused_weight = fused_weight,
            fused_bias = fused_bias,
        },

        nodes: [
            { "Conv", Operator::Conv(conv_op()),
              [x, fused_weight, fused_bias] => y: &[1, C_OUT, 4, 4] },
        ]
    };

    compare_graphs_structural_with_epsilon(&graph, &expected, EPSILON).unwrap();
}

#[test]
fn test_conv_bn_not_fused_when_multiple_users() {
    let weight = make_tensor(&[C_OUT, C_IN, 1, 1], vec![1.0; C_OUT * C_IN]);
    let (scale, bias, mean, var) = bn_params();
    let bn_scale = make_1d_tensor(scale);
    let bn_bias = make_1d_tensor(bias);
    let bn_mean = make_1d_tensor(mean);
    let bn_var = make_1d_tensor(var);

    let mut graph = build_graph! {
        name: "conv_bn_multi_user",

        inputs: {
            x: (FloatType::F32, &[1, C_IN, 4, 4]),
        },

        outputs: {
            y1: (FloatType::F32, &[1, C_OUT, 4, 4]),
            y2: (FloatType::F32, &[1, C_OUT, 4, 4]),
        },

        initializers: {
            weight = weight.clone(),
            bn_scale = bn_scale.clone(),
            bn_bias = bn_bias.clone(),
            bn_mean = bn_mean.clone(),
            bn_var = bn_var.clone(),
        },

        nodes: [
            { "Conv", Operator::Conv(conv_op()),
              [x, weight] => conv_out: &[1, C_OUT, 4, 4] },

            { "BN", Operator::BatchNormalization(BatchNormalization {
                epsilon: 0.0,
                momentum: 0.9,
            }),
              [conv_out, bn_scale, bn_bias, bn_mean, bn_var] => y1: &[1, C_OUT, 4, 4] },

            { "Identity", Operator::Identity,
              [conv_out] => y2: &[1, C_OUT, 4, 4] },
        ]
    };

    let expected = build_graph! {
        name: "expected",

        inputs: {
            x: (FloatType::F32, &[1, C_IN, 4, 4]),
        },

        outputs: {
            y1: (FloatType::F32, &[1, C_OUT, 4, 4]),
            y2: (FloatType::F32, &[1, C_OUT, 4, 4]),
        },

        initializers: {
            weight = weight,
            bn_scale = bn_scale,
            bn_bias = bn_bias,
            bn_mean = bn_mean,
            bn_var = bn_var,
        },

        nodes: [
            { "Conv", Operator::Conv(conv_op()),
              [x, weight] => conv_out: &[1, C_OUT, 4, 4] },

            { "BN", Operator::BatchNormalization(BatchNormalization {
                epsilon: 0.0,
                momentum: 0.9,
            }),
              [conv_out, bn_scale, bn_bias, bn_mean, bn_var] => y1: &[1, C_OUT, 4, 4] },

            { "Identity", Operator::Identity,
              [conv_out] => y2: &[1, C_OUT, 4, 4] },
        ]
    };

    run_conv_bn_fusion(&mut graph);

    compare_graphs_structural(&graph, &expected).unwrap();
}
