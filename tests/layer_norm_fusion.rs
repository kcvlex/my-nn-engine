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

// Extracted from GPT-2
fn build_gpt2_layer_norm_subgraph(epsilon: f64) -> Graph {
    let epsilon = ScalarData::Float(FloatType::F32, epsilon).to_tensor_data(1);
    let epsilon = Tensor::new(ResolvedTensorDims::new(&[]), epsilon).unwrap();

    build_graph! {
        name: "gpt2_layer_norm",

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

// Extracted from BERT (bertsquad-12)
fn build_bert_layer_norm_subgraph(epsilon: f64) -> Graph {
    let epsilon = ScalarData::Float(FloatType::F32, epsilon).to_tensor_data(1);
    let epsilon = Tensor::new(ResolvedTensorDims::new(&[]), epsilon).unwrap();

    build_graph! {
        name: "bert_layer_norm",

        inputs: {
            x: (FloatType::F32, &[2, 10, 768]),
            gamma: (FloatType::F32, &[768]),
            beta: (FloatType::F32, &[768]),
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

            { "Reciprocal", Operator::Reciprocal,
              [std_out] => inv_stddev: &[2, 10, 1] },

            { "ScaleInv", Operator::Mul,
              [inv_stddev, gamma] => scale_inv: &[2, 10, 768] },

            { "MulX", Operator::Mul,
              [x, scale_inv] => y_unbiased: &[2, 10, 768] },

            { "MulMean", Operator::Mul,
              [mean_out, scale_inv] => mean_scaled: &[2, 10, 768] },

            { "EffBias", Operator::Sub,
              [beta, mean_scaled] => eff_bias: &[2, 10, 768] },

            { "Y", Operator::Add,
              [y_unbiased, eff_bias] => y: &[2, 10, 768] },
        ]
    }
}

#[test]
fn test_gpt2_layer_norm_is_fused() {
    let epsilon = 1e-5;
    let mut graph = build_gpt2_layer_norm_subgraph(epsilon);

    let mut modifier = SimpleGraphOp::new(&graph);
    let canonicalize = Canonicalize::default();
    canonicalize.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);
    let pass = LayerNormFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let expected = build_graph! {
        name: "expected",
        inputs: {
            x: (FloatType::F32, &[2, 10, 768]),
            scale: (FloatType::F32, &[768]),
            bias: (FloatType::F32, &[768]),
        },
        outputs: {
            y: (FloatType::F32, &[2, 10, 768]),
        },
        initializers: {},
        nodes: [
            { "LayerNorm", Operator::LayerNormalization(LayerNormalization { axis: TensorIndex::new(-1), epsilon }),
              [x, scale, bias] => y: &[2, 10, 768] },
        ]
    };
    compare_graphs_structural(&graph, &expected).unwrap();
}

#[test]
fn test_bert_layer_norm_is_fused() {
    let epsilon = 1e-5;
    let mut graph = build_bert_layer_norm_subgraph(epsilon);

    let mut modifier = SimpleGraphOp::new(&graph);
    let pass = LayerNormFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let expected = build_graph! {
        name: "expected",
        inputs: {
            x: (FloatType::F32, &[2, 10, 768]),
            gamma: (FloatType::F32, &[768]),
            beta: (FloatType::F32, &[768]),
        },
        outputs: {
            y: (FloatType::F32, &[2, 10, 768]),
        },
        initializers: {},
        nodes: [
            { "LayerNorm", Operator::LayerNormalization(LayerNormalization { axis: TensorIndex::new(-1), epsilon }),
              [x, gamma, beta] => y: &[2, 10, 768] },
        ]
    };
    compare_graphs_structural(&graph, &expected).unwrap();
}
