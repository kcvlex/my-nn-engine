mod common;

use std::collections::HashMap;

use common::create_value;
use common::make_1d_tensor;
use my_nn_engine::graph::operator::*;
use my_nn_engine::graph::utils::compare_graphs_structural_with_epsilon;
use my_nn_engine::graph::Graph;
use my_nn_engine::graph::Node;
use my_nn_engine::graph::ValueId;
use my_nn_engine::graph::ValueInfo;
use my_nn_engine::tensor::data::TensorData;
use my_nn_engine::tensor::types::DataType;
use my_nn_engine::tensor::types::FloatType;
use my_nn_engine::tensor::types::ResolvedTensorDims;
use my_nn_engine::tensor::types::TensorType;
use my_nn_engine::tensor::Tensor;
use my_nn_engine::transform::modify::NodeDelete;
use my_nn_engine::transform::modify::SimpleGraphOp;
use my_nn_engine::transform::optimize::const_folding::ConstantFolding;
use my_nn_engine::transform::optimize::gemm_add_fusion::GemmAddFusion;
use my_nn_engine::transform::Pass;

const EPSILON: f64 = 1e-6;

fn build_gemm_add_same_shape() -> Graph {
    let bias = Tensor::new(
        ResolvedTensorDims::new(&[4, 8]),
        TensorData::Float(FloatType::F32, vec![1.0; 32]),
    )
    .unwrap();

    build_graph! {
        name: "gemm_add_same_shape",

        inputs: {
            a: (FloatType::F32, &[4, 6]),
            b: (FloatType::F32, &[6, 8]),
        },

        outputs: {
            y: (FloatType::F32, &[4, 8]),
        },

        initializers: {
            bias = bias,
        },

        nodes: [
            { "Gemm", Operator::Gemm(Gemm { alpha: 1.0, beta: 0.0, trans_a: false, trans_b: false }),
              [a, b] => gemm_out: &[4, 8] },

            { "Add", Operator::Add,
              [gemm_out, bias] => y: &[4, 8] },
        ]
    }
}

fn build_gemm_add_broadcast_bias() -> Graph {
    let bias = make_1d_tensor(vec![1.0; 8]);

    build_graph! {
        name: "gemm_add_broadcast",

        inputs: {
            a: (FloatType::F32, &[4, 6]),
            b: (FloatType::F32, &[6, 8]),
        },

        outputs: {
            y: (FloatType::F32, &[4, 8]),
        },

        initializers: {
            bias = bias,
        },

        nodes: [
            { "Gemm", Operator::Gemm(Gemm { alpha: 1.0, beta: 0.0, trans_a: false, trans_b: false }),
              [a, b] => gemm_out: &[4, 8] },

            { "Add", Operator::Add,
              [gemm_out, bias] => y: &[4, 8] },
        ]
    }
}

fn run_gemm_add_fusion(graph: &mut Graph) {
    let mut modifier = SimpleGraphOp::new(graph);
    GemmAddFusion::default().run(graph, &mut modifier);
    modifier.update_deleted_nodes(graph);
}

fn run_const_fold(graph: &mut Graph) {
    let mut modifier = SimpleGraphOp::new(graph);
    ConstantFolding {
        check_strides: true,
    }
    .run(graph, &mut modifier);
    modifier.update_deleted_nodes(graph);
}

#[test]
fn test_gemm_add_same_shape_is_fused() {
    let mut graph = build_gemm_add_same_shape();
    run_gemm_add_fusion(&mut graph);

    let bias = Tensor::new(
        ResolvedTensorDims::new(&[4, 8]),
        TensorData::Float(FloatType::F32, vec![1.0; 32]),
    )
    .unwrap();

    let expected = build_graph! {
        name: "expected",

        inputs: {
            a: (FloatType::F32, &[4, 6]),
            b: (FloatType::F32, &[6, 8]),
        },

        outputs: {
            y: (FloatType::F32, &[4, 8]),
        },

        initializers: {
            bias = bias,
        },

        nodes: [
            { "Gemm", Operator::Gemm(Gemm { alpha: 1.0, beta: 1.0, trans_a: false, trans_b: false }),
              [a, b, bias] => y: &[4, 8] },
        ]
    };

    compare_graphs_structural_with_epsilon(&graph, &expected, EPSILON).unwrap();
}

#[test]
fn test_gemm_add_broadcast_is_fused() {
    let mut graph = build_gemm_add_broadcast_bias();
    run_gemm_add_fusion(&mut graph);
    run_const_fold(&mut graph);

    // After ConstantFold, broadcast bias [8] becomes [4, 8] filled with 1.0
    let bias = Tensor::new(
        ResolvedTensorDims::new(&[4, 8]),
        TensorData::Float(FloatType::F32, vec![1.0; 32]),
    )
    .unwrap();

    let expected = build_graph! {
        name: "expected",

        inputs: {
            a: (FloatType::F32, &[4, 6]),
            b: (FloatType::F32, &[6, 8]),
        },

        outputs: {
            y: (FloatType::F32, &[4, 8]),
        },

        initializers: {
            bias = bias,
        },

        nodes: [
            { "Gemm", Operator::Gemm(Gemm { alpha: 1.0, beta: 1.0, trans_a: false, trans_b: false }),
              [a, b, bias] => y: &[4, 8] },
        ]
    };

    compare_graphs_structural_with_epsilon(&graph, &expected, EPSILON).unwrap();
}
