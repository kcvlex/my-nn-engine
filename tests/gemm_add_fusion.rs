mod common;

use std::collections::HashMap;

use common::create_value;
use common::find_nodes;
use common::make_1d_tensor;
use my_onnx::onnx::model::Graph;
use my_onnx::onnx::model::Node;
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
use my_onnx::transform::optimize::const_fold::ConstantFold;
use my_onnx::transform::optimize::gemm_add_fusion::GemmAddFusion;
use my_onnx::transform::Pass;

// Gemm([M, K] x [K, N]) -> Add([M, N], [M, N]) — same shape, original behavior
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

// Gemm([M, K] x [K, N]) -> Add([M, N], [N]) — broadcast bias (im2col pattern)
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

#[test]
fn test_gemm_add_same_shape_is_fused() {
    let mut graph = build_gemm_add_same_shape();

    let mut modifier = SimpleGraphOp::new(&graph);
    GemmAddFusion::default().run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let adds = find_nodes(&graph, |n| matches!(n.op, Operator::Add));
    assert_eq!(adds.len(), 0);

    let gemms = find_nodes(&graph, |n| matches!(n.op, Operator::Gemm(_)));
    assert_eq!(gemms.len(), 1);

    let gemm_node = &graph.nodes[gemms[0]];
    let Operator::Gemm(ref gemm) = gemm_node.op else {
        panic!()
    };
    assert_eq!(gemm.beta, 1.0);
    assert_eq!(gemm_node.inputs.len(), 3);

    // No Contiguous needed — bias was same shape
    let conts = find_nodes(&graph, |n| matches!(n.op, Operator::Contiguous(_)));
    assert_eq!(conts.len(), 0);
}

#[test]
fn test_gemm_add_broadcast_is_fused() {
    let mut graph = build_gemm_add_broadcast_bias();

    let mut modifier = SimpleGraphOp::new(&graph);
    GemmAddFusion::default().run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let adds = find_nodes(&graph, |n| matches!(n.op, Operator::Add));
    assert_eq!(adds.len(), 0);

    let gemms = find_nodes(&graph, |n| matches!(n.op, Operator::Gemm(_)));
    assert_eq!(gemms.len(), 1);

    let gemm_node = &graph.nodes[gemms[0]];
    let Operator::Gemm(ref gemm) = gemm_node.op else {
        panic!()
    };
    assert_eq!(gemm.beta, 1.0);
    assert_eq!(gemm_node.inputs.len(), 3);

    // A Contiguous node is inserted for broadcast
    let conts = find_nodes(&graph, |n| matches!(n.op, Operator::Contiguous(_)));
    assert_eq!(conts.len(), 1);

    // After ConstantFold, the Contiguous should be folded away
    ConstantFold {
        check_strides: true,
    }
    .run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let conts = find_nodes(&graph, |n| matches!(n.op, Operator::Contiguous(_)));
    assert_eq!(conts.len(), 0);

    // The Gemm C input should now be a [4, 8] initializer
    let gemm_node = &graph.nodes[gemms[0]];
    let c_value = gemm_node.inputs[args::GEMM_C];
    let c_tensor = graph.initializer.get(&c_value).unwrap();
    assert_eq!(c_tensor.dims, ResolvedTensorDims::new(&[4, 8]));
    let TensorData::Float(_, ref data) = c_tensor.data else {
        panic!()
    };
    assert!(data.iter().all(|v| (*v - 1.0).abs() < 1e-6));
}
