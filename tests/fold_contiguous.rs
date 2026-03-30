mod common;

use std::collections::HashMap;

use common::create_value;
use my_onnx::onnx::model::Graph;
use my_onnx::onnx::model::Node;
use my_onnx::onnx::model::ValueId;
use my_onnx::onnx::operator::*;
use my_onnx::onnx::utils::compare_graphs_structural;
use my_onnx::tensor::types::DataType;
use my_onnx::tensor::types::FloatType;
use my_onnx::transform::modify::NodeDelete;
use my_onnx::transform::modify::SimpleGraphOp;
use my_onnx::transform::utils::ContiguousFolding;
use my_onnx::transform::Pass;

fn run_pass(graph: &mut Graph) {
    let mut modifier = SimpleGraphOp::new(graph);
    let pass = ContiguousFolding::default();
    pass.run(graph, &mut modifier);
    modifier.update_deleted_nodes(graph);
}

fn transpose_ops(perm: Vec<usize>) -> Vec<ReinterpretType> {
    vec![ReinterpretType::Transpose(Transpose { perm: Some(perm) })]
}

fn reshape_ops(before: Vec<usize>, after: Vec<usize>) -> Vec<ReinterpretType> {
    vec![ReinterpretType::single_reshape(before, after)]
}

// Reinterpret -> Contiguous => Contiguous(reinterpret_ops)
#[test]
fn test_reinterpret_then_contiguous() {
    let re_ops = transpose_ops(vec![0, 2, 1, 3]);

    let mut graph = build_graph! {
        name: "reinterpret_cont",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: { out: (FloatType::F32, &[2, 8, 4, 16]) },
        initializers: {},
        nodes: [
            { "Reinterpret", Operator::Reinterpret(Reinterpret { ops: re_ops.clone() }),
              [x] => reinterpreted: &[2, 8, 4, 16] },
            { "Contiguous", Operator::Contiguous(Contiguous { ops: vec![] }),
              [reinterpreted] => out: &[2, 8, 4, 16] },
        ]
    };

    run_pass(&mut graph);

    let expected = build_graph! {
        name: "expected",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: { out: (FloatType::F32, &[2, 8, 4, 16]) },
        initializers: {},
        nodes: [
            { "Contiguous", Operator::Contiguous(Contiguous { ops: re_ops }),
              [x] => out: &[2, 8, 4, 16] },
        ]
    };

    compare_graphs_structural(&graph, &expected).unwrap();
}

// Contiguous -> Reinterpret => Contiguous(reinterpret_ops)
#[test]
fn test_contiguous_then_reinterpret() {
    let re_ops = reshape_ops(vec![2, 4, 8, 16], vec![2, 32, 16]);

    let mut graph = build_graph! {
        name: "cont_reinterpret",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: { out: (FloatType::F32, &[2, 32, 16]) },
        initializers: {},
        nodes: [
            { "Contiguous", Operator::Contiguous(Contiguous { ops: vec![] }),
              [x] => cont_out: &[2, 4, 8, 16] },
            { "Reinterpret", Operator::Reinterpret(Reinterpret { ops: re_ops.clone() }),
              [cont_out] => out: &[2, 32, 16] },
        ]
    };

    run_pass(&mut graph);

    let expected = build_graph! {
        name: "expected",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: { out: (FloatType::F32, &[2, 32, 16]) },
        initializers: {},
        nodes: [
            { "Contiguous", Operator::Contiguous(Contiguous { ops: re_ops }),
              [x] => out: &[2, 32, 16] },
        ]
    };

    compare_graphs_structural(&graph, &expected).unwrap();
}

// Reinterpret -> Contiguous -> Reinterpret => Contiguous(before + after)
#[test]
fn test_reinterpret_contiguous_reinterpret() {
    let before_ops = transpose_ops(vec![0, 2, 1, 3]);
    let after_ops = reshape_ops(vec![2, 8, 4, 16], vec![2, 32, 16]);

    let mut graph = build_graph! {
        name: "re_cont_re",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: { out: (FloatType::F32, &[2, 32, 16]) },
        initializers: {},
        nodes: [
            { "Reinterpret1", Operator::Reinterpret(Reinterpret { ops: before_ops.clone() }),
              [x] => re1_out: &[2, 8, 4, 16] },
            { "Contiguous", Operator::Contiguous(Contiguous { ops: vec![] }),
              [re1_out] => cont_out: &[2, 8, 4, 16] },
            { "Reinterpret2", Operator::Reinterpret(Reinterpret { ops: after_ops.clone() }),
              [cont_out] => out: &[2, 32, 16] },
        ]
    };

    run_pass(&mut graph);

    let mut combined_ops = before_ops;
    combined_ops.extend(after_ops);

    let expected = build_graph! {
        name: "expected",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: { out: (FloatType::F32, &[2, 32, 16]) },
        initializers: {},
        nodes: [
            { "Contiguous", Operator::Contiguous(Contiguous { ops: combined_ops }),
              [x] => out: &[2, 32, 16] },
        ]
    };

    compare_graphs_structural(&graph, &expected).unwrap();
}

// Contiguous -> Reinterpret -> Contiguous => Contiguous(reinterpret_ops)
#[test]
fn test_contiguous_reinterpret_contiguous() {
    let re_ops = transpose_ops(vec![0, 2, 1, 3]);

    let mut graph = build_graph! {
        name: "cont_re_cont",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: { out: (FloatType::F32, &[2, 8, 4, 16]) },
        initializers: {},
        nodes: [
            { "Contiguous1", Operator::Contiguous(Contiguous { ops: vec![] }),
              [x] => cont1_out: &[2, 4, 8, 16] },
            { "Reinterpret", Operator::Reinterpret(Reinterpret { ops: re_ops.clone() }),
              [cont1_out] => re_out: &[2, 8, 4, 16] },
            { "Contiguous2", Operator::Contiguous(Contiguous { ops: vec![] }),
              [re_out] => out: &[2, 8, 4, 16] },
        ]
    };

    run_pass(&mut graph);

    let expected = build_graph! {
        name: "expected",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: { out: (FloatType::F32, &[2, 8, 4, 16]) },
        initializers: {},
        nodes: [
            { "Contiguous", Operator::Contiguous(Contiguous { ops: re_ops }),
              [x] => out: &[2, 8, 4, 16] },
        ]
    };

    compare_graphs_structural(&graph, &expected).unwrap();
}

// Standalone Contiguous => unchanged
#[test]
fn test_no_fold_standalone() {
    let mut graph = build_graph! {
        name: "standalone",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: { out: (FloatType::F32, &[2, 4, 8, 16]) },
        initializers: {},
        nodes: [
            { "Contiguous", Operator::Contiguous(Contiguous { ops: vec![] }),
              [x] => out: &[2, 4, 8, 16] },
        ]
    };

    run_pass(&mut graph);

    let expected = build_graph! {
        name: "expected",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: { out: (FloatType::F32, &[2, 4, 8, 16]) },
        initializers: {},
        nodes: [
            { "Contiguous", Operator::Contiguous(Contiguous { ops: vec![] }),
              [x] => out: &[2, 4, 8, 16] },
        ]
    };

    compare_graphs_structural(&graph, &expected).unwrap();
}

// Reinterpret (multiple users) -> Contiguous => unchanged
#[test]
fn test_no_fold_multi_user() {
    let re_ops = transpose_ops(vec![0, 2, 1, 3]);

    let mut graph = build_graph! {
        name: "multi_user",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: {
            out1: (FloatType::F32, &[2, 8, 4, 16]),
            out2: (FloatType::F32, &[2, 8, 4, 16]),
        },
        initializers: {},
        nodes: [
            { "Reinterpret", Operator::Reinterpret(Reinterpret { ops: re_ops.clone() }),
              [x] => reinterpreted: &[2, 8, 4, 16] },
            { "Contiguous", Operator::Contiguous(Contiguous { ops: vec![] }),
              [reinterpreted] => out1: &[2, 8, 4, 16] },
            { "Add", Operator::Add,
              [reinterpreted, reinterpreted] => out2: &[2, 8, 4, 16] },
        ]
    };

    let expected = build_graph! {
        name: "expected",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: {
            out1: (FloatType::F32, &[2, 8, 4, 16]),
            out2: (FloatType::F32, &[2, 8, 4, 16]),
        },
        initializers: {},
        nodes: [
            { "Reinterpret", Operator::Reinterpret(Reinterpret { ops: re_ops }),
              [x] => reinterpreted: &[2, 8, 4, 16] },
            { "Contiguous", Operator::Contiguous(Contiguous { ops: vec![] }),
              [reinterpreted] => out1: &[2, 8, 4, 16] },
            { "Add", Operator::Add,
              [reinterpreted, reinterpreted] => out2: &[2, 8, 4, 16] },
        ]
    };

    run_pass(&mut graph);

    compare_graphs_structural(&graph, &expected).unwrap();
}

// Contiguous(existing_ops) -> Reinterpret => Contiguous(existing + trailing)
#[test]
fn test_contiguous_with_existing_ops_and_trailing_reinterpret() {
    let existing_ops = transpose_ops(vec![0, 2, 1, 3]);
    let trailing_ops = reshape_ops(vec![2, 8, 4, 16], vec![2, 32, 16]);

    let mut graph = build_graph! {
        name: "existing_ops",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: { out: (FloatType::F32, &[2, 32, 16]) },
        initializers: {},
        nodes: [
            { "Contiguous", Operator::Contiguous(Contiguous { ops: existing_ops.clone() }),
              [x] => cont_out: &[2, 8, 4, 16] },
            { "Reinterpret", Operator::Reinterpret(Reinterpret { ops: trailing_ops.clone() }),
              [cont_out] => out: &[2, 32, 16] },
        ]
    };

    run_pass(&mut graph);

    let mut combined_ops = existing_ops;
    combined_ops.extend(trailing_ops);

    let expected = build_graph! {
        name: "expected",
        inputs: { x: (FloatType::F32, &[2, 4, 8, 16]) },
        outputs: { out: (FloatType::F32, &[2, 32, 16]) },
        initializers: {},
        nodes: [
            { "Contiguous", Operator::Contiguous(Contiguous { ops: combined_ops }),
              [x] => out: &[2, 32, 16] },
        ]
    };

    compare_graphs_structural(&graph, &expected).unwrap();
}
