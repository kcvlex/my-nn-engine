mod common;

use std::collections::HashMap;

use common::create_value;
use common::find_nodes;
use my_onnx::onnx::model::Graph;
use my_onnx::onnx::model::Node;
use my_onnx::onnx::model::ValueId;
use my_onnx::onnx::operator::*;
use my_onnx::tensor::types::DataType;
use my_onnx::tensor::types::FloatType;
use my_onnx::transform::epilog::fold_cont::FoldContiguous;
use my_onnx::transform::modify::NodeDelete;
use my_onnx::transform::modify::SimpleGraphOp;
use my_onnx::transform::Pass;

fn run_pass(graph: &mut Graph) {
    let mut modifier = SimpleGraphOp::new(graph);
    let pass = FoldContiguous::default();
    pass.run(graph, &mut modifier);
    modifier.update_deleted_nodes(graph);
}

fn transpose_ops(perm: Vec<usize>) -> Vec<ReinterpretType> {
    vec![ReinterpretType::Transpose(Transpose { perm: Some(perm) })]
}

fn reshape_ops(before: Vec<usize>, after: Vec<usize>) -> Vec<ReinterpretType> {
    vec![ReinterpretType::single_reshape(before, after)]
}

// Reinterpret -> Contiguous
// =>
// Contiguous
#[test]
fn test_reinterpret_then_contiguous() {
    let re_ops = transpose_ops(vec![0, 2, 1, 3]);

    let mut graph = build_graph! {
        name: "reinterpret_cont",
        inputs: {
            x: (FloatType::F32, &[2, 4, 8, 16]),
        },
        outputs: {
            out: (FloatType::F32, &[2, 8, 4, 16]),
        },
        initializers: {},
        nodes: [
            { "Reinterpret", Operator::Reinterpret(Reinterpret { ops: re_ops.clone() }),
              [x] => reinterpreted: &[2, 8, 4, 16] },
            { "Contiguous", Operator::Contiguous(Contiguous { ops: vec![] }),
              [reinterpreted] => out: &[2, 8, 4, 16] },
        ]
    };

    run_pass(&mut graph);

    let cont_nodes = find_nodes(&graph, |node| matches!(&node.op, Operator::Contiguous(_)));
    assert_eq!(cont_nodes.len(), 1, "should have exactly one Contiguous");

    let cont = &graph.nodes[cont_nodes[0]];
    let Operator::Contiguous(Contiguous { ops }) = &cont.op else {
        panic!("expected Contiguous");
    };
    assert_eq!(ops, &re_ops, "Contiguous should absorb the Reinterpret ops");
}

// Contiguous -> Reinterpret
// =>
// Contiguous
#[test]
fn test_contiguous_then_reinterpret() {
    let re_ops = reshape_ops(vec![2, 4, 8, 16], vec![2, 32, 16]);

    let mut graph = build_graph! {
        name: "cont_reinterpret",
        inputs: {
            x: (FloatType::F32, &[2, 4, 8, 16]),
        },
        outputs: {
            out: (FloatType::F32, &[2, 32, 16]),
        },
        initializers: {},
        nodes: [
            { "Contiguous", Operator::Contiguous(Contiguous { ops: vec![] }),
              [x] => cont_out: &[2, 4, 8, 16] },
            { "Reinterpret", Operator::Reinterpret(Reinterpret { ops: re_ops.clone() }),
              [cont_out] => out: &[2, 32, 16] },
        ]
    };

    run_pass(&mut graph);

    let cont_nodes = find_nodes(&graph, |node| matches!(&node.op, Operator::Contiguous(_)));
    assert_eq!(cont_nodes.len(), 1);

    let cont = &graph.nodes[cont_nodes[0]];
    let Operator::Contiguous(Contiguous { ops }) = &cont.op else {
        panic!("expected Contiguous");
    };
    assert_eq!(
        ops, &re_ops,
        "Contiguous should absorb the trailing Reinterpret ops"
    );
}

// Reinterpret -> Contiguous -> Reinterpret
// =>
// Contiguous
#[test]
fn test_reinterpret_contiguous_reinterpret() {
    let before_ops = transpose_ops(vec![0, 2, 1, 3]);
    let after_ops = reshape_ops(vec![2, 8, 4, 16], vec![2, 32, 16]);

    let mut graph = build_graph! {
        name: "re_cont_re",
        inputs: {
            x: (FloatType::F32, &[2, 4, 8, 16]),
        },
        outputs: {
            out: (FloatType::F32, &[2, 32, 16]),
        },
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

    let cont_nodes = find_nodes(&graph, |node| matches!(&node.op, Operator::Contiguous(_)));
    assert_eq!(cont_nodes.len(), 1);

    let cont = &graph.nodes[cont_nodes[0]];
    let Operator::Contiguous(Contiguous { ops }) = &cont.op else {
        panic!("expected Contiguous");
    };
    let mut expected = before_ops;
    expected.extend(after_ops);
    assert_eq!(ops, &expected, "should combine before and after ops");
}

// Contiguous -> Reinterpret -> Contiguous
// =>
// Contiguous
#[test]
fn test_contiguous_reinterpret_contiguous() {
    let re_ops = transpose_ops(vec![0, 2, 1, 3]);

    let mut graph = build_graph! {
        name: "cont_re_cont",
        inputs: {
            x: (FloatType::F32, &[2, 4, 8, 16]),
        },
        outputs: {
            out: (FloatType::F32, &[2, 8, 4, 16]),
        },
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

    let cont_nodes = find_nodes(&graph, |node| matches!(&node.op, Operator::Contiguous(_)));
    assert_eq!(
        cont_nodes.len(),
        1,
        "two Contiguous should be folded into one"
    );

    let cont = &graph.nodes[cont_nodes[0]];
    let Operator::Contiguous(Contiguous { ops }) = &cont.op else {
        panic!("expected Contiguous");
    };
    assert_eq!(ops, &re_ops, "should have the Reinterpret ops");
}

// Contiguous (standalone, no adjacent Reinterpret)
// =>
// Contiguous (unchanged)
#[test]
fn test_no_fold_standalone() {
    let mut graph = build_graph! {
        name: "standalone",
        inputs: {
            x: (FloatType::F32, &[2, 4, 8, 16]),
        },
        outputs: {
            out: (FloatType::F32, &[2, 4, 8, 16]),
        },
        initializers: {},
        nodes: [
            { "Contiguous", Operator::Contiguous(Contiguous { ops: vec![] }),
              [x] => out: &[2, 4, 8, 16] },
        ]
    };

    run_pass(&mut graph);

    let cont_nodes = find_nodes(&graph, |node| matches!(&node.op, Operator::Contiguous(_)));
    assert_eq!(cont_nodes.len(), 1);

    let Operator::Contiguous(Contiguous { ops }) = &graph.nodes[cont_nodes[0]].op else {
        panic!("expected Contiguous");
    };
    assert!(
        ops.is_empty(),
        "standalone Contiguous should remain unchanged"
    );
}

// Reinterpret (multiple users) -> Contiguous
// =>
// Reinterpret -> Contiguous (unchanged)
#[test]
fn test_no_fold_multi_user() {
    let re_ops = transpose_ops(vec![0, 2, 1, 3]);

    let mut graph = build_graph! {
        name: "multi_user",
        inputs: {
            x: (FloatType::F32, &[2, 4, 8, 16]),
        },
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

    run_pass(&mut graph);

    let cont_nodes = find_nodes(&graph, |node| matches!(&node.op, Operator::Contiguous(_)));
    assert_eq!(cont_nodes.len(), 1);

    let Operator::Contiguous(Contiguous { ops }) = &graph.nodes[cont_nodes[0]].op else {
        panic!("expected Contiguous");
    };
    assert!(
        ops.is_empty(),
        "should not fold when Reinterpret has multiple users"
    );
}

// Contiguous(existing_ops) -> Reinterpret
// =>
// Contiguous(existing_ops + trailing_ops)
#[test]
fn test_contiguous_with_existing_ops_and_trailing_reinterpret() {
    let existing_ops = transpose_ops(vec![0, 2, 1, 3]);
    let trailing_ops = reshape_ops(vec![2, 8, 4, 16], vec![2, 32, 16]);

    let mut graph = build_graph! {
        name: "existing_ops",
        inputs: {
            x: (FloatType::F32, &[2, 4, 8, 16]),
        },
        outputs: {
            out: (FloatType::F32, &[2, 32, 16]),
        },
        initializers: {},
        nodes: [
            { "Contiguous", Operator::Contiguous(Contiguous { ops: existing_ops.clone() }),
              [x] => cont_out: &[2, 8, 4, 16] },
            { "Reinterpret", Operator::Reinterpret(Reinterpret { ops: trailing_ops.clone() }),
              [cont_out] => out: &[2, 32, 16] },
        ]
    };

    run_pass(&mut graph);

    let cont_nodes = find_nodes(&graph, |node| matches!(&node.op, Operator::Contiguous(_)));
    assert_eq!(cont_nodes.len(), 1);

    let Operator::Contiguous(Contiguous { ops }) = &graph.nodes[cont_nodes[0]].op else {
        panic!("expected Contiguous");
    };
    let mut expected = existing_ops;
    expected.extend(trailing_ops);
    assert_eq!(
        ops, &expected,
        "should concatenate existing and trailing ops"
    );
}
