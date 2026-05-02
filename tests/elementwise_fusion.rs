mod common;

use std::collections::HashMap;

use common::create_value;
use my_nn_engine::graph::operator::*;
use my_nn_engine::graph::Graph;
use my_nn_engine::graph::Node;
use my_nn_engine::graph::ValueId;
use my_nn_engine::options::*;
use my_nn_engine::schedule::*;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::tensor::types::DataType;
use my_nn_engine::tensor::types::FloatType;
use my_nn_engine::transform::transform_graph;

fn count_elementwise_kernels(schedule: &Schedule) -> usize {
    schedule
        .kernels
        .iter()
        .filter(|(_, k)| matches!(k.body, KernelBody::ElementWises(_)))
        .count()
}

fn count_opaque_kernels(schedule: &Schedule, pred: impl Fn(&Operator) -> bool) -> usize {
    schedule
        .kernels
        .iter()
        .filter(|(_, k)| match &k.body {
            KernelBody::Opaque(o) => pred(&o.op),
            _ => false,
        })
        .count()
}

// Input:
//   x -> Sigmoid -> Exp -> Tanh -> Log -> y
//
// Output:
//   x -> [ElementWises] -> y
#[test]
fn test_chain_single() {
    let mut graph = build_graph! {
        name: "chain_single",
        inputs: { x: (FloatType::F32, &[2, 3]) },
        outputs: { y: (FloatType::F32, &[2, 3]) },
        initializers: {},
        nodes: [
            { "Sigmoid", Operator::Sigmoid, [x] => a: &[2, 3] },
            { "Exp", Operator::Exp, [a] => b: &[2, 3] },
            { "Tanh", Operator::Tanh, [b] => c: &[2, 3] },
            { "Log", Operator::Log, [c] => y: &[2, 3] },
        ]
    };

    let options = Options::builder().build();
    transform_graph(&mut graph, &options, &SessionConfig::default());
    let schedule = Schedule::new(graph, options);
    assert_eq!(count_elementwise_kernels(&schedule), 1);
}

// Input:
//   x -> Sigmoid -> Exp -> y0 -> Tanh -> Log -> y1
//
// Output:
//   x -> [ElementWises] -> y0 -> [ElementWises] -> y1
#[test]
fn test_chain_branch() {
    let mut graph = build_graph! {
        name: "chain_branch",
        inputs: { x: (FloatType::F32, &[2, 3]) },
        outputs: {
            y0: (FloatType::F32, &[2, 3]),
            y1: (FloatType::F32, &[2, 3]),
        },
        initializers: {},
        nodes: [
            { "Sigmoid", Operator::Sigmoid, [x] => a: &[2, 3] },
            { "Exp", Operator::Exp, [a] => y0: &[2, 3] },
            { "Tanh", Operator::Tanh, [y0] => b: &[2, 3] },
            { "Log", Operator::Log, [b] => y1: &[2, 3] },
        ]
    };

    let options = Options::builder().build();
    transform_graph(&mut graph, &options, &SessionConfig::default());
    let schedule = Schedule::new(graph, options);
    assert_eq!(count_elementwise_kernels(&schedule), 2);
}

// Input:
//            +-> Exp -> Tanh -+
//            |                v
//   x -> Sigmoid         Add(c,c) -> Log -+
//   |        |                             v
//   |        +----------------------> Add(a,e) -+
//   |                                           v
//   +-------------------------------------> MatMul -> y
//
// Output:
//   x -> [ElementWises] -+
//   |                     v
//   +---------------> [Gemm] -> y
#[test]
fn test_elementwise_complex() {
    let mut graph = build_graph! {
        name: "elementwise_complex",
        inputs: { x: (FloatType::F32, &[4, 4]) },
        outputs: { y: (FloatType::F32, &[4, 4]) },
        initializers: {},
        nodes: [
            { "Sigmoid", Operator::Sigmoid, [x] => a: &[4, 4] },
            { "Exp", Operator::Exp, [a] => b: &[4, 4] },
            { "Tanh", Operator::Tanh, [b] => c: &[4, 4] },
            { "Add", Operator::Add, [c, c] => d: &[4, 4] },
            { "Log", Operator::Log, [d] => e: &[4, 4] },
            { "Add2", Operator::Add, [a, e] => f: &[4, 4] },
            { "MatMul", Operator::MatMul, [x, f] => y: &[4, 4] },
        ]
    };

    let options = Options::builder().build();
    transform_graph(&mut graph, &options, &SessionConfig::default());
    let schedule = Schedule::new(graph, options);
    assert_eq!(count_elementwise_kernels(&schedule), 1);
    assert_eq!(
        count_opaque_kernels(&schedule, |op| matches!(op, Operator::Gemm(_))),
        1
    );
}
