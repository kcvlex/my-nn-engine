mod common;

use std::collections::HashMap;

use my_nn_engine::graph::operator::*;
use my_nn_engine::graph::Graph;
use my_nn_engine::graph::Node;
use my_nn_engine::graph::ValueId;
use my_nn_engine::graph::ValueInfo;
use my_nn_engine::options::Options;
use my_nn_engine::options::PrefetchPolicy;
use my_nn_engine::options::Target;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::session::SessionError;
use my_nn_engine::tensor::data::TensorData;
use my_nn_engine::tensor::types::DataType;
use my_nn_engine::tensor::types::FloatType;
use my_nn_engine::tensor::types::ResolvedTensorDims;
use my_nn_engine::tensor::types::ResolvedTensorType;
use my_nn_engine::tensor::types::TensorType;
use my_nn_engine::tensor::Tensor;

use crate::common::create_value;

type TestResult = Result<(), SessionError>;

fn bf16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let bf16_bits = (bits >> 16) as u16;
    f32::from_bits((bf16_bits as u32) << 16)
}

fn make_bf16_tensor(dims: &[usize], values: &[f32]) -> Tensor {
    let data: Vec<f64> = values.iter().map(|&x| bf16_round(x) as f64).collect();
    Tensor::new(
        ResolvedTensorDims::new(dims),
        TensorData::Float(FloatType::BF16, data),
    )
    .unwrap()
}

fn extract_bf16(t: &Tensor) -> Vec<f32> {
    let TensorData::Float(FloatType::BF16, ref data) = t.data else {
        panic!("expected BF16 output, got {:?}", t.data.elem_type());
    };
    data.iter().map(|&x| x as f32).collect()
}

fn run_bf16_sigmoid(target: Target) -> TestResult {
    let bf16_ty: DataType = FloatType::BF16.into();
    let dims = &[4usize];

    let mut graph = Graph::empty_graph("bf16_sigmoid".to_string());
    let mut registry: HashMap<&str, ValueId> = HashMap::new();

    let x = create_value(&mut graph, "x".to_string(), dims, bf16_ty);
    registry.insert("x", x);
    let node = graph.nodes.alloc(Node::create_node(
        vec![],
        vec![x],
        "Input_x".to_string(),
        Operator::Input(x),
    ));
    graph.inputs.push(node);

    let y = graph.values.alloc(ValueInfo {
        name: "y".to_string(),
        ty: Some(TensorType::Resolved(ResolvedTensorType::new(
            bf16_ty,
            ResolvedTensorDims::new(dims),
        ))),
    });
    graph.nodes.alloc(Node::create_node(
        vec![Some(x)],
        vec![y],
        "sigmoid".to_string(),
        Operator::Sigmoid,
    ));
    let node = graph.nodes.alloc(Node::create_node(
        vec![Some(y)],
        vec![],
        "Output_y".to_string(),
        Operator::Output(y),
    ));
    graph.outputs.push(node);

    let opts = Options::builder().target(target).build();
    let mut session = Session::from_graph(graph, &opts, &SessionConfig::default())?;

    let x_vals = [-2.0f32, -0.5, 0.5, 2.0];
    let inputs = vec![make_bf16_tensor(dims, &x_vals)];
    let outputs = session.run(&inputs)?;
    let got = extract_bf16(&outputs[0]);

    let expected: Vec<f32> = x_vals
        .iter()
        .map(|&v| {
            let xf = bf16_round(v);
            bf16_round(1.0 / (1.0 + (-xf).exp()))
        })
        .collect();

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 5e-2,
            "bf16 sigmoid[{i}]: got {g}, expected {e}"
        );
    }
    Ok(())
}

fn run_bf16_rms_norm(target: Target) -> TestResult {
    let bf16_ty: DataType = FloatType::BF16.into();
    let dim_size = 8usize;
    let dims = &[1usize, dim_size];

    let mut graph = Graph::empty_graph("bf16_rms".to_string());

    let x = create_value(&mut graph, "x".to_string(), dims, bf16_ty);
    let node = graph.nodes.alloc(Node::create_node(
        vec![],
        vec![x],
        "Input_x".to_string(),
        Operator::Input(x),
    ));
    graph.inputs.push(node);

    let scale_vals: Vec<f64> = (0..dim_size).map(|_| bf16_round(1.0) as f64).collect();
    let scale_tensor = Tensor::new(
        ResolvedTensorDims::new(&[dim_size]),
        TensorData::Float(FloatType::BF16, scale_vals),
    )
    .unwrap();
    let scale = graph.values.alloc(ValueInfo {
        name: "scale".to_string(),
        ty: Some(TensorType::Resolved(scale_tensor.tensor_type())),
    });
    graph.set_initializer(scale, scale_tensor);

    let y = graph.values.alloc(ValueInfo {
        name: "y".to_string(),
        ty: Some(TensorType::Resolved(ResolvedTensorType::new(
            bf16_ty,
            ResolvedTensorDims::new(dims),
        ))),
    });
    graph.nodes.alloc(Node::create_node(
        vec![Some(x), Some(scale)],
        vec![y],
        "rms".to_string(),
        Operator::RMSNormalization(RMSNormalization {
            axis: TensorIndex::new(-1),
            epsilon: 1e-6,
        }),
    ));
    let node = graph.nodes.alloc(Node::create_node(
        vec![Some(y)],
        vec![],
        "Output_y".to_string(),
        Operator::Output(y),
    ));
    graph.outputs.push(node);

    let opts = Options::builder().target(target).build();
    let mut session = Session::from_graph(graph, &opts, &SessionConfig::default())?;

    let x_vals: Vec<f32> = vec![1.0, 2.0, -1.5, 0.5, 0.0, 3.0, -2.0, 1.0];
    let inputs = vec![make_bf16_tensor(dims, &x_vals)];
    let outputs = session.run(&inputs)?;
    let got = extract_bf16(&outputs[0]);

    let xs: Vec<f32> = x_vals.iter().map(|&x| bf16_round(x)).collect();
    let mean_sq: f32 = xs.iter().map(|&x| x * x).sum::<f32>() / dim_size as f32;
    let inv_std = 1.0 / (mean_sq + 1e-6f32).sqrt();
    let expected: Vec<f32> = xs.iter().map(|&x| bf16_round(x * inv_std)).collect();

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 5e-2,
            "bf16 rms_norm[{i}]: got {g}, expected {e}"
        );
    }
    Ok(())
}

fn run_bf16_matmul(target: Target) -> TestResult {
    let bf16_ty: DataType = FloatType::BF16.into();
    let a_dims = &[1usize, 4];
    let b_dims = &[4usize, 4];
    let out_dims = &[1usize, 4];

    let mut graph = Graph::empty_graph("bf16_matmul".to_string());

    let a = create_value(&mut graph, "a".to_string(), a_dims, bf16_ty);
    let node = graph.nodes.alloc(Node::create_node(
        vec![],
        vec![a],
        "Input_a".to_string(),
        Operator::Input(a),
    ));
    graph.inputs.push(node);
    let b = create_value(&mut graph, "b".to_string(), b_dims, bf16_ty);
    let node = graph.nodes.alloc(Node::create_node(
        vec![],
        vec![b],
        "Input_b".to_string(),
        Operator::Input(b),
    ));
    graph.inputs.push(node);

    let c = graph.values.alloc(ValueInfo {
        name: "c".to_string(),
        ty: Some(TensorType::Resolved(ResolvedTensorType::new(
            bf16_ty,
            ResolvedTensorDims::new(out_dims),
        ))),
    });
    graph.nodes.alloc(Node::create_node(
        vec![Some(a), Some(b)],
        vec![c],
        "matmul".to_string(),
        Operator::MatMul,
    ));
    let node = graph.nodes.alloc(Node::create_node(
        vec![Some(c)],
        vec![],
        "Output_c".to_string(),
        Operator::Output(c),
    ));
    graph.outputs.push(node);

    let opts = Options::builder().target(target).build();
    let mut session = Session::from_graph(graph, &opts, &SessionConfig::default())?;

    let a_vals = [1.0f32, 0.5, -1.0, 2.0];
    let b_vals = [
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let inputs = vec![
        make_bf16_tensor(a_dims, &a_vals),
        make_bf16_tensor(b_dims, &b_vals),
    ];
    let outputs = session.run(&inputs)?;
    let got = extract_bf16(&outputs[0]);
    let expected: Vec<f32> = a_vals.iter().map(|&x| bf16_round(x)).collect();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-2,
            "bf16 matmul[{i}]: got {g}, expected {e}"
        );
    }
    Ok(())
}

fn run_bf16_add(target: Target) -> TestResult {
    let bf16_ty: DataType = FloatType::BF16.into();
    let dims = &[4usize];

    let mut graph = Graph::empty_graph("bf16_add".to_string());
    let mut registry: HashMap<&str, ValueId> = HashMap::new();

    let a = create_value(&mut graph, "a".to_string(), dims, bf16_ty);
    registry.insert("a", a);
    let node = graph.nodes.alloc(Node::create_node(
        vec![],
        vec![a],
        "Input_a".to_string(),
        Operator::Input(a),
    ));
    graph.inputs.push(node);

    let b = create_value(&mut graph, "b".to_string(), dims, bf16_ty);
    registry.insert("b", b);
    let node = graph.nodes.alloc(Node::create_node(
        vec![],
        vec![b],
        "Input_b".to_string(),
        Operator::Input(b),
    ));
    graph.inputs.push(node);

    let c = graph.values.alloc(ValueInfo {
        name: "c".to_string(),
        ty: Some(TensorType::Resolved(ResolvedTensorType::new(
            bf16_ty,
            ResolvedTensorDims::new(dims),
        ))),
    });
    graph.nodes.alloc(Node::create_node(
        vec![Some(a), Some(b)],
        vec![c],
        "add".to_string(),
        Operator::Add,
    ));
    let node = graph.nodes.alloc(Node::create_node(
        vec![Some(c)],
        vec![],
        "Output_c".to_string(),
        Operator::Output(c),
    ));
    graph.outputs.push(node);

    let opts = Options::builder().target(target).build();
    let mut session = Session::from_graph(graph, &opts, &SessionConfig::default())?;

    let a_vals = [1.0f32, 2.0, 3.0, 4.0];
    let b_vals = [0.5f32, -1.0, 1.5, 2.5];

    let inputs = vec![
        make_bf16_tensor(dims, &a_vals),
        make_bf16_tensor(dims, &b_vals),
    ];
    let outputs = session.run(&inputs)?;
    let got = extract_bf16(&outputs[0]);

    let expected: Vec<f32> = a_vals
        .iter()
        .zip(b_vals.iter())
        .map(|(&a, &b)| bf16_round(bf16_round(a) + bf16_round(b)))
        .collect();

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-2, "bf16 add[{i}]: got {g}, expected {e}");
    }
    Ok(())
}

#[test]
fn bf16_sigmoid_cpu() -> TestResult {
    run_bf16_sigmoid(Target::CPU)
}

#[test]
fn bf16_rms_norm_cpu() -> TestResult {
    run_bf16_rms_norm(Target::CPU)
}

#[test]
fn bf16_add_cpu() -> TestResult {
    run_bf16_add(Target::CPU)
}

#[test]
fn bf16_matmul_cpu() -> TestResult {
    run_bf16_matmul(Target::CPU)
}

#[test]
#[cfg(feature = "cuda")]
fn bf16_sigmoid_cuda() -> TestResult {
    run_bf16_sigmoid(Target::CUDA(PrefetchPolicy::Disabled))
}

#[test]
#[cfg(feature = "cuda")]
fn bf16_rms_norm_cuda() -> TestResult {
    run_bf16_rms_norm(Target::CUDA(PrefetchPolicy::Disabled))
}

#[test]
#[cfg(feature = "cuda")]
fn bf16_matmul_cuda() -> TestResult {
    run_bf16_matmul(Target::CUDA(PrefetchPolicy::Disabled))
}

#[test]
#[cfg(feature = "cuda")]
fn bf16_add_cuda() -> TestResult {
    run_bf16_add(Target::CUDA(PrefetchPolicy::Disabled))
}
