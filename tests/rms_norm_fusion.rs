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
use my_onnx::transform::optimize::rms_norm_fusion::RMSNormFusion;
use my_onnx::transform::Pass;

fn scalar_f32(value: f64) -> Tensor {
    let data = ScalarData::Float(FloatType::F32, value).to_tensor_data(1);
    Tensor::new(ResolvedTensorDims::new(&[]), data).unwrap()
}

// Bare RMSNorm pattern (no surrounding Cast nodes):
//   Pow(X, 2) → ReduceMean<axes=[-1]> → Add(eps) → Sqrt → Div(1, Sqrt) → Mul(X, _) → Mul(scale, _)
fn build_basic_rms_norm_subgraph(epsilon: f64) -> Graph {
    let pow_two = scalar_f32(2.0);
    let one = scalar_f32(1.0);
    let eps = scalar_f32(epsilon);

    build_graph! {
        name: "rms_norm_basic",

        inputs: {
            x: (FloatType::F32, &[2, 10, 2048]),
            scale: (FloatType::F32, &[2048]),
        },

        outputs: {
            y: (FloatType::F32, &[2, 10, 2048]),
        },

        initializers: {
            pow_two = pow_two,
            one = one,
            eps = eps,
        },

        nodes: [
            { "Pow", Operator::Pow,
              [x, pow_two] => x_sq: &[2, 10, 2048] },

            { "MeanSq", Operator::ReduceMean(Reduce { axes: vec![-1], keepdims: true }),
              [x_sq] => mean_sq: &[2, 10, 1] },

            { "MeanSqEps", Operator::Add,
              [mean_sq, eps] => mean_sq_eps: &[2, 10, 1] },

            { "Sqrt", Operator::Sqrt,
              [mean_sq_eps] => stddev: &[2, 10, 1] },

            { "Recip", Operator::Div,
              [one, stddev] => recip: &[2, 10, 1] },

            { "Normalized", Operator::Mul,
              [x, recip] => normalized: &[2, 10, 2048] },

            { "Y", Operator::Mul,
              [scale, normalized] => y: &[2, 10, 2048] },
        ]
    }
}

// TinyLlama / HF Llama pattern with intermediate Cast (fp32→fp32 no-op when model is fp32 export):
//   Cast(X) [pre] → Pow → ReduceMean → Add(eps) → Sqrt → Div(1, Sqrt) → Mul(X_cast, _) → Cast → Mul(scale, _)
fn build_tinyllama_rms_norm_subgraph(epsilon: f64) -> Graph {
    let pow_two = scalar_f32(2.0);
    let one = scalar_f32(1.0);
    let eps = scalar_f32(epsilon);

    build_graph! {
        name: "rms_norm_tinyllama",

        inputs: {
            x: (FloatType::F32, &[2, 10, 2048]),
            scale: (FloatType::F32, &[2048]),
        },

        outputs: {
            y: (FloatType::F32, &[2, 10, 2048]),
        },

        initializers: {
            pow_two = pow_two,
            one = one,
            eps = eps,
        },

        nodes: [
            { "CastIn", Operator::Cast(Cast { to: DataType::Float(FloatType::F32) }),
              [x] => x_cast: &[2, 10, 2048] },

            { "Pow", Operator::Pow,
              [x_cast, pow_two] => x_sq: &[2, 10, 2048] },

            { "MeanSq", Operator::ReduceMean(Reduce { axes: vec![-1], keepdims: true }),
              [x_sq] => mean_sq: &[2, 10, 1] },

            { "MeanSqEps", Operator::Add,
              [mean_sq, eps] => mean_sq_eps: &[2, 10, 1] },

            { "Sqrt", Operator::Sqrt,
              [mean_sq_eps] => stddev: &[2, 10, 1] },

            { "Recip", Operator::Div,
              [one, stddev] => recip: &[2, 10, 1] },

            { "Normalized", Operator::Mul,
              [x_cast, recip] => normalized: &[2, 10, 2048] },

            { "CastOut", Operator::Cast(Cast { to: DataType::Float(FloatType::F32) }),
              [normalized] => normalized_cast: &[2, 10, 2048] },

            { "Y", Operator::Mul,
              [scale, normalized_cast] => y: &[2, 10, 2048] },
        ]
    }
}

#[test]
fn test_basic_rms_norm_is_fused() {
    let epsilon = 1e-6;
    let mut graph = build_basic_rms_norm_subgraph(epsilon);

    let mut modifier = SimpleGraphOp::new(&graph);
    let pass = RMSNormFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let expected = build_graph! {
        name: "expected",
        inputs: {
            x: (FloatType::F32, &[2, 10, 2048]),
            scale: (FloatType::F32, &[2048]),
        },
        outputs: {
            y: (FloatType::F32, &[2, 10, 2048]),
        },
        initializers: {},
        nodes: [
            { "RMSNorm", Operator::RMSNormalization(RMSNormalization { axis: TensorIndex::new(-1), epsilon }),
              [x, scale] => y: &[2, 10, 2048] },
        ]
    };
    compare_graphs_structural(&graph, &expected).unwrap();
}

#[test]
fn test_tinyllama_rms_norm_is_fused() {
    let epsilon = 1e-6;
    let mut graph = build_tinyllama_rms_norm_subgraph(epsilon);

    let mut modifier = SimpleGraphOp::new(&graph);
    let pass = RMSNormFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let expected = build_graph! {
        name: "expected",
        inputs: {
            x: (FloatType::F32, &[2, 10, 2048]),
            scale: (FloatType::F32, &[2048]),
        },
        outputs: {
            y: (FloatType::F32, &[2, 10, 2048]),
        },
        initializers: {},
        nodes: [
            { "CastIn", Operator::Cast(Cast { to: DataType::Float(FloatType::F32) }),
              [x] => x_cast: &[2, 10, 2048] },
            { "RMSNorm", Operator::RMSNormalization(RMSNormalization { axis: TensorIndex::new(-1), epsilon }),
              [x_cast, scale] => y: &[2, 10, 2048] },
        ]
    };
    compare_graphs_structural(&graph, &expected).unwrap();
}
