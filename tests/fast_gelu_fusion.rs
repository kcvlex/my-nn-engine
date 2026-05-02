mod common;

use std::collections::HashMap;

use common::create_value;
use my_nn_engine::onnx::model::Graph;
use my_nn_engine::onnx::model::Node;
use my_nn_engine::onnx::model::ValueId;
use my_nn_engine::onnx::model::ValueInfo;
use my_nn_engine::onnx::operator::*;
use my_nn_engine::onnx::utils::compare_graphs_structural;
use my_nn_engine::tensor::data::ScalarData;
use my_nn_engine::tensor::types::DataType;
use my_nn_engine::tensor::types::FloatType;
use my_nn_engine::tensor::types::ResolvedTensorDims;
use my_nn_engine::tensor::types::TensorType;
use my_nn_engine::tensor::Tensor;
use my_nn_engine::transform::modify::NodeDelete;
use my_nn_engine::transform::modify::SimpleGraphOp;
use my_nn_engine::transform::optimize::fast_gelu_fusion::FastGeLUFusion;
use my_nn_engine::transform::Pass;

fn make_scalar(val: f64) -> Tensor {
    let data = ScalarData::Float(FloatType::F32, val).to_tensor_data(1);
    Tensor::new(ResolvedTensorDims::new(&[]), data).unwrap()
}

// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn build_unfused_fast_gelu_graph() -> Graph {
    let coeff = make_scalar(0.044715);
    let sqrt_2_over_pi = make_scalar((2.0f64 / std::f64::consts::PI).sqrt());
    let one = make_scalar(1.0);
    let half = make_scalar(0.5);
    let three = {
        let data = ScalarData::Float(FloatType::F32, 3.0).to_tensor_data(1);
        Tensor::new(ResolvedTensorDims::new(&[]), data).unwrap()
    };

    build_graph! {
        name: "fast_gelu_test",

        inputs: {
            x: (FloatType::F32, &[2, 10, 768]),
        },

        outputs: {
            y: (FloatType::F32, &[2, 10, 768]),
        },

        initializers: {
            coeff = coeff,
            sqrt_2_over_pi = sqrt_2_over_pi,
            one = one,
            half = half,
            three = three,
        },

        nodes: [
            // x^3
            { "Cube", Operator::Pow,
              [x, three] => cube_out: &[2, 10, 768] },

            // 0.044715 * x^3
            { "ScaledCube", Operator::Mul,
              [cube_out, coeff] => scaled_cube_out: &[2, 10, 768] },

            // x + 0.044715 * x^3
            { "InnerSum", Operator::Add,
              [x, scaled_cube_out] => inner_sum_out: &[2, 10, 768] },

            // sqrt(2/pi) * (x + 0.044715 * x^3)
            { "ScaledInner", Operator::Mul,
              [inner_sum_out, sqrt_2_over_pi] => scaled_inner_out: &[2, 10, 768] },

            // tanh(...)
            { "Tanh", Operator::Tanh,
              [scaled_inner_out] => tanh_out: &[2, 10, 768] },

            // 1 + tanh(...)
            { "OnePlusTanh", Operator::Add,
              [tanh_out, one] => one_plus_tanh_out: &[2, 10, 768] },

            // 0.5 * x
            { "HalfX", Operator::Mul,
              [x, half] => half_x_out: &[2, 10, 768] },

            // 0.5 * x * (1 + tanh(...))
            { "Y", Operator::Mul,
              [half_x_out, one_plus_tanh_out] => y: &[2, 10, 768] },
        ]
    }
}

#[test]
fn test_fast_gelu_is_fused() {
    let mut graph = build_unfused_fast_gelu_graph();

    let mut modifier = SimpleGraphOp::new(&graph);
    let pass = FastGeLUFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let expected = build_graph! {
        name: "expected",
        inputs: {
            x: (FloatType::F32, &[2, 10, 768]),
        },
        outputs: {
            y: (FloatType::F32, &[2, 10, 768]),
        },
        initializers: {},
        nodes: [
            { "GeLU", Operator::GeLU(GeLU { approximate: true }),
              [x] => y: &[2, 10, 768] },
        ]
    };
    compare_graphs_structural(&graph, &expected).unwrap();
}
