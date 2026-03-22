use itertools::Itertools;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::tensor::data::ScalarData;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct Canonicalize {}

impl<T: GraphOp> Pass<T> for Canonicalize {
    fn summary(&self) -> &'static str {
        "Canonicalize some patterns"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let mut ids = graph.nodes.iter().map(|(id, _)| id).collect_vec();
        ids.reverse();

        for id in ids {
            self.rewrite(id, graph, modifier);
        }
    }
}

impl Canonicalize {
    fn rewrite<T: GraphOp>(&self, id: NodeId, graph: &mut Graph, modifier: &mut T) {
        let node = &graph.nodes[id];
        match &node.op {
            Operator::Pow => {
                let exponent = node.inputs[1];
                let Some(tensor) = graph.initializer.get(&exponent) else {
                    return;
                };
                let Some(scalar) = tensor.data.to_scalar_data() else {
                    return;
                };
                if !matches!(
                    scalar,
                    ScalarData::SInt(_, 2) | ScalarData::UInt(_, 2) | ScalarData::Float(_, 2.0)
                ) {
                    return;
                }

                let new_inputs = vec![node.inputs[0], node.inputs[0]];
                let old_output = node.outputs[0];
                let new_output = modifier.register_new_value(
                    graph,
                    format!("Canonicalize_Square_{:?}", id),
                    graph.get_resolved_tensor_type(old_output).unwrap().clone(),
                );
                modifier.register_new_node(
                    graph,
                    Node::create_node(
                        new_inputs,
                        vec![new_output],
                        format!("Canonicalize_Square_{:?}", id),
                        Operator::Mul,
                    ),
                );
                modifier.replace_input_value(graph, old_output, new_output);
            }

            Operator::Div => {
                let lhs = node.inputs[0];
                let rhs = node.inputs[1];
                let output = node.outputs[0];

                let reciprocal = modifier.register_new_value(
                    graph,
                    format!("Canonicalize_Reciprocal_{:?}", id),
                    graph.get_resolved_tensor_type(rhs).unwrap().clone(),
                );
                modifier.register_new_node(
                    graph,
                    Node::create_node(
                        vec![rhs],
                        vec![reciprocal],
                        format!("Canonicalize_Reciprocal_{:?}", id),
                        Operator::Reciprocal,
                    ),
                );

                let new_output = modifier.register_new_value(
                    graph,
                    format!("Canonicalize_Mul_{:?}", id),
                    graph.get_resolved_tensor_type(output).unwrap().clone(),
                );
                modifier.register_new_node(
                    graph,
                    Node::create_node(
                        vec![lhs, reciprocal],
                        vec![new_output],
                        format!("Canonicalize_Mul_{:?}", id),
                        Operator::Mul,
                    ),
                );

                modifier.replace_input_value(graph, output, new_output);
            }

            Operator::MatMul => {
                let lhs = node.inputs[args::MATMUL_LHS];
                let rhs = node.inputs[args::MATMUL_RHS];
                let ldim = graph.get_resolved_tensor_type(lhs).unwrap().dims.ndim();
                let rdim = graph.get_resolved_tensor_type(rhs).unwrap().dims.ndim();
                let old_output = node.outputs[0];
                let ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
                // Only convert 2D MatMul to Gemm here.
                // 3D+ MatMul → BatchedGemm is done later (after AttentionFusion).
                if ldim != 2 || rdim != 2 {
                    return;
                }
                let name = format!("MatMul2Gemm_{:?}", id);
                let op = Operator::Gemm(Gemm::default());
                let new_output = modifier.register_new_value(graph, name.clone(), ty);
                modifier.register_new_node(
                    graph,
                    Node {
                        inputs: vec![lhs, rhs],
                        outputs: vec![new_output],
                        name,
                        op,
                        meta: NodeMeta::default(),
                    },
                );
                modifier.replace_input_value(graph, old_output, new_output);
            }

            _ => (),
        }
    }
}

#[derive(Default)]
pub struct MatMul2BatchedGemm {}

impl<T: GraphOp> Pass<T> for MatMul2BatchedGemm {
    fn summary(&self) -> &'static str {
        "Convert batched MatMul to BatchedGemm"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids: Vec<_> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if !matches!(node.op, Operator::MatMul) {
                    return None;
                }
                let lhs = node.inputs[args::MATMUL_LHS];
                let rhs = node.inputs[args::MATMUL_RHS];
                let ldim = graph.get_resolved_tensor_type(lhs)?.dims.ndim();
                let rdim = graph.get_resolved_tensor_type(rhs)?.dims.ndim();
                if ldim < 3 || rdim < 3 {
                    return None;
                }
                // BatchedGemm requires batch dims to match exactly.
                // If broadcast is needed (e.g. [2,3,4] @ [1,3,5]), keep as MatMul.
                let l_dims = &graph.get_resolved_tensor_type(lhs)?.dims;
                let r_dims = &graph.get_resolved_tensor_type(rhs)?.dims;
                if l_dims[..ldim - 2] != r_dims[..rdim - 2] {
                    return None;
                }
                Some(id)
            })
            .collect();

        for id in ids {
            let node = &graph.nodes[id];
            let lhs = node.inputs[args::MATMUL_LHS];
            let rhs = node.inputs[args::MATMUL_RHS];
            let old_output = node.outputs[0];
            let ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
            let name = format!("MatMul2BatchedGemm_{:?}", id);
            let new_output = modifier.register_new_value(graph, name.clone(), ty);
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![lhs, rhs],
                    outputs: vec![new_output],
                    name,
                    op: Operator::BatchedGemm(BatchedGemm {
                        alpha: 1.0,
                        beta: 0.0,
                        trans_a: false,
                        trans_b: false,
                    }),
                    meta: NodeMeta::default(),
                },
            );
            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}
