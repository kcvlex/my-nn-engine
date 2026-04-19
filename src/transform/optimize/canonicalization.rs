use itertools::Itertools;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::tensor::data::ScalarData;
use crate::transform::modify::GraphOp;
use crate::transform::utils::*;
use crate::transform::Pass;

#[derive(Default)]
pub struct Canonicalization {}

impl<T: GraphOp> Pass<T> for Canonicalization {
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

impl Canonicalization {
    fn rewrite<T: GraphOp>(&self, id: NodeId, graph: &mut Graph, modifier: &mut T) {
        let op = graph.nodes[id].op.clone();
        let inputs = graph.nodes[id].inputs.clone();
        let outputs = graph.nodes[id].outputs.clone();
        match &op {
            Operator::Pow => {
                let exponent = inputs[1].unwrap();
                let scalar = {
                    let Some(tensor) = graph.get_initializer(exponent) else {
                        return;
                    };
                    let Some(scalar) = tensor.data.to_scalar_data() else {
                        return;
                    };
                    scalar
                };
                if !matches!(
                    scalar,
                    ScalarData::SInt(_, 2) | ScalarData::UInt(_, 2) | ScalarData::Float(_, 2.0)
                ) {
                    return;
                }

                let new_inputs = vec![inputs[0], inputs[0]];
                let old_output = outputs[0];
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
                let lhs = inputs[0].unwrap();
                let rhs = inputs[1].unwrap();
                let output = outputs[0];

                let reciprocal = modifier.register_new_value(
                    graph,
                    format!("Canonicalize_Reciprocal_{:?}", id),
                    graph.get_resolved_tensor_type(rhs).unwrap().clone(),
                );
                modifier.register_new_node(
                    graph,
                    Node::create_node(
                        vec![Some(rhs)],
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
                        vec![Some(lhs), Some(reciprocal)],
                        vec![new_output],
                        format!("Canonicalize_Mul_{:?}", id),
                        Operator::Mul,
                    ),
                );

                modifier.replace_input_value(graph, output, new_output);
            }

            Operator::MatMul => {
                let lhs = inputs[args::MATMUL_LHS].unwrap();
                let rhs = inputs[args::MATMUL_RHS].unwrap();
                let ldim = graph.get_resolved_tensor_type(lhs).unwrap().dims.ndim();
                let rdim = graph.get_resolved_tensor_type(rhs).unwrap().dims.ndim();
                let old_output = outputs[0];
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
                        inputs: vec![Some(lhs), Some(rhs)],
                        outputs: vec![new_output],
                        name,
                        op,
                        meta: NodeMeta::default(),
                    },
                );
                modifier.replace_input_value(graph, old_output, new_output);
            }

            Operator::Squeeze(_) | Operator::Unsqueeze(_) | Operator::Flatten(_) => {
                let input = inputs[0].unwrap();
                let old_output = outputs[0];
                let node_name = graph.nodes[id].name.clone();
                let output_dims = graph
                    .get_resolved_tensor_type(old_output)
                    .unwrap()
                    .dims
                    .clone();
                let reshaped = ReshapeGenerator::default()
                    .set_input(input)
                    .set_dims(&output_dims[..])
                    .set_allow_contiguous(false)
                    .set_node_name(format!("Canonicalize2Reshape_{node_name}"))
                    .set_value_name(format!("Canonicalize2Reshape_Reshaped_{}", input.index()))
                    .generate(graph, modifier)
                    .unwrap();
                modifier.replace_input_value(graph, old_output, reshaped);
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
                let lhs = node.inputs[args::MATMUL_LHS].unwrap();
                let rhs = node.inputs[args::MATMUL_RHS].unwrap();
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
            let lhs = node.inputs[args::MATMUL_LHS].unwrap();
            let rhs = node.inputs[args::MATMUL_RHS].unwrap();
            let old_output = node.outputs[0];
            let ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
            let name = format!("MatMul2BatchedGemm_{:?}", id);
            let new_output = modifier.register_new_value(graph, name.clone(), ty);
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![Some(lhs), Some(rhs)],
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
