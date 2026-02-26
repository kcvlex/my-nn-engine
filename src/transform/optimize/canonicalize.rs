use itertools::Itertools;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
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

            _ => (),
        }
    }
}
