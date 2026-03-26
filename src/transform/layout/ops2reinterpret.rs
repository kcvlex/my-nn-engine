use itertools::Itertools;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::onnx::utils::simple_topological_order;
use crate::transform::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct Ops2Reinterpret {}

fn bundle_reshape_and_transpose<T: GraphOp>(
    graph: &Graph,
    node_id: NodeId,
    modifier: &T,
) -> (Reinterpret, ValueId) {
    let input_value = graph.nodes[node_id].inputs[0];
    let (source, chain) = modifier.walk_chain_backward(graph, input_value, |node| {
        matches!(node.op, Operator::Reshape | Operator::Transpose(_))
    });

    let all_nodes: Vec<NodeId> = chain
        .into_iter()
        .rev()
        .chain(std::iter::once(node_id))
        .collect();
    let ops = all_nodes
        .iter()
        .map(|&id| match &graph.nodes[id].op {
            Operator::Reshape => {
                let input_shape = graph
                    .get_resolved_tensor_type(graph.nodes[id].inputs[0])
                    .unwrap();
                let output_shape = graph
                    .get_resolved_tensor_type(graph.nodes[id].outputs[0])
                    .unwrap();
                ReinterpretType::Reshape {
                    before: input_shape.dims.iter().copied().collect(),
                    after: output_shape.dims.iter().copied().collect(),
                }
            }
            Operator::Transpose(perm) => ReinterpretType::Transpose(perm.clone()),
            _ => unreachable!(),
        })
        .collect();

    (Reinterpret { ops }, source)
}

impl<T: GraphOp> Pass<T> for Ops2Reinterpret {
    fn summary(&self) -> &'static str {
        "Convert Reshape/Transpose to Reinterpret"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = simple_topological_order(graph)
            .into_iter()
            .filter(|id| {
                matches!(
                    graph.nodes[*id].op,
                    Operator::Reshape | Operator::Transpose(_)
                )
            })
            .rev()
            .collect_vec();

        for id in ids.iter() {
            let old_output = graph.nodes[*id].outputs[0];
            let Some(to_bundle) = modifier.used_node(old_output) else {
                continue;
            };
            let to_bundle = to_bundle.iter().any(|(user_id, _)| {
                !matches!(
                    graph.nodes[*user_id].op,
                    Operator::Reshape | Operator::Transpose(_)
                )
            });
            if !to_bundle {
                continue;
            }

            let (re, input) = bundle_reshape_and_transpose(graph, *id, modifier);
            let new_output = modifier.register_new_value(
                graph,
                format!("Reinterpret_{:?}", old_output),
                graph.get_resolved_tensor_type(old_output).unwrap().clone(),
            );
            modifier.register_new_node(
                graph,
                Node::create_node(
                    vec![input],
                    vec![new_output],
                    format!("Reinterpret_{:?}", id),
                    Operator::Reinterpret(re),
                ),
            );
            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}
