use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::operator::*;
use crate::onnx::utils::simple_topological_order;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

fn is_reinterpret_or_contiguous(node: &Node) -> bool {
    matches!(node.op, Operator::Reinterpret(_) | Operator::Contiguous(_))
}

fn extract_ops(node: &Node) -> &[ReinterpretType] {
    match &node.op {
        Operator::Reinterpret(re) => &re.ops,
        Operator::Contiguous(cont) => &cont.ops,
        _ => &[],
    }
}

#[derive(Default)]
pub struct FoldContiguous {}

impl<T: GraphOp> Pass<T> for FoldContiguous {
    fn summary(&self) -> &'static str {
        "Fold Contiguous/Reinterpret chains"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let topo = simple_topological_order(graph);

        let chain_starts: Vec<NodeId> = topo
            .into_iter()
            .filter(|id| {
                let node = &graph.nodes[*id];
                if !is_reinterpret_or_contiguous(node) {
                    return false;
                }
                let input = node.inputs[0];
                let Some((def_id, _)) = modifier.defined_node(input) else {
                    return true;
                };
                if !is_reinterpret_or_contiguous(&graph.nodes[def_id]) {
                    return true;
                }
                modifier.used_node(input).is_none_or(|u| u.len() != 1)
            })
            .collect();

        for start_id in chain_starts {
            let start_output = graph.nodes[start_id].outputs[0];
            let (final_output, forward_chain) =
                modifier.walk_chain_forward(graph, start_output, is_reinterpret_or_contiguous);

            let all_nodes: Vec<NodeId> = std::iter::once(start_id).chain(forward_chain).collect();

            let has_contiguous = all_nodes
                .iter()
                .any(|&id| matches!(graph.nodes[id].op, Operator::Contiguous(_)));
            if !has_contiguous {
                continue;
            }

            if all_nodes.len() == 1 {
                continue;
            }

            let all_ops: Vec<ReinterpretType> = all_nodes
                .iter()
                .flat_map(|&id| extract_ops(&graph.nodes[id]).iter().cloned())
                .collect();

            let true_input = graph.nodes[start_id].inputs[0];
            let output_ty = graph
                .get_resolved_tensor_type(final_output)
                .unwrap()
                .clone();
            let new_output = modifier.register_new_value(
                graph,
                format!("FoldedContiguous_{:?}", start_id),
                output_ty,
            );
            modifier.register_new_node(
                graph,
                Node::create_node(
                    vec![true_input],
                    vec![new_output],
                    format!("FoldedContiguous_{:?}", start_id),
                    Operator::Contiguous(Contiguous { ops: all_ops }),
                ),
            );
            modifier.replace_input_value(graph, final_output, new_output);
        }
    }
}
