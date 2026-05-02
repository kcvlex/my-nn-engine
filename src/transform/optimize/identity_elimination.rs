use crate::graph::operator::*;
use crate::graph::Graph;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct IdentityElimination {}

impl<T: GraphOp> Pass<T> for IdentityElimination {
    fn summary(&self) -> &'static str {
        "Eliminate Identity nodes"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| match node.op {
                Operator::Identity => Some(id),
                _ => None,
            })
            .collect::<Vec<_>>();
        for id in ids.iter() {
            let node = &graph.nodes[*id];
            let input = node.inputs[0].unwrap();
            let output = node.outputs[0];
            if graph.get_resolved_tensor_type(input) != graph.get_resolved_tensor_type(output) {
                continue;
            }
            modifier.replace_input_value(graph, output, input);
        }
    }
}
