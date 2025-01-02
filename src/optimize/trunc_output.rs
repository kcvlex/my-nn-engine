use crate::model::{Graph, Node, ValueId};
use crate::operator::*;
use crate::optimize::opinfo::*;
use crate::optimize::optimizer;

#[derive(Default)]
pub struct TruncOutput {}

impl optimizer::Pass for TruncOutput {
    fn summary(&self) -> &'static str {
        "Elminate pure nodes before Output nodes"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut optimizer::GraphModifier) {
        let res = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| match node.op {
                Operator::Output(_) => Some(id),
                _ => None,
            })
            .collect::<Vec<_>>();

        for node_id in res.into_iter() {
            loop {
                let value_id = match graph.nodes[node_id].op {
                    Operator::Output(v) => v,
                    _ => unreachable!(),
                };
                let defined_node = modifier.defined_node(value_id).unwrap();
                let defined_node = &graph.nodes[defined_node.0];
                if !defined_node.op.is_identity() {
                    break;
                }
                if matches!(defined_node.op, Operator::Input(_)) {
                    break;
                }
                let input_value = defined_node.inputs[0];
                modifier
                    .replace_input_value_if(graph, value_id, input_value, |id, _| id == node_id);
            }
        }
    }
}
