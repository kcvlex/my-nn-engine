use itertools::Itertools;

use crate::graph::operator::*;
use crate::graph::utils::simple_topological_order;
use crate::graph::Graph;
use crate::graph::Node;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct ReorderNodes {}

impl<T: GraphOp> Pass<T> for ReorderNodes {
    fn summary(&self) -> &'static str {
        "Reorder nodes"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let mut ids = simple_topological_order(graph)
            .into_iter()
            .filter(|id| graph.nodes[*id].op.is_elementwise() && graph.nodes[*id].inputs.len() == 1)
            .rev()
            .collect_vec();

        while let Some(id) = ids.pop() {
            let input = graph.nodes[id].inputs[0].unwrap();
            let used = modifier.used_node(input).unwrap();
            if used.len() != 1 {
                continue;
            }
            let Some((def, _)) = modifier.defined_node(input) else {
                continue;
            };
            if graph.nodes[def].outputs.len() != 1 {
                continue;
            }
            if graph.nodes[def].op.operator_type() != OperatorType::Bijective {
                continue;
            }

            let input = graph.nodes[def].inputs[0].unwrap();
            let old_output = graph.nodes[id].outputs[0];

            let new_intermediate_value = modifier.register_new_value(
                graph,
                format!("{}_up", graph.values[old_output].name),
                graph.get_resolved_tensor_type(input).unwrap().clone(),
            );
            let new_node = modifier.register_new_node(
                graph,
                Node::create_node(
                    vec![Some(input)],
                    vec![new_intermediate_value],
                    format!("{}_up", graph.nodes[id].name),
                    graph.nodes[id].op.clone(),
                ),
            );

            let mut inputs = graph.nodes[def].inputs.clone();
            // TODO: Is 0 always correct?
            inputs[0] = Some(new_intermediate_value);
            let new_output = modifier.register_new_value(
                graph,
                format!("{}_down", graph.values[old_output].name),
                graph.get_resolved_tensor_type(old_output).unwrap().clone(),
            );
            modifier.register_new_node(
                graph,
                Node::create_node(
                    inputs,
                    vec![new_output],
                    format!("{}_down", graph.nodes[def].name),
                    graph.nodes[def].op.clone(),
                ),
            );

            modifier.replace_input_value(graph, old_output, new_output);
            ids.push(new_node);
        }
    }
}
