use std::collections::HashSet;

use crate::graph::operator::Operator;
use crate::graph::Graph;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct CleanupTensors {}

impl<T: GraphOp> Pass<T> for CleanupTensors {
    fn summary(&self) -> &'static str {
        "Cleanup unused tensors (initializers) from the graph"
    }

    fn run(&self, graph: &mut Graph, _modifier: &mut T) {
        let used_values: HashSet<_> = graph
            .nodes
            .iter()
            .filter(|(_, node)| !matches!(node.op, Operator::Input(_)))
            .flat_map(|(_, node)| node.inputs.iter())
            .filter_map(|v| *v)
            .collect();

        graph.remove_initializer(|v| !used_values.contains(v));
    }
}
