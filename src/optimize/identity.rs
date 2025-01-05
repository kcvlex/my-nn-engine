use crate::model::Graph;
use crate::operator::*;
use crate::optimize::optimizer::{GraphModifier, Pass};

#[derive(Default)]
pub struct Reshape2Identity {}

impl<T: GraphModifier> Pass<T> for Reshape2Identity {
    fn summary(&self) -> &'static str {
        "Convert Reshape to Identity"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| match node.op {
                Operator::Reshape => Some(id),
                _ => None,
            })
            .collect::<Vec<_>>();
        for id in ids.iter() {
            modifier.replace_op(graph, *id, Operator::Identity);
        }
    }
}
