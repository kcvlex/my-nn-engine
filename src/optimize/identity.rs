use crate::onnx::model::Graph;
use crate::onnx::operator::*;
use crate::optimize::optimizer::{GraphModifier, Pass};

#[derive(Default)]
pub struct Ops2Identity {}

impl<T: GraphModifier> Pass<T> for Ops2Identity {
    fn summary(&self) -> &'static str {
        "Convert Reshape/Transpose to Identity"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| match node.op {
                Operator::Reshape | Operator::Transpose(_) => Some(id),
                _ => None,
            })
            .collect::<Vec<_>>();
        for id in ids.iter() {
            modifier.replace_op(graph, *id, Operator::Identity);
        }
    }
}
