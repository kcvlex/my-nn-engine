use crate::onnx::model::Graph;
use crate::onnx::operator::*;
use crate::transform::modify::GraphOp;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::SimplePassManager;
use crate::transform::{Pass, PassManager};

#[derive(Default)]
pub struct Ops2Identity {}

impl<T: GraphOp> Pass<T> for Ops2Identity {
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

pub fn create_epilog_passes() -> SimplePassManager<SimpleGraphOp> {
    let mut manager = SimplePassManager::new("Epilog".to_string());
    manager.add_pass(Box::new(Ops2Identity::default()));
    manager
}
