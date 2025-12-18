use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::options::*;
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
            let node = &graph.nodes[*id];
            let input = node.inputs[0];
            let old_output = node.outputs[0];
            let new_output = modifier.register_new_value(
                graph,
                format!("Identity_{}", old_output.index()),
                graph.get_resolved_tensor_type(old_output).unwrap().clone(),
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![input],
                    outputs: vec![new_output],
                    name: format!("Identity_{}", id.index()),
                    op: Operator::Identity,
                    meta: NodeMeta::default(),
                },
            );
            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}

#[derive(Default)]
pub struct ElimCont {}
impl<T: GraphOp> Pass<T> for ElimCont {
    fn summary(&self) -> &'static str {
        "Eliminate unnecessary Contiguous nodes"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter(|(_, node)| match node.op {
                Operator::Contiguous => {
                    let input_shape = graph.get_resolved_tensor_type(node.inputs[0]).unwrap();
                    input_shape.is_contiguous()
                }
                _ => false,
            })
            .map(|(id, _node)| id)
            .collect::<Vec<_>>();

        for id in ids.iter() {
            let node = &graph.nodes[*id];
            let input = node.inputs[0];
            let output = node.outputs[0];
            modifier.replace_input_value(graph, output, input);
        }
    }
}

pub fn create_epilog_passes(opt: &Options) -> SimplePassManager<SimpleGraphOp> {
    let mut manager = SimplePassManager::new("Epilog".to_string());
    manager.add_pass(Box::new(Ops2Identity::default()));
    if matches!(opt.target, Target::CUDA) {
        manager.add_pass(Box::new(ElimCont::default()));
    }
    manager
}
