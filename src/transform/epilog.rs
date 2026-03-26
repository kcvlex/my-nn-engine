mod cleanup;
mod lower_nhwc2nchw;

use crate::onnx::model::Graph;
use crate::onnx::operator::*;
use crate::options::*;
use crate::transform::layout::fold_cont;
use crate::transform::layout::ops2reinterpret;
use crate::transform::modify::GraphOp;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::Pass;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

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
                Operator::Contiguous(_) => {
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
    manager.add_pass(Box::new(lower_nhwc2nchw::LowerNHWC2NCHW::default()));
    manager.add_pass(Box::new(ops2reinterpret::Ops2Reinterpret::default()));
    if matches!(opt.target, Target::CUDA) {
        manager.add_pass(Box::new(ElimCont::default()));
    }
    manager.add_pass(Box::new(fold_cont::FoldContiguous::default()));
    manager.add_pass(Box::new(cleanup::CleanupTensors::default()));
    manager
}
