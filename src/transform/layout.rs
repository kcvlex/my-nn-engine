mod detect_nhwc2nchw;
pub mod fold_cont;
mod fold_nhwc2nchw;
pub mod insert_cont;
pub mod ops2reinterpret;
mod sink_nhwc2nchw;
pub mod strides;

use crate::onnx::model::Graph;
use crate::onnx::operator::Operator;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::shape::verify;
use crate::transform::Options;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn should_enable_nhwc(graph: &Graph, opt: &Options) -> bool {
    if let Some(v) = opt.enable_nhwc_optimization {
        return v;
    }
    let has_conv = graph
        .nodes
        .iter()
        .any(|(_, node)| matches!(node.op, Operator::Conv(_)));
    let all_inputs_4d = graph.inputs.iter().all(|&id| {
        let Operator::Input(value_id) = graph.nodes[id].op else {
            return false;
        };
        graph
            .get_resolved_tensor_type(value_id)
            .map(|ty| ty.dims.ndim() == 4)
            .unwrap_or(false)
    });
    has_conv && all_inputs_4d
}

pub fn create_layout_passes(opt: &Options, enable_nhwc: bool) -> SimplePassManager<SimpleGraphOp> {
    let mut manager = SimplePassManager::new("Layout".to_string());
    manager.add_pass(Box::new(insert_cont::InsertContiguous::default()));
    manager.add_pass(Box::new(strides::AssignStrides { target: opt.target }));
    if opt.verify_after_strides {
        manager.add_pass(Box::new(verify::VerifyShape {
            target: opt.target,
            check_strides: true,
        }));
    }
    manager.add_pass(Box::new(ops2reinterpret::Ops2Reinterpret::default()));
    manager.add_pass(Box::new(fold_cont::FoldContiguous::backward_only()));
    if enable_nhwc {
        manager.add_pass(Box::new(detect_nhwc2nchw::DetectNHWC2NCHW::default()));
        manager.add_pass(Box::new(sink_nhwc2nchw::SinkNHWC2NCHW::default()));
        manager.add_pass(Box::new(fold_nhwc2nchw::FoldNHWC2NCHW::default()));
    }
    manager
}
