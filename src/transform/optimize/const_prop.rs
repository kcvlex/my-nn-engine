use crate::onnx::model::Graph;
use crate::onnx::model::NodeId;
use crate::onnx::operator::args;
use crate::onnx::operator::*;
use crate::transform::modify::GraphOp;

// Propagate constant inputs "into" the given node.
pub fn prop_constant<T: GraphOp>(graph: &mut Graph, node_id: NodeId, modifier: &mut T) {
    match &mut graph.nodes[node_id].op {
        Operator::Resize(resize) => {
            if resize.scale.is_some() {
                return;
            }
            let scales = graph.nodes[node_id]
                .inputs
                .get(args::RESIZE_SCALES)
                .and_then(|id| graph.initializer.get(id))
                .and_then(|tensor| tensor.to_1d_floats());

            let sizes = graph.nodes[node_id]
                .inputs
                .get(args::RESIZE_SIZES)
                .and_then(|id| graph.initializer.get(id))
                .and_then(|tensor| tensor.to_1d_sints());

            let scale = match (scales, sizes) {
                (Some(scales), None) => ResizeScale::Scales(scales),
                (Some(scales), Some(sizes)) if scales.is_empty() => ResizeScale::Sizes(sizes),
                (None, Some(sizes)) => ResizeScale::Sizes(sizes),
                (Some(_), Some(_)) => unreachable!(),
                (None, None) => unimplemented!(),
            };

            let Operator::Resize(resize) = &mut graph.nodes[node_id].op else {
                unreachable!();
            };
            resize.scale = Some(scale);

            // TODO: Don't drop ROI.
            let mut args = [args::RESIZE_ROI, args::RESIZE_SCALES, args::RESIZE_SIZES];
            args.sort();
            for arg in args.iter().rev() {
                if graph.nodes[node_id].inputs.get(*arg).is_some() {
                    modifier.drop_node_input(graph, node_id, *arg);
                }
            }
        }

        _ => (),
    }
}
