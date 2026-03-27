use crate::onnx::model::Graph;
use crate::onnx::model::NodeId;
use crate::onnx::operator::*;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct FoldNHWC2NCHW {}

impl<T: GraphOp> Pass<T> for FoldNHWC2NCHW {
    fn summary(&self) -> &'static str {
        "Fold NHWC2NCHW into Conv by setting layout to NHWC"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let targets: Vec<(NodeId, NodeId)> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.op, Operator::NHWC2NCHW))
            .filter_map(|(nchw_id, nchw_node)| {
                let nchw_output = nchw_node.outputs[0];
                let users = modifier.used_node(nchw_output)?;
                let users: Vec<_> = users.iter().copied().collect();
                if users.len() != 1 {
                    return None;
                }
                let (user_id, user_input_idx) = users[0];
                if user_input_idx != 0 {
                    return None;
                }
                if !matches!(graph.nodes[user_id].op, Operator::Conv(_)) {
                    return None;
                }
                Some((nchw_id, user_id))
            })
            .collect();

        for (nchw_id, conv_id) in targets {
            let nchw_input = graph.nodes[nchw_id].inputs[0];
            let nchw_output = graph.nodes[nchw_id].outputs[0];

            // Set Conv layout to NHWC
            let Operator::Conv(ref mut conv) = graph.nodes[conv_id].op else {
                unreachable!();
            };
            conv.layout = Layout::NHWC;

            // Wire NHWC2NCHW's input directly to Conv
            modifier.replace_input_value_if_without_typecheck(
                graph,
                nchw_output,
                nchw_input,
                |id, _| id == conv_id,
            );
        }
    }
}
