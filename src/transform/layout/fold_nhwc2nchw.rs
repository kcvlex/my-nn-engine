use crate::onnx::model::Graph;
use crate::onnx::model::NodeId;
use crate::onnx::operator::*;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct FoldNHWC2NCHW {}

impl<T: GraphOp> Pass<T> for FoldNHWC2NCHW {
    fn summary(&self) -> &'static str {
        "Fold NHWC2NCHW into Conv/Im2Col by setting layout to NHWC"
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
                match &graph.nodes[user_id].op {
                    Operator::Conv(_) => Some((nchw_id, user_id)),
                    Operator::Im2Col(Im2Col {
                        channel: Channel::Meld(_),
                        ..
                    }) => Some((nchw_id, user_id)),
                    _ => None,
                }
            })
            .collect();

        for (nchw_id, target_id) in targets {
            let nchw_input = graph.nodes[nchw_id].inputs[0];
            let nchw_output = graph.nodes[nchw_id].outputs[0];

            match &mut graph.nodes[target_id].op {
                Operator::Conv(ref mut conv) => conv.layout = Layout::NHWC,
                Operator::Im2Col(ref mut im2col) => im2col.layout = Layout::NHWC,
                _ => unreachable!(),
            }

            modifier.replace_input_value_if_without_typecheck(
                graph,
                nchw_output,
                nchw_input,
                |id, _| id == target_id,
            );
        }
    }
}
