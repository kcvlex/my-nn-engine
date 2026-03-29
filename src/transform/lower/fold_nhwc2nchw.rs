use crate::onnx::model::Graph;
use crate::onnx::model::NodeId;
use crate::onnx::operator::*;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::TensorType;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct FoldNHWC2NCHW {}

impl<T: GraphOp> Pass<T> for FoldNHWC2NCHW {
    fn summary(&self) -> &'static str {
        "Fold NHWC2NCHW into Conv"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        self.fold_into_conv_input(graph, modifier);
        self.fold_from_conv_output(graph, modifier);
    }
}

impl FoldNHWC2NCHW {
    /// NHWC2NCHW -> Conv: set Conv.input_layout=NHWC
    fn fold_into_conv_input<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T) {
        let targets: Vec<(NodeId, Vec<NodeId>)> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.op, Operator::NHWC2NCHW))
            .filter_map(|(nchw_id, nchw_node)| {
                let nchw_output = nchw_node.outputs[0];
                let users = modifier.used_node(nchw_output)?;
                let user_ids: Vec<_> = users
                    .iter()
                    .filter_map(|&(user_id, user_input_idx)| {
                        if user_input_idx != 0 {
                            return None;
                        }
                        match &graph.nodes[user_id].op {
                            Operator::Conv(_) => Some(user_id),
                            _ => None,
                        }
                    })
                    .collect();
                if user_ids.len() != users.len() {
                    return None;
                }
                Some((nchw_id, user_ids))
            })
            .collect();

        for (nchw_id, user_ids) in targets {
            let nchw_input = graph.nodes[nchw_id].inputs[0];
            let nchw_output = graph.nodes[nchw_id].outputs[0];

            for &conv_id in &user_ids {
                let Operator::Conv(ref mut conv) = graph.nodes[conv_id].op else {
                    unreachable!();
                };
                assert!(conv.input_layout == Layout::NCHW);
                conv.input_layout = Layout::NHWC;
            }

            modifier.replace_input_value_if_without_typecheck(
                graph,
                nchw_output,
                nchw_input,
                |id, _| user_ids.contains(&id),
            );
        }
    }

    /// Conv(output_layout=NHWC) -> NHWC2NCHW: set Conv.output_layout=NCHW
    fn fold_from_conv_output<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T) {
        let targets: Vec<(NodeId, NodeId)> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.op, Operator::NHWC2NCHW))
            .filter_map(|(nchw_id, nchw_node)| {
                let input = nchw_node.inputs[0];
                let (def_id, _) = modifier.defined_node(input)?;
                let Operator::Conv(ref conv) = graph.nodes[def_id].op else {
                    return None;
                };
                assert!(conv.output_layout == Layout::NHWC);
                Some((nchw_id, def_id))
            })
            .collect();

        for (nchw_id, conv_id) in targets {
            let nchw_input = graph.nodes[nchw_id].inputs[0];
            let nchw_output = graph.nodes[nchw_id].outputs[0];

            let Operator::Conv(ref mut conv) = graph.nodes[conv_id].op else {
                unreachable!();
            };
            conv.output_layout = Layout::NCHW;

            // Update Conv output type back to NCHW contiguous
            let nchw_ty = graph.get_resolved_tensor_type(nchw_output).unwrap().clone();
            graph.values[nchw_input].ty = Some(TensorType::Resolved(nchw_ty));

            modifier.replace_input_value_if_without_typecheck(
                graph,
                nchw_output,
                nchw_input,
                |_, _| true,
            );
        }
    }
}
