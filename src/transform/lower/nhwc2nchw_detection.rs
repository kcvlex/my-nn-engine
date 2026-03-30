use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct NHWC2NCHWDetection {}

impl<T: GraphOp> Pass<T> for NHWC2NCHWDetection {
    fn summary(&self) -> &'static str {
        "Detect Contiguous(Transpose[0,3,1,2]) and convert to NHWC2NCHW"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                let Operator::Contiguous(Contiguous { ops }) = &node.op else {
                    return None;
                };
                if ops.len() != 1 {
                    return None;
                }
                let ReinterpretType::Transpose(Transpose { perm: Some(perm) }) = &ops[0] else {
                    return None;
                };
                if perm == &[0, 3, 1, 2] {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();

        for id in ids {
            let input = graph.nodes[id].inputs[0];
            let old_output = graph.nodes[id].outputs[0];
            let output_ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
            let new_output = modifier.register_new_value(
                graph,
                format!("NHWC2NCHWDetection_{}", id.index()),
                output_ty,
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![input],
                    outputs: vec![new_output],
                    name: format!("NHWC2NCHWDetection_{}", id.index()),
                    op: Operator::NHWC2NCHW,
                    meta: NodeMeta::default(),
                },
            );
            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}
