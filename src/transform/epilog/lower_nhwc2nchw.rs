use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct NHWC2NCHWLowering {}

impl<T: GraphOp> Pass<T> for NHWC2NCHWLowering {
    fn summary(&self) -> &'static str {
        "Lower NHWC2NCHW nodes to Contiguous with Transpose"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids: Vec<_> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.op, Operator::NHWC2NCHW))
            .map(|(id, _)| id)
            .collect();

        for id in ids {
            let input = graph.nodes[id].inputs[0].unwrap();
            let old_output = graph.nodes[id].outputs[0];
            let output_ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
            let cont_output = modifier.register_new_value(
                graph,
                format!("LowerNHWC2NCHW_{}", id.index()),
                output_ty,
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![Some(input)],
                    outputs: vec![cont_output],
                    name: format!("NHWC2NCHWLowering_{}", id.index()),
                    op: Operator::Contiguous(Contiguous {
                        ops: vec![ReinterpretType::Transpose(Transpose {
                            perm: Some(vec![0, 3, 1, 2]),
                        })],
                    }),
                    meta: NodeMeta::default(),
                },
            );
            modifier.replace_input_value(graph, old_output, cont_output);
        }
    }
}
