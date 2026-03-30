use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::TensorType;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct NHWC2NCHWInsertion {}

impl<T: GraphOp> Pass<T> for NHWC2NCHWInsertion {
    fn summary(&self) -> &'static str {
        "Set Conv output_layout=NHWC and insert NHWC2NCHW after Conv"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let conv_ids: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.op, Operator::Conv(_)))
            .map(|(id, _)| id)
            .collect();

        for conv_id in conv_ids {
            let Operator::Conv(ref conv) = graph.nodes[conv_id].op else {
                unreachable!();
            };
            assert!(conv.output_layout == Layout::NCHW);

            let old_output = graph.nodes[conv_id].outputs[0];
            let nchw_ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
            assert!(nchw_ty.dims.ndim() == 4);

            let Operator::Conv(ref mut conv) = graph.nodes[conv_id].op else {
                unreachable!();
            };
            assert!(conv.output_layout == Layout::NCHW);
            conv.output_layout = Layout::NHWC;

            let nhwc_dims = ResolvedTensorDims::new(&[
                nchw_ty.dims[0],
                nchw_ty.dims[2],
                nchw_ty.dims[3],
                nchw_ty.dims[1],
            ]);
            let nhwc_ty = ResolvedTensorType::new(nchw_ty.elem_type, nhwc_dims);
            graph.values[old_output].ty = Some(TensorType::Resolved(nhwc_ty));

            let nchw_output = modifier.register_new_value(
                graph,
                format!("Conv_NHWC2NCHW_{}", conv_id.index()),
                nchw_ty,
            );
            let nhwc2nchw_id = modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![old_output],
                    outputs: vec![nchw_output],
                    name: format!("Conv_NHWC2NCHW_{}", conv_id.index()),
                    op: Operator::NHWC2NCHW,
                    meta: NodeMeta::default(),
                },
            );

            // Replace all uses of old Conv output except NHWC2NCHW
            modifier.replace_input_value_if_without_typecheck(
                graph,
                old_output,
                nchw_output,
                |id, _| id != nhwc2nchw_id,
            );
        }
    }
}
