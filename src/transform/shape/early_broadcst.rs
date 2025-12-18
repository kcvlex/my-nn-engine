use crate::onnx::model::Graph;
use crate::onnx::operator::args;
use crate::onnx::operator::Operator;
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::transform::GraphOp;
use crate::transform::Pass;
use itertools::Itertools;

#[derive(Default)]
pub struct EarlyBroadcast {}

impl<T: GraphOp> Pass<T> for EarlyBroadcast {
    fn summary(&self) -> &'static str {
        "Perform early broadcast primary for constant tensors"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let bn_params_idx = &[
            args::BATCHNORM_SCALE,
            args::BATCHNORM_BIAS,
            args::BATCHNORM_MEAN,
            args::BATCHNORM_VAR,
        ];

        let ids = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.op, Operator::BatchNormalization(_)))
            .map(|(id, _)| id)
            .collect::<Vec<_>>();

        for id in ids {
            let node = &graph.nodes[id];
            assert!(matches!(node.op, Operator::BatchNormalization(_)));
            let output_dims = graph
                .get_resolved_tensor_type(node.outputs[0])
                .unwrap()
                .dims
                .clone();
            let channels = output_dims[1];
            for param in bn_params_idx
                .iter()
                .map(|i| node.inputs[*i])
                .collect_vec()
                .iter()
                .copied()
            {
                let param_type = graph.get_resolved_tensor_type(param).unwrap();
                assert_eq!(&param_type.dims[..], &[channels]);
                let elem_type = param_type.elem_type;
                let dims = output_dims.clone();
                let strides = {
                    let mut res = vec![0; dims.ndim()];
                    res[1] = 1;
                    ResolvedTensorDims::new(res)
                };
                let new_type = ResolvedTensorType::with_stride(elem_type, dims, strides);
                modifier.replace_tensor_type(graph, param, new_type);
            }
        }
    }
}
