use itertools::Itertools;

use crate::onnx::model::Graph;
use crate::onnx::operator::args;
use crate::onnx::operator::Operator;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::transform::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct EarlyBroadcast {}

impl<T: GraphOp> Pass<T> for EarlyBroadcast {
    fn summary(&self) -> &'static str {
        "Perform early broadcast primarily for constant tensors"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        self.run_on_batchnorm(graph, modifier);
        self.run_on_gemm(graph, modifier);
    }
}

impl EarlyBroadcast {
    fn run_on_batchnorm<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T) {
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
                    ResolvedTensorDims::new(&res)
                };
                let new_type = ResolvedTensorType::with_stride(elem_type, dims, strides);
                modifier.replace_tensor_type(graph, param, new_type);
            }
        }
    }

    fn run_on_gemm<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter(|(_, node)| match node.op {
                Operator::Gemm(_) => node.inputs.get(args::GEMM_C).is_some(),
                _ => false,
            })
            .map(|(id, _)| id)
            .collect::<Vec<_>>();

        for id in ids {
            let node = &graph.nodes[id];
            let bias = node.inputs[args::GEMM_C];
            if !modifier
                .used_node(bias)
                .map(|s| s.len() == 1)
                .unwrap_or(false)
            {
                unimplemented!("EarlyBroadcast for Gemm with shared bias is not implemented");
            }
            if !graph.initializer.contains_key(&bias) {
                unimplemented!("EarlyBroadcast for Gemm with non-constant bias is not implemented");
            }
            let target_dims = graph
                .get_resolved_tensor_type(node.outputs[0])
                .unwrap()
                .dims
                .clone();
            let tensor = graph
                .initializer
                .get(&bias)
                .unwrap()
                .broadcast(&target_dims);
            modifier.replace_tensor(graph, bias, tensor);
        }
    }
}
