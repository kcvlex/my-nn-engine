use crate::onnx::model::{Graph, Node};
use crate::onnx::operator::*;
use crate::optimize::optimizer::{GraphModifier, Pass};
use crate::optimize::util::TransposeGenerator;

#[derive(Default)]
pub struct DecomposeBatchNormalization {}

impl<T: GraphModifier> Pass<T> for DecomposeBatchNormalization {
    fn summary(&self) -> &'static str {
        "Decompose BatchNormalization into some Operators"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if let Operator::BatchNormalization(_) = node.op {
                    Some(id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        // TODO: When the input is 1D
        for node_id in ids.iter() {
            let input = graph.nodes[*node_id].inputs[args::BATCHNORM_DATA];
            let all_inputs = graph.nodes[*node_id].inputs.clone();

            let old_output = graph.nodes[*node_id].outputs[0];
            let input_ty = graph.get_resolved_tensor_type(input).unwrap().clone();
            let mut perms = (0..input_ty.dims.ndim()).collect::<Vec<_>>();
            perms.swap(0, 1);

            let transposed_input = TransposeGenerator::default()
                .set_input(input)
                .set_perms(perms.clone())
                .set_node_name(format!("Transpose_{}", node_id.index()))
                .set_value_name(format!("Transpose_{}", node_id.index()))
                .generate(graph, modifier)
                .unwrap();

            // let channel = input_ty.dims[1];
            // let reduced_ty =
            //     ResolvedTensorType::new(input_ty.elem_type, ResolvedTensorDims::new(vec![channel]));
            // let mean_value = modifier.register_new_value(
            //     graph,
            //     format!("BatchNormalizationMean_{}", node_id.index()),
            //     reduced_ty.clone(),
            // );
            // modifier.register_new_node(
            //     graph,
            //     Node {
            //         inputs: vec![transposed_input],
            //         outputs: vec![mean_value],
            //         op: Operator::ReduceMatrix(ReduceOp::Mean),
            //         name: format!("BatchNormalizationMean_{}", node_id.index()),
            //         mark_as_deleted: false,
            //     },
            // );

            // let variance_value = modifier.register_new_value(
            //     graph,
            //     format!("BatchNormalizationVariance_{}", node_id.index()),
            //     reduced_ty.clone(),
            // );
            // modifier.register_new_node(
            //     graph,
            //     Node {
            //         inputs: vec![transposed_input, mean_value],
            //         outputs: vec![variance_value],
            //         op: Operator::ReduceMatrix(ReduceOp::Variance),
            //         name: format!("BatchNormalizationVariance_{}", node_id.index()),
            //         mark_as_deleted: false,
            //     },
            // );

            let bachnorm_pc_value = modifier.register_new_value(
                graph,
                format!("BatchNormalizationPC_{}", node_id.index()),
                graph
                    .get_resolved_tensor_type(transposed_input)
                    .unwrap()
                    .clone(),
            );
            let mut inputs = all_inputs;
            inputs[args::BATCHNORM_DATA] = transposed_input;
            modifier.register_new_node(
                graph,
                Node {
                    inputs,
                    outputs: vec![bachnorm_pc_value],
                    op: Operator::BatchNormalizationPerChannel(match graph.nodes[*node_id].op {
                        Operator::BatchNormalization(ref bn) => bn.clone(),
                        _ => unreachable!(),
                    }),
                    name: format!("BatchNormalizationPC_{}", node_id.index()),
                    mark_as_deleted: false,
                },
            );

            let new_output = TransposeGenerator::default()
                .set_input(bachnorm_pc_value)
                .set_perms(perms)
                .set_node_name(format!("Transpose_BatchNormalizationPC{}", node_id.index()))
                .set_value_name(format!("Transpose_BatchNormalizationPC{}", node_id.index()))
                .generate(graph, modifier)
                .unwrap();
            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}
