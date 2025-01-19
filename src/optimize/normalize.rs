use crate::onnx::model::{Graph, Node, NodeMeta, ValueInfo};
use crate::onnx::operator::*;
use crate::optimize::optimizer::{GraphModifier, Pass};
use crate::optimize::util::ReshapeGenerator;
use crate::tensor::{resolved_dimensions::ResolvedTensorDims, tensor::ResolvedTensorType};

// TODO: Bundle all passes into a single one

#[derive(Default)]
pub struct ContigousOutput {}

// This pass is assumed to be run before shape inference
impl<T: GraphModifier> Pass<T> for ContigousOutput {
    fn summary(&self) -> &'static str {
        "Insert contiguous before all Outputs"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph.outputs.clone();
        for id in ids.iter() {
            let input = graph.nodes[*id].inputs[0];
            let input_ty = graph.values[input].ty.clone();
            let new_value = graph.values.alloc(ValueInfo {
                name: format!("Contiguous_Output_{}", id.index()),
                ty: input_ty,
            });
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![input],
                    outputs: vec![new_value],
                    op: Operator::Contiguous,
                    name: format!("Contiguous_Output_{}", id.index()),
                    meta: NodeMeta::default(),
                },
            );
            modifier.replace_input_value_if(graph, input, new_value, |_, node| {
                matches!(node.op, Operator::Output(_))
            });

            // Forget the dimension information of old output to make shape inference easier
            // TODO: Maybe incorrect if the Input node is directly connected to the Output node
            graph.values[input].ty = None;
        }
    }
}

#[derive(Default)]
pub struct EliminateGlobalAvgPool {}

impl<T: GraphModifier> Pass<T> for EliminateGlobalAvgPool {
    fn summary(&self) -> &'static str {
        "Convert GlobalAveragePool to another operator"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if let Operator::GlobalAveragePool = node.op {
                    Some(id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for id in ids.iter() {
            let input = graph.nodes[*id].inputs[0];
            let old_output = graph.nodes[*id].outputs[0];
            let input_ty = graph.get_resolved_tensor_type(input).unwrap().clone();
            let nbatch = input_ty.dims[0];
            let channel = input_ty.dims[1];
            let row = nbatch * channel;
            let col = input_ty.dims.size() / row;
            let elem_ty = input_ty.elem_type;

            let reshaped = ReshapeGenerator::default()
                .set_input(input)
                .set_dims(vec![row, col].into())
                .set_node_name(format!("GlobalAveragePool_Reshaped_{}", input.index()))
                .set_value_name(format!("GlobalAveragePool_Reshaped_{}", input.index()))
                .generate(graph, modifier)
                .unwrap();

            let pool_output = modifier.register_new_value(
                graph,
                format!("GlobalAveragePool_Output_{}", id.index()),
                ResolvedTensorType::new(elem_ty, ResolvedTensorDims::new(vec![row, 1])),
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![reshaped],
                    outputs: vec![pool_output],
                    op: Operator::ReduceMatrix(ReduceOp::Mean),
                    name: format!("GlobalAveragePool_{}", id.index()),
                    meta: NodeMeta::default(),
                },
            );

            let new_output = ReshapeGenerator::default()
                .set_input(pool_output)
                .set_dims(vec![nbatch, channel].into())
                .set_node_name(format!(
                    "GlobalAveragePool_Reshaped_{}",
                    pool_output.index()
                ))
                .set_value_name(format!(
                    "GlobalAveragePool_Reshaped_{}",
                    pool_output.index()
                ))
                .generate(graph, modifier)
                .unwrap();

            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}
