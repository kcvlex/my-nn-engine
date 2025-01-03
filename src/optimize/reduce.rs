use crate::model::{Graph, Node, ValueId};
use crate::operator::*;
use crate::optimize::optimizer::{GraphModifier, Pass};
use crate::tensor::tensor::{ResolvedTensorType, TensorData};

#[derive(Default)]
pub struct Reduce2ReduceMatrix {}

struct ReduceInfo {
    axes: Vec<usize>,
    op: ReduceOp,
}

impl<T: GraphModifier> Pass<T> for Reduce2ReduceMatrix {
    fn summary(&self) -> &'static str {
        "Convert ReduceXXX nodes to Transpose + ReduceMatrix"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let res = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| match node.op {
                Operator::ReduceMax(ref reduce)
                | Operator::ReduceMean(ref reduce)
                | Operator::ReduceSum(ref reduce) => {
                    let input_value = node.inputs[0];
                    let input_ty = graph.get_resolved_tensor_type(input_value).unwrap();
                    let input_rank = input_ty.dims.ndim();
                    let axes = reduce.normalize_axes(input_rank).unwrap();
                    let op = match node.op {
                        Operator::ReduceMax(_) => ReduceOp::Max,
                        Operator::ReduceMean(_) => ReduceOp::Mean,
                        Operator::ReduceSum(_) => ReduceOp::Sum,
                        _ => unreachable!(),
                    };
                    Some((id, ReduceInfo { axes, op }))
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        for (i, (id, info)) in res.iter().enumerate() {
            let input_value = &graph.nodes[*id].inputs[0].clone();
            let input_ty = &graph
                .get_resolved_tensor_type(*input_value)
                .unwrap()
                .clone();
            let rank = input_ty.dims.ndim();
            let mut drop = vec![false; rank];
            for &axis in info.axes.iter() {
                drop[axis] = true;
            }
            let mut perms = Vec::with_capacity(rank);
            for i in 0..rank {
                if !drop[i] {
                    perms.push(i);
                }
            }
            perms.extend(info.axes.iter());

            let transpose_required = perms.iter().enumerate().any(|(i, &v)| i != v);
            let input_v = if transpose_required {
                let transposed_dims = input_ty.dims.transpose(&perms);
                let transposed_output = modifier.register_new_value(
                    graph,
                    format!("Reduce2ReduceMatrix_Transpose_{i}"),
                    ResolvedTensorType::new(input_ty.elem_type, transposed_dims),
                );
                modifier.register_new_node(
                    graph,
                    Node {
                        inputs: vec![*input_value],
                        outputs: vec![transposed_output],
                        name: format!("Reduce2ReduceMatrix_Transpose_{i}"),
                        op: Operator::Transpose(perms),
                        mark_as_deleted: false,
                    },
                );
                transposed_output
            } else {
                *input_value
            };

            let old_output = graph.nodes[*id].outputs[0].clone();
            let output_ty = &graph.get_resolved_tensor_type(old_output).unwrap().clone();
            let row = output_ty.dims.size();
            let col = input_ty.dims.size() / row;

            let reshaped_output = modifier.register_new_value(
                graph,
                format!("Reduce2ReduceMatrix_Reshape_{i}"),
                ResolvedTensorType::new(input_ty.elem_type, vec![row, col].into()),
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![input_v],
                    outputs: vec![reshaped_output],
                    name: format!("Reduce2ReduceMatrix_Reshape_{i}"),
                    op: Operator::Reshape,
                    mark_as_deleted: false,
                },
            );

            let reduce_matrix_output = modifier.register_new_value(
                graph,
                format!("Reduce2ReduceMatrix_Output_{i}"),
                ResolvedTensorType::new(input_ty.elem_type, vec![row].into()),
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![reshaped_output],
                    outputs: vec![reduce_matrix_output],
                    name: format!("Reduce2ReduceMatrix_{i}"),
                    op: Operator::ReduceMatrix(info.op),
                    mark_as_deleted: false,
                },
            );

            let reshaped_output = modifier.register_new_value(
                graph,
                format!("Reduce2ReduceMatrix_ReshapeBack_{i}"),
                output_ty.clone(),
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![reduce_matrix_output],
                    outputs: vec![reshaped_output],
                    name: format!("Reduce2ReduceMatrix_ReshapeBack_{i}"),
                    op: Operator::Reshape,
                    mark_as_deleted: false,
                },
            );

            modifier.replace_input_value(graph, old_output, reshaped_output);
        }
    }
}
