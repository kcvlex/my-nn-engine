use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::transform::modify::GraphOp;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::utils::*;
use crate::transform::Pass;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

#[derive(Default)]
pub struct EliminateGlobalAvgPool {}

impl<T: GraphOp> Pass<T> for EliminateGlobalAvgPool {
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

            let mut new_dims = vec![
                1;
                graph
                    .get_resolved_tensor_type(old_output)
                    .unwrap()
                    .dims
                    .ndim()
            ];
            new_dims[0] = nbatch;
            new_dims[1] = channel;
            let new_output = ReshapeGenerator::default()
                .set_input(pool_output)
                .set_dims(new_dims.into())
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

#[derive(Default)]
pub struct Reduce2ReduceMatrix {}

struct ReduceInfo {
    axes: Vec<usize>,
    op: ReduceOp,
}

impl<T: GraphOp> Pass<T> for Reduce2ReduceMatrix {
    fn summary(&self) -> &'static str {
        "Convert ReduceXXX nodes to Transpose + ReduceMatrix"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let res = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| match node.op {
                Operator::ReduceMax(ref reduce) |
                Operator::ReduceMean(ref reduce) |
                Operator::ReduceSum(ref reduce) => {
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
            let perms = (0..rank)
                .filter(|&i| !drop[i])
                .chain(info.axes.iter().copied())
                .collect();

            let input_v = TransposeGenerator::default()
                .set_input(*input_value)
                .set_perm(perms)
                .set_node_name(format!("Reduce2ReduceMatrix_Transpose_{i}"))
                .set_value_name(format!("Reduce2ReduceMatrix_Transpose_{i}"))
                .generate(graph, modifier)
                .unwrap();

            let old_output = graph.nodes[*id].outputs[0];
            let output_ty = &graph.get_resolved_tensor_type(old_output).unwrap().clone();
            let row = output_ty.dims.size();
            let col = input_ty.dims.size() / row;

            let reshaped_output = ReshapeGenerator::default()
                .set_input(input_v)
                .set_dims(vec![row, col].into())
                .set_node_name(format!("Reduce2ReduceMatrix_Reshape_{i}"))
                .set_value_name(format!("Reduce2ReduceMatrix_Reshape_{i}"))
                .generate(graph, modifier)
                .unwrap();

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
                    meta: NodeMeta::default(),
                },
            );

            let reshaped_output = ReshapeGenerator::default()
                .set_input(reduce_matrix_output)
                .set_dims(output_ty.dims.clone())
                .set_node_name(format!("Reduce2ReduceMatrix_ReshapeBack_{i}"))
                .set_value_name(format!("Reduce2ReduceMatrix_ReshapeBack_{i}"))
                .generate(graph, modifier)
                .unwrap();
            modifier.replace_input_value(graph, old_output, reshaped_output);
        }
    }
}

#[derive(Default)]
pub struct MatMul2Gemm {}

impl<T: GraphOp> Pass<T> for MatMul2Gemm {
    fn summary(&self) -> &'static str {
        "Convert 2-D MatMul to Gemm"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let mut res = Vec::new();
        for (id, node) in graph.nodes.iter() {
            let (lhs, rhs) = if matches!(node.op, Operator::MatMul) {
                (node.inputs[args::MATMUL_LHS], node.inputs[args::MATMUL_RHS])
            } else {
                continue;
            };

            let ldim = graph.get_resolved_tensor_type(lhs).unwrap().dims.ndim();
            let rdim = graph.get_resolved_tensor_type(rhs).unwrap().dims.ndim();
            if ldim == 2 && rdim == 2 {
                res.push(id)
            }
        }
        for (index, id) in res.into_iter().enumerate() {
            let lhs = graph.nodes[id].inputs[args::MATMUL_LHS];
            let rhs = graph.nodes[id].inputs[args::MATMUL_RHS];
            let old_output = graph.nodes[id].outputs[0];
            let ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
            let new_output =
                modifier.register_new_value(graph, format!("MatMul2Gemm_Output_{index}"), ty);
            let new_node = Node {
                inputs: vec![lhs, rhs],
                outputs: vec![new_output],
                name: format!("MatMul2Gemm_{index}"),
                op: Operator::Gemm(Gemm::default()),
                meta: NodeMeta::default(),
            };
            modifier.register_new_node(graph, new_node);
            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}

#[derive(Default)]
pub struct Squeeze2Reshape {}

impl<T: GraphOp> Pass<T> for Squeeze2Reshape {
    fn summary(&self) -> &'static str {
        "Convert Squeeze/Unsqueeze to Reshape"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if matches!(node.op, Operator::Squeeze(_) | Operator::Unsqueeze(_)) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for id in ids.iter() {
            let input = graph.nodes[*id].inputs[0];
            let old_output = graph.nodes[*id].outputs[0];
            let output_dims = graph
                .get_resolved_tensor_type(old_output)
                .unwrap()
                .dims
                .clone();
            let reshaped = ReshapeGenerator::default()
                .set_input(input)
                .set_dims(output_dims)
                .set_node_name(format!("Squeeze2Reshape_Reshaped_{}", input.index()))
                .set_value_name(format!("Squeeze2Reshape_Reshaped_{}", input.index()))
                .generate(graph, modifier)
                .unwrap();
            modifier.replace_input_value(graph, old_output, reshaped);
        }
    }
}

pub fn create_lower_passes() -> SimplePassManager<SimpleGraphOp> {
    let mut passes = SimplePassManager::new("Lowering".to_string());
    passes.add_pass(Box::new(Reduce2ReduceMatrix::default()));
    passes.add_pass(Box::new(EliminateGlobalAvgPool::default()));
    passes.add_pass(Box::new(MatMul2Gemm::default()));
    passes.add_pass(Box::new(Squeeze2Reshape::default()));
    passes
}
