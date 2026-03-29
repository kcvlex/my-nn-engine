pub mod im2col;
pub mod insert_cont;
pub mod strides;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::options::*;
use crate::tensor::data::TensorData;
use crate::tensor::types::DataType;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::Tensor;
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
                .set_dims(&[row, col])
                .set_node_name(format!("GlobalAveragePool_Reshaped_{}", input.index()))
                .set_value_name(format!("GlobalAveragePool_Reshaped_{}", input.index()))
                .generate(graph, modifier)
                .unwrap();

            let pool_output = modifier.register_new_value(
                graph,
                format!("GlobalAveragePool_Output_{}", id.index()),
                ResolvedTensorType::new(elem_ty, ResolvedTensorDims::new(&[row, 1])),
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
                .set_dims(&new_dims)
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
                .set_dims(&[row, col])
                .set_node_name(format!("Reduce2ReduceMatrix_Reshape_{i}"))
                .set_value_name(format!("Reduce2ReduceMatrix_Reshape_{i}"))
                .generate(graph, modifier)
                .unwrap();

            let reduce_matrix_output = modifier.register_new_value(
                graph,
                format!("Reduce2ReduceMatrix_Output_{i}"),
                ResolvedTensorType::new(input_ty.elem_type, ResolvedTensorDims::new(&[row])),
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
                .set_dims(&output_ty.dims[..])
                .set_node_name(format!("Reduce2ReduceMatrix_ReshapeBack_{i}"))
                .set_value_name(format!("Reduce2ReduceMatrix_ReshapeBack_{i}"))
                .generate(graph, modifier)
                .unwrap();
            modifier.replace_input_value(graph, old_output, reshaped_output);
        }
    }
}

pub fn create_lower_passes(opt: &Options) -> SimplePassManager<SimpleGraphOp> {
    let mut passes = SimplePassManager::new("Lowering".to_string());
    passes.add_pass(Box::new(Reduce2ReduceMatrix::default()));
    passes.add_pass(Box::new(EliminateGlobalAvgPool::default()));
    if matches!(opt.target, Target::CPU) {
        passes.add_pass(Box::new(DecomposeAttention::default()));
    }

    // Layout
    passes.add_pass(Box::new(strides::AssignStrides { target: opt.target }));
    passes.add_pass(Box::new(insert_cont::InsertContiguous::default()));
    if opt.verify_after_strides {
        passes.add_pass(Box::new(crate::transform::shape::verify::VerifyShape {
            target: opt.target,
            check_strides: true,
        }));
    }

    // Decompose Conv/MaxPool after layout
    if matches!(opt.target, Target::CPU) {
        passes.add_pass(Box::new(im2col::DecomposeConv::default()));
        passes.add_pass(Box::new(im2col::DecomposeMaxPool::default()));
    }
    passes
}

#[derive(Default)]
pub struct DecomposeAttention {}

impl<T: GraphOp> Pass<T> for DecomposeAttention {
    fn summary(&self) -> &'static str {
        "Decompose Attention into BatchedGemm + Softmax (CPU)"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let attn_ids: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(&node.op, Operator::Attention(_)))
            .map(|(id, _)| id)
            .collect();

        for attn_id in attn_ids {
            let node = &graph.nodes[attn_id];
            let Operator::Attention(attn) = &node.op else {
                unreachable!()
            };
            let attn = *attn;
            let q = node.inputs[args::ATTENTION_Q];
            let k = node.inputs[args::ATTENTION_K];
            let v = node.inputs[args::ATTENTION_V];
            let mask = node.inputs.get(args::ATTENTION_MASK).copied();
            let old_output = node.outputs[0];

            let q_ty = graph.get_resolved_tensor_type(q).unwrap().clone();
            let k_ty = graph.get_resolved_tensor_type(k).unwrap().clone();
            let out_ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();

            // Q: [b, h, seq_q, d], K: [b, h, seq_k, d]
            // BatchedGemm(Q, K, trans_b=true, alpha=scale) → [b, h, seq_q, seq_k]
            let ndim = q_ty.dims.ndim();
            let seq_q = q_ty.dims[ndim - 2];
            let seq_k = k_ty.dims[ndim - 2];
            let mut qk_dims: Vec<usize> = q_ty.dims.iter().copied().collect();
            qk_dims[ndim - 2] = seq_q;
            qk_dims[ndim - 1] = seq_k;
            let qk_ty = ResolvedTensorType::new(q_ty.elem_type, ResolvedTensorDims::new(&qk_dims));

            let qk = modifier.register_new_value(
                graph,
                format!("DecompAttn_QK_{:?}", attn_id),
                qk_ty.clone(),
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![q, k],
                    outputs: vec![qk],
                    name: format!("DecompAttn_QK_{:?}", attn_id),
                    op: Operator::BatchedGemm(BatchedGemm {
                        alpha: attn.scale as f64,
                        beta: 0.0,
                        trans_a: false,
                        trans_b: true,
                    }),
                    meta: NodeMeta::default(),
                },
            );

            // Apply mask
            let mut qk_masked = qk;
            let mask = if let Some(mask) = mask {
                Some(mask)
            } else if attn.is_causal {
                let mut data = vec![0.0f64; seq_q * seq_k];
                for r in 0..seq_q {
                    for c in 0..seq_k {
                        if c > r {
                            data[r * seq_k + c] = f64::NEG_INFINITY;
                        }
                    }
                }
                let DataType::Float(float_ty) = q_ty.elem_type else {
                    panic!("Attention requires float type");
                };
                let mask_tensor = Tensor::new(
                    ResolvedTensorDims::new(&[seq_q, seq_k]),
                    TensorData::Float(float_ty, data),
                )
                .unwrap();
                Some(modifier.register_new_tensor(
                    graph,
                    mask_tensor,
                    format!("DecompAttn_CausalMask_{:?}", attn_id),
                ))
            } else {
                None
            };
            if let Some(mask) = mask {
                let qk_with_mask = modifier.register_new_value(
                    graph,
                    format!("DecompAttn_QKMask_{:?}", attn_id),
                    qk_ty.clone(),
                );
                modifier.register_new_node(
                    graph,
                    Node {
                        inputs: vec![qk_masked, mask],
                        outputs: vec![qk_with_mask],
                        name: format!("DecompAttn_QKMask_{:?}", attn_id),
                        op: Operator::Add,
                        meta: NodeMeta::default(),
                    },
                );
                qk_masked = qk_with_mask;
            }

            // Softmax(QK, axis=-1)
            let qk_softmax = modifier.register_new_value(
                graph,
                format!("DecompAttn_Softmax_{:?}", attn_id),
                qk_ty,
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![qk_masked],
                    outputs: vec![qk_softmax],
                    name: format!("DecompAttn_Softmax_{:?}", attn_id),
                    op: Operator::Softmax(Softmax {
                        axis: TensorIndex::new(-1),
                    }),
                    meta: NodeMeta::default(),
                },
            );

            // Output = BatchedGemm(Softmax(QK), V)
            let new_output =
                modifier.register_new_value(graph, format!("DecompAttn_Out_{:?}", attn_id), out_ty);
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![qk_softmax, v],
                    outputs: vec![new_output],
                    name: format!("DecompAttn_Out_{:?}", attn_id),
                    op: Operator::BatchedGemm(BatchedGemm {
                        alpha: 1.0,
                        beta: 0.0,
                        trans_a: false,
                        trans_b: false,
                    }),
                    meta: NodeMeta::default(),
                },
            );

            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}
