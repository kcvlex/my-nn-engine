use itertools::Itertools;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::tensor::data::ScalarData;
use crate::tensor::data::TensorData;
use crate::tensor::types::FloatType;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::Tensor;
use crate::transform::modify::GraphOp;
use crate::transform::utils::*;
use crate::transform::Pass;

#[derive(Default)]
pub struct Canonicalization {}

impl<T: GraphOp> Pass<T> for Canonicalization {
    fn summary(&self) -> &'static str {
        "Canonicalize some patterns"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let mut ids = graph.nodes.iter().map(|(id, _)| id).collect_vec();
        ids.reverse();

        for id in ids {
            self.rewrite(id, graph, modifier);
        }
    }
}

impl Canonicalization {
    fn rewrite<T: GraphOp>(&self, id: NodeId, graph: &mut Graph, modifier: &mut T) {
        let op = graph.nodes[id].op.clone();
        let inputs = graph.nodes[id].inputs.clone();
        let outputs = graph.nodes[id].outputs.clone();
        match &op {
            Operator::Pow => {
                let exponent = inputs[1].unwrap();
                let scalar = {
                    let Some(tensor) = graph.get_initializer(exponent) else {
                        return;
                    };
                    let Some(scalar) = tensor.data.to_scalar_data() else {
                        return;
                    };
                    scalar
                };
                if !matches!(
                    scalar,
                    ScalarData::SInt(_, 2) | ScalarData::UInt(_, 2) | ScalarData::Float(_, 2.0)
                ) {
                    return;
                }

                let new_inputs = vec![inputs[0], inputs[0]];
                let old_output = outputs[0];
                let new_output = modifier.register_new_value(
                    graph,
                    format!("Canonicalize_Square_{:?}", id),
                    graph.get_resolved_tensor_type(old_output).unwrap().clone(),
                );
                modifier.register_new_node(
                    graph,
                    Node::create_node(
                        new_inputs,
                        vec![new_output],
                        format!("Canonicalize_Square_{:?}", id),
                        Operator::Mul,
                    ),
                );
                modifier.replace_input_value(graph, old_output, new_output);
            }

            Operator::Div => {
                let lhs = inputs[0].unwrap();
                let rhs = inputs[1].unwrap();
                let output = outputs[0];

                let reciprocal = modifier.register_new_value(
                    graph,
                    format!("Canonicalize_Reciprocal_{:?}", id),
                    graph.get_resolved_tensor_type(rhs).unwrap().clone(),
                );
                modifier.register_new_node(
                    graph,
                    Node::create_node(
                        vec![Some(rhs)],
                        vec![reciprocal],
                        format!("Canonicalize_Reciprocal_{:?}", id),
                        Operator::Reciprocal,
                    ),
                );

                let new_output = modifier.register_new_value(
                    graph,
                    format!("Canonicalize_Mul_{:?}", id),
                    graph.get_resolved_tensor_type(output).unwrap().clone(),
                );
                modifier.register_new_node(
                    graph,
                    Node::create_node(
                        vec![Some(lhs), Some(reciprocal)],
                        vec![new_output],
                        format!("Canonicalize_Mul_{:?}", id),
                        Operator::Mul,
                    ),
                );

                modifier.replace_input_value(graph, output, new_output);
            }

            Operator::MatMul => {
                let lhs = inputs[args::MATMUL_LHS].unwrap();
                let rhs = inputs[args::MATMUL_RHS].unwrap();
                let l_ty = graph.get_resolved_tensor_type(lhs).unwrap().clone();
                let r_ty = graph.get_resolved_tensor_type(rhs).unwrap().clone();
                let ldim = l_ty.dims.ndim();
                let rdim = r_ty.dims.ndim();
                let old_output = outputs[0];
                let ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
                if ldim == 2 && rdim == 2 {
                    // 2D MatMul -> Gemm. 3D+ MatMul -> BatchedGemm is done later
                    // (after AttentionFusion).
                    let name = format!("MatMul2Gemm_{:?}", id);
                    let op = Operator::Gemm(Gemm::default());
                    let new_output = modifier.register_new_value(graph, name.clone(), ty);
                    modifier.register_new_node(
                        graph,
                        Node {
                            inputs: vec![Some(lhs), Some(rhs)],
                            outputs: vec![new_output],
                            name,
                            op,
                            meta: NodeMeta::default(),
                        },
                    );
                    modifier.replace_input_value(graph, old_output, new_output);
                    return;
                }

                // lhs: [1, 1, ..., M, K]
                // rhs: [1, 1, ..., K, N]
                //
                // Then,
                //
                // MatMul(lhs, rhs) -> Unsqueeze(Gemm(Squeeze(lhs), Squeeze(rhs))))
                if l_ty.dims.iter().take(ldim - 2).all(|&d| d == 1) &&
                    r_ty.dims.iter().take(rdim - 2).all(|&d| d == 1)
                {
                    let m = l_ty.dims[ldim - 2];
                    let k = l_ty.dims[ldim - 1];
                    let n = r_ty.dims[rdim - 1];

                    let lhs_2d = ReshapeGenerator::default()
                        .set_input(lhs)
                        .set_dims(&[m, k])
                        .set_allow_contiguous(false)
                        .set_node_name(format!("MatMul2Gemm_LhsSqueeze_{:?}", id))
                        .set_value_name(format!("MatMul2Gemm_LhsSqueeze_{:?}", id))
                        .generate(graph, modifier)
                        .unwrap();
                    let rhs_2d = ReshapeGenerator::default()
                        .set_input(rhs)
                        .set_dims(&[k, n])
                        .set_allow_contiguous(false)
                        .set_node_name(format!("MatMul2Gemm_RhsSqueeze_{:?}", id))
                        .set_value_name(format!("MatMul2Gemm_RhsSqueeze_{:?}", id))
                        .generate(graph, modifier)
                        .unwrap();

                    let gemm_out_ty =
                        ResolvedTensorType::new(ty.elem_type, ResolvedTensorDims::new(&[m, n]));
                    let gemm_out = modifier.register_new_value(
                        graph,
                        format!("MatMul2Gemm_GemmOut_{:?}", id),
                        gemm_out_ty,
                    );
                    modifier.register_new_node(
                        graph,
                        Node {
                            inputs: vec![Some(lhs_2d), Some(rhs_2d)],
                            outputs: vec![gemm_out],
                            name: format!("MatMul2Gemm_{:?}", id),
                            op: Operator::Gemm(Gemm::default()),
                            meta: NodeMeta::default(),
                        },
                    );

                    let target_dims: Vec<usize> = ty.dims.iter().copied().collect();
                    let final_out = ReshapeGenerator::default()
                        .set_input(gemm_out)
                        .set_dims(&target_dims)
                        .set_allow_contiguous(false)
                        .set_node_name(format!("MatMul2Gemm_OutUnsqueeze_{:?}", id))
                        .set_value_name(format!("MatMul2Gemm_OutUnsqueeze_{:?}", id))
                        .generate(graph, modifier)
                        .unwrap();

                    modifier.replace_input_value(graph, old_output, final_out);
                    return;
                }
                // Pre-scaled Q/K: MatMul(Mul(Q, s_q), Mul(K, s_k)) is
                // equivalent to Mul(MatMul(Q, K), s_q * s_k). Enables uniform
                // attention fusion when the scale is baked into Q/K upstream.
                let (lhs_new, lhs_scale) = strip_scalar_mul(graph, modifier, lhs);
                let (rhs_new, rhs_scale) = strip_scalar_mul(graph, modifier, rhs);
                let (lhs_scale, rhs_scale) = match (lhs_scale, rhs_scale) {
                    (None, None) => return,
                    (a, b) => (a.unwrap_or(1.0), b.unwrap_or(1.0)),
                };
                let combined = lhs_scale * rhs_scale;

                let scale_tensor = Tensor::new(
                    ResolvedTensorDims::new(&[]),
                    ScalarData::Float(FloatType::F32, combined).to_tensor_data(1),
                )
                .unwrap();
                let scale_val = modifier.register_new_tensor(
                    graph,
                    scale_tensor,
                    format!("MatMulScalarPullOut_Scale_{:?}", id),
                );

                let matmul_out = modifier.register_new_value(
                    graph,
                    format!("MatMulScalarPullOut_MatMul_{:?}", id),
                    ty.clone(),
                );
                modifier.register_new_node(
                    graph,
                    Node::create_node(
                        vec![Some(lhs_new), Some(rhs_new)],
                        vec![matmul_out],
                        format!("MatMulScalarPullOut_MatMul_{:?}", id),
                        Operator::MatMul,
                    ),
                );
                let scaled_out = modifier.register_new_value(
                    graph,
                    format!("MatMulScalarPullOut_Scaled_{:?}", id),
                    ty,
                );
                modifier.register_new_node(
                    graph,
                    Node::create_node(
                        vec![Some(matmul_out), Some(scale_val)],
                        vec![scaled_out],
                        format!("MatMulScalarPullOut_Mul_{:?}", id),
                        Operator::Mul,
                    ),
                );
                modifier.replace_input_value(graph, old_output, scaled_out);
            }

            // GPT-2 attention mask canonicalization:
            //   Mul(x, causal_0_1) -> Sub(_, penalty)
            // is equivalent to
            //   Add(x, additive_const_mask)
            // where additive_const_mask is 0 on the lower triangle and
            // -penalty on the upper triangle. Emits Add + a new initializer;
            // downstream const-folding compacts any remaining paths.
            Operator::Sub => {
                if inputs.len() != 2 {
                    return;
                }
                let sub_lhs = inputs[0].unwrap();
                let sub_rhs = inputs[1].unwrap();

                let Some((mul_node, _)) = modifier.defined_node(sub_lhs) else {
                    return;
                };
                let Operator::Mul = graph.nodes[mul_node].op.clone() else {
                    return;
                };
                if graph.nodes[mul_node].inputs.len() != 2 {
                    return;
                }
                let mul_lhs = graph.nodes[mul_node].inputs[0].unwrap();
                let mul_rhs = graph.nodes[mul_node].inputs[1].unwrap();
                let (qk_scaled, causal_val) = match (
                    graph.get_initializer(mul_lhs).is_some(),
                    graph.get_initializer(mul_rhs).is_some(),
                ) {
                    (false, true) => (mul_lhs, mul_rhs),
                    (true, false) => (mul_rhs, mul_lhs),
                    _ => return,
                };

                let Some(mask_tensor) = build_causal_additive_mask(graph, causal_val, sub_rhs)
                else {
                    return;
                };
                let mask_val = modifier.register_new_tensor(
                    graph,
                    mask_tensor,
                    format!("Canonicalize_CausalMask_{:?}", id),
                );

                let sub_output = outputs[0];
                let output_ty = graph.get_resolved_tensor_type(sub_output).unwrap().clone();
                let new_output = modifier.register_new_value(
                    graph,
                    format!("Canonicalize_MaskedQK_{:?}", id),
                    output_ty,
                );
                modifier.register_new_node(
                    graph,
                    Node::create_node(
                        vec![Some(qk_scaled), Some(mask_val)],
                        vec![new_output],
                        format!("Canonicalize_Add_{:?}", id),
                        Operator::Add,
                    ),
                );
                modifier.replace_input_value(graph, sub_output, new_output);
            }

            Operator::Squeeze(_) | Operator::Unsqueeze(_) | Operator::Flatten(_) => {
                let input = inputs[0].unwrap();
                let old_output = outputs[0];
                let node_name = graph.nodes[id].name.clone();
                let output_dims = graph
                    .get_resolved_tensor_type(old_output)
                    .unwrap()
                    .dims
                    .clone();
                let reshaped = ReshapeGenerator::default()
                    .set_input(input)
                    .set_dims(&output_dims[..])
                    .set_allow_contiguous(false)
                    .set_node_name(format!("Canonicalize2Reshape_{node_name}"))
                    .set_value_name(format!("Canonicalize2Reshape_Reshaped_{}", input.index()))
                    .generate(graph, modifier)
                    .unwrap();
                modifier.replace_input_value(graph, old_output, reshaped);
            }

            _ => (),
        }
    }
}

fn strip_scalar_mul<T: GraphOp>(
    graph: &Graph,
    modifier: &T,
    value: ValueId,
) -> (ValueId, Option<f64>) {
    let Some((def_node, _)) = modifier.defined_node(value) else {
        return (value, None);
    };
    let Operator::Mul = graph.nodes[def_node].op else {
        return (value, None);
    };
    if graph.nodes[def_node].inputs.len() != 2 {
        return (value, None);
    }
    // Single-consumer: pulling the scale out shouldn't affect other users.
    if modifier.used_node(value).map_or(true, |u| u.len() != 1) {
        return (value, None);
    }
    let a = graph.nodes[def_node].inputs[0].unwrap();
    let b = graph.nodes[def_node].inputs[1].unwrap();
    let (non_scalar, scalar_val) = match (graph.get_initializer(a), graph.get_initializer(b)) {
        (Some(t), None) => (b, t),
        (None, Some(t)) => (a, t),
        _ => return (value, None),
    };
    let Some(scalar) = scalar_val.data.to_scalar_data() else {
        return (value, None);
    };
    let ScalarData::Float(_, s) = scalar else {
        return (value, None);
    };
    (non_scalar, Some(s))
}

// Builds the additive mask tensor for the GPT-2-style
//   Mul(x, causal_0_1) -> Sub(_, penalty)
// pattern: 0 on the lower triangle, -penalty on the upper triangle.
// Returns None when the input tensors don't match the expected shape/content.
fn build_causal_additive_mask(
    graph: &Graph,
    causal_val: ValueId,
    penalty_val: ValueId,
) -> Option<Tensor> {
    let causal = graph.get_initializer(causal_val)?;
    let TensorData::Float(causal_ty, ref causal_data) = causal.data else {
        return None;
    };
    let dims = causal.dims.clone();
    let ndim = dims.ndim();
    if ndim < 2 {
        return None;
    }
    let (rows, cols) = (dims[ndim - 2], dims[ndim - 1]);
    if rows == 0 || cols == 0 {
        return None;
    }
    let n = rows * cols;
    if dims.size() % n != 0 {
        return None;
    }
    for batch_start in (0..dims.size()).step_by(n) {
        for r in 0..rows {
            for c in 0..cols {
                let v = causal_data[batch_start + r * cols + c];
                let expected = if c <= r { 1.0 } else { 0.0 };
                if v != expected {
                    return None;
                }
            }
        }
    }

    let penalty = graph.get_initializer(penalty_val)?;
    let TensorData::Float(penalty_ty, ref penalty_data) = penalty.data else {
        return None;
    };
    if penalty.dims != dims {
        return None;
    }
    let mut penalty_scalar: Option<f64> = None;
    for batch_start in (0..dims.size()).step_by(n) {
        for r in 0..rows {
            for c in 0..cols {
                let v = penalty_data[batch_start + r * cols + c];
                if c <= r {
                    if v != 0.0 {
                        return None;
                    }
                } else {
                    match penalty_scalar {
                        None => penalty_scalar = Some(v),
                        Some(p) if p != v => return None,
                        _ => (),
                    }
                }
            }
        }
    }
    let penalty = penalty_scalar.unwrap_or(f64::INFINITY);

    let mut additive = vec![0.0f64; dims.size()];
    for batch_start in (0..dims.size()).step_by(n) {
        for r in 0..rows {
            for c in 0..cols {
                if c > r {
                    additive[batch_start + r * cols + c] = -penalty;
                }
            }
        }
    }
    let elem_ty = if matches!(causal_ty, FloatType::F32) || matches!(penalty_ty, FloatType::F32) {
        FloatType::F32
    } else {
        FloatType::F64
    };
    Tensor::new(dims, TensorData::Float(elem_ty, additive)).ok()
}

#[derive(Default)]
pub struct MatMul2BatchedGemm {}

impl<T: GraphOp> Pass<T> for MatMul2BatchedGemm {
    fn summary(&self) -> &'static str {
        "Convert batched MatMul to BatchedGemm"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids: Vec<_> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.op, Operator::MatMul))
            .map(|(id, _)| id)
            .collect();

        for id in ids {
            if try_lower_matmul(id, graph, modifier).is_none() {
                // Leave the MatMul in place; only CPU can fall back to a
                // generic build_matmul kernel.
                continue;
            }
        }
    }
}

// Lowers a single MatMul to a BatchedGemm by promoting ranks and inserting
// Expand ops to broadcast mismatched batch dims. Returns None if the MatMul
// is 2D x 2D (already handled by Canonicalization -> Gemm) or has a broadcast
// that cannot be resolved (e.g. 3 vs 5 on a non-1 dim).
fn try_lower_matmul<T: GraphOp>(id: NodeId, graph: &mut Graph, modifier: &mut T) -> Option<()> {
    let lhs = graph.nodes[id].inputs[args::MATMUL_LHS].unwrap();
    let rhs = graph.nodes[id].inputs[args::MATMUL_RHS].unwrap();
    let l_dims = graph.get_resolved_tensor_type(lhs)?.dims.clone();
    let r_dims = graph.get_resolved_tensor_type(rhs)?.dims.clone();
    let ldim = l_dims.ndim();
    let rdim = r_dims.ndim();
    if ldim <= 2 && rdim <= 2 {
        return None;
    }

    let max_dim = ldim.max(rdim);
    let lhs_promoted = prepend_ones(graph, modifier, lhs, max_dim)?;
    let rhs_promoted = prepend_ones(graph, modifier, rhs, max_dim)?;

    let l_dims = graph.get_resolved_tensor_type(lhs_promoted)?.dims.clone();
    let r_dims = graph.get_resolved_tensor_type(rhs_promoted)?.dims.clone();
    let batch_ndim = max_dim - 2;
    let mut batch_dims = Vec::with_capacity(batch_ndim);
    for i in 0..batch_ndim {
        let a = l_dims[i];
        let b = r_dims[i];
        let resolved = if a == b {
            a
        } else if a == 1 {
            b
        } else if b == 1 {
            a
        } else {
            return None;
        };
        batch_dims.push(resolved);
    }

    let lhs_final = broadcast_batch(graph, modifier, id, "Lhs", lhs_promoted, &batch_dims)?;
    let rhs_final = broadcast_batch(graph, modifier, id, "Rhs", rhs_promoted, &batch_dims)?;

    let old_output = graph.nodes[id].outputs[0];
    let ty = graph.get_resolved_tensor_type(old_output)?.clone();
    let name = format!("MatMul2BatchedGemm_{:?}", id);
    let new_output = modifier.register_new_value(graph, name.clone(), ty);
    modifier.register_new_node(
        graph,
        Node {
            inputs: vec![Some(lhs_final), Some(rhs_final)],
            outputs: vec![new_output],
            name,
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
    Some(())
}

fn prepend_ones<T: GraphOp>(
    graph: &mut Graph,
    modifier: &mut T,
    value: ValueId,
    target_ndim: usize,
) -> Option<ValueId> {
    let ty = graph.get_resolved_tensor_type(value)?.clone();
    let cur_ndim = ty.dims.ndim();
    if cur_ndim == target_ndim {
        return Some(value);
    }
    let mut new_dims = vec![1usize; target_ndim - cur_ndim];
    new_dims.extend(ty.dims.iter().copied());
    ReshapeGenerator::default()
        .set_input(value)
        .set_dims(&new_dims)
        .set_allow_contiguous(true)
        .set_node_name(format!("MatMul2BatchedGemm_Promote_{}", value.index()))
        .set_value_name(format!("MatMul2BatchedGemm_Promote_{}", value.index()))
        .generate(graph, modifier)
        .ok()
}

// Expands `value` so its leading batch dims match `batch_dims`, leaving the
// last two matrix dims untouched. Returns `value` unchanged when no broadcast
// is needed.
fn broadcast_batch<T: GraphOp>(
    graph: &mut Graph,
    modifier: &mut T,
    id: NodeId,
    side_tag: &str,
    value: ValueId,
    batch_dims: &[usize],
) -> Option<ValueId> {
    let ty = graph.get_resolved_tensor_type(value)?.clone();
    let ndim = ty.dims.ndim();
    debug_assert!(ndim == batch_dims.len() + 2);
    if ty.dims[..batch_dims.len()] == *batch_dims {
        return Some(value);
    }

    let mut target_dims: Vec<usize> = batch_dims.to_vec();
    target_dims.push(ty.dims[ndim - 2]);
    target_dims.push(ty.dims[ndim - 1]);

    let shape_name = format!("MatMul2BatchedGemm_Shape_{side_tag}_{:?}", id);
    let shape_tensor = ResolvedTensorDims::new(&target_dims).to_tensor();
    let shape_val = modifier.register_new_tensor(graph, shape_tensor, shape_name);

    let expanded_ty = ResolvedTensorType::new(ty.elem_type, ResolvedTensorDims::new(&target_dims));
    let expanded = modifier.register_new_value(
        graph,
        format!("MatMul2BatchedGemm_Expand_{side_tag}_{:?}", id),
        expanded_ty,
    );
    modifier.register_new_node(
        graph,
        Node {
            inputs: vec![Some(value), Some(shape_val)],
            outputs: vec![expanded],
            name: format!("MatMul2BatchedGemm_Expand_{side_tag}_{:?}", id),
            op: Operator::Expand,
            meta: NodeMeta::default(),
        },
    );
    Some(expanded)
}
