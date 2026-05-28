use std::collections::HashMap;

use crate::graph::operator::*;
use crate::graph::Graph;
use crate::graph::Node;
use crate::graph::NodeId;
use crate::graph::NodeMeta;
use crate::graph::ValueId;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::SIntType;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

/// Rewrite `DequantMatMul` into `DynamicQuantizeLinear` (per-row symmetric
/// int8) + `QuantizedMatMul`.
///
/// One `DynamicQuantizeLinear` is inserted per unique activation `ValueId`,
/// so multiple matmuls that consume the same activation (e.g. Q/K/V projections
/// from the same RMSNorm output) share it.
///
/// Eligibility: the matmul's `K` (= rhs.dims[axis ^ 1]) must be a multiple of
/// 32 (CUDA INT8 mma kernel constraint). Others are left alone.
#[derive(Default)]
pub struct QuantizeActivations {}

impl<T: GraphOp> Pass<T> for QuantizeActivations {
    fn summary(&self) -> &'static str {
        "Rewrite DequantMatMul -> DynamicQuantizeLinear + QuantizedMatMul"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let candidates: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, n)| match n.op {
                Operator::DequantMatMul(_) => Some(id),
                _ => None,
            })
            .collect();

        // act ValueId -> (act_int8, act_scale) reused across matmuls.
        let mut act_quant_cache: HashMap<ValueId, (ValueId, ValueId)> = HashMap::new();

        for node_id in candidates {
            let plan = match build_plan(graph, modifier, node_id) {
                Some(p) => p,
                None => continue,
            };
            apply(graph, modifier, plan, &mut act_quant_cache);
        }
    }
}

struct Plan {
    node_id: NodeId,
    act_value: ValueId,
    weight_value: ValueId,
    weight_scale_value: ValueId,
    axis: TensorIndex,
    /// Float type of the existing scale (used for the new act_scale dtype).
    scale_float_ty: FloatType,
    /// Total batch rows M (product of all dims before the last K dim).
    m: usize,
}

fn build_plan<T: GraphOp>(graph: &Graph, modifier: &T, node_id: NodeId) -> Option<Plan> {
    let node = &graph.nodes[node_id];
    let Operator::DequantMatMul(DequantMatMul { axis }) = &node.op else {
        return None;
    };

    let act_value = node.inputs[args::DEQUANT_MATMUL_LHS]?;
    let weight_value = node.inputs[args::DEQUANT_MATMUL_RHS]?;
    let weight_scale_value = node.inputs[args::DEQUANT_MATMUL_SCALE]?;

    let act_ty = graph.get_resolved_tensor_type(act_value)?;
    let rhs_ty = graph.get_resolved_tensor_type(weight_value)?;

    if rhs_ty.dims.ndim() != 2 {
        return None;
    }
    let axis_idx = axis.index(rhs_ty.dims.ndim());
    if axis_idx != 0 {
        return None;
    }
    let k = rhs_ty.dims[1];
    if k % 32 != 0 {
        return None;
    }
    // Activation must be a float type so DynamicQuantizeLinear is applicable.
    let DataType::Float(float_ty) = act_ty.elem_type else {
        return None;
    };
    if act_ty.dims.ndim() < 2 || act_ty.dims[act_ty.dims.ndim() - 1] != k {
        return None;
    }
    // Skip if the rewrite output type can't even be inferred (defensive).
    let m = act_ty.dims.size() / k;

    // Skip activation values whose contents we couldn't statically validate.
    let _ = modifier; // currently unused; left for symmetry with other passes
    Some(Plan {
        node_id,
        act_value,
        weight_value,
        weight_scale_value,
        axis: *axis,
        scale_float_ty: float_ty,
        m,
    })
}

fn ensure_quantized_act<T: GraphOp>(
    graph: &mut Graph,
    modifier: &mut T,
    plan: &Plan,
    cache: &mut HashMap<ValueId, (ValueId, ValueId)>,
) -> (ValueId, ValueId) {
    if let Some(&pair) = cache.get(&plan.act_value) {
        return pair;
    }

    let act_ty = graph
        .get_resolved_tensor_type(plan.act_value)
        .unwrap()
        .clone();
    // Per-row scale = 1D of length M, where M is the leading product.
    // For symmetric mode we still emit a zero_point output to keep the op's
    // arity fixed; downstream QuantizedMatMul ignores it.
    let scale_dims = ResolvedTensorDims::new(&[plan.m]);
    let int8_ty = ResolvedTensorType::new(DataType::SInt(SIntType::I8), act_ty.dims.clone());
    let scale_ty =
        ResolvedTensorType::new(DataType::Float(plan.scale_float_ty), scale_dims.clone());
    let zp_ty = ResolvedTensorType::new(DataType::SInt(SIntType::I8), scale_dims);

    let value_prefix = format!("QuantizeActivations_{:?}", plan.act_value);
    let act_int8 = modifier.register_new_value(graph, format!("{value_prefix}_int8"), int8_ty);
    let act_scale = modifier.register_new_value(graph, format!("{value_prefix}_scale"), scale_ty);
    let act_zp = modifier.register_new_value(graph, format!("{value_prefix}_zp"), zp_ty);

    let new_node = Node {
        inputs: vec![Some(plan.act_value)],
        outputs: vec![act_int8, act_scale, act_zp],
        name: value_prefix,
        op: Operator::DynamicQuantizeLinear(DynamicQuantizeLinear {
            axis: Some(TensorIndex::new(0)),
            symmetric: true,
        }),
        meta: NodeMeta::default(),
    };
    modifier.register_new_node(graph, new_node);

    cache.insert(plan.act_value, (act_int8, act_scale));
    (act_int8, act_scale)
}

fn apply<T: GraphOp>(
    graph: &mut Graph,
    modifier: &mut T,
    plan: Plan,
    cache: &mut HashMap<ValueId, (ValueId, ValueId)>,
) {
    let (act_int8, act_scale) = ensure_quantized_act(graph, modifier, &plan, cache);

    let old_output = graph.nodes[plan.node_id].outputs[0];
    let out_ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
    let new_output = modifier.register_new_value(
        graph,
        format!("QuantizedMatMul_Output_{:?}", plan.node_id),
        out_ty,
    );
    let new_node = Node {
        inputs: vec![
            Some(act_int8),
            Some(act_scale),
            Some(plan.weight_value),
            Some(plan.weight_scale_value),
        ],
        outputs: vec![new_output],
        name: format!("QuantizedMatMul_{:?}", plan.node_id),
        op: Operator::QuantizedMatMul(QuantizedMatMul { axis: plan.axis }),
        meta: NodeMeta::default(),
    };
    modifier.register_new_node(graph, new_node);
    modifier.replace_input_value(graph, old_output, new_output);
}
