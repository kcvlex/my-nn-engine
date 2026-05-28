use std::collections::HashMap;

use itertools::Itertools;
use num::traits::Euclid;

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

#[derive(Default)]
pub struct QuantizeActivations {}

impl<T: GraphOp> Pass<T> for QuantizeActivations {
    fn summary(&self) -> &'static str {
        "Rewrite DequantMatMul -> DynamicQuantizeLinear + QuantizedMatMul"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let candidates = graph
            .nodes
            .iter()
            .filter_map(|(id, n)| match n.op {
                Operator::DequantMatMul(_) => Some(id),
                _ => None,
            })
            .collect_vec();

        let mut impl_ = QuantizeActivationsImpl::default();
        for node_id in candidates {
            if let Some(plan) = Plan::build(graph, node_id) {
                impl_.apply(graph, modifier, &plan);
            }
        }
    }
}

struct Plan {
    node_id: NodeId,
    act_value: ValueId,
    weight_value: ValueId,
    weight_scale_value: ValueId,
    axis: TensorIndex,
    scale_float_ty: FloatType,
    m: usize,
}

impl Plan {
    fn build(graph: &Graph, node_id: NodeId) -> Option<Self> {
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
        let DataType::Float(scale_float_ty) = act_ty.elem_type else {
            return None;
        };
        if act_ty.dims.ndim() != 2 || act_ty.dims.last().unwrap() != &k {
            return None;
        }

        // Skip if the rewrite output type can't even be inferred.
        let (m, remainder) = act_ty.dims.size().div_rem_euclid(&k);
        if remainder != 0 {
            return None;
        }

        Some(Self {
            node_id,
            act_value,
            weight_value,
            weight_scale_value,
            axis: *axis,
            scale_float_ty,
            m,
        })
    }
}

#[derive(Default)]
struct QuantizeActivationsImpl {
    act_quant_cache: HashMap<ValueId, (ValueId, ValueId)>,
}

impl QuantizeActivationsImpl {
    fn ensure_quantized_act<T: GraphOp>(
        &mut self,
        graph: &mut Graph,
        modifier: &mut T,
        plan: &Plan,
    ) -> (ValueId, ValueId) {
        if let Some(&pair) = self.act_quant_cache.get(&plan.act_value) {
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
        let act_scale =
            modifier.register_new_value(graph, format!("{value_prefix}_scale"), scale_ty);
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

        self.act_quant_cache
            .insert(plan.act_value, (act_int8, act_scale));
        (act_int8, act_scale)
    }

    fn apply<T: GraphOp>(&mut self, graph: &mut Graph, modifier: &mut T, plan: &Plan) {
        let (act_int8, act_scale) = self.ensure_quantized_act(graph, modifier, plan);

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
}
