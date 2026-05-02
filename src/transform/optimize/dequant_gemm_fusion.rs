use crate::graph::operator::*;
use crate::graph::Graph;
use crate::graph::Node;
use crate::graph::NodeId;
use crate::graph::NodeMeta;
use crate::graph::ValueId;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct DequantGemmFusion {}

impl<T: GraphOp> Pass<T> for DequantGemmFusion {
    fn summary(&self) -> &'static str {
        "Fuse DequantizeLinear + Gemm into DequantMatMul"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| match node.op {
                Operator::Gemm(_) => Some(id),
                _ => None,
            })
            .collect::<Vec<_>>();

        for gemm_id in ids {
            let Some(plan) = match_pattern(graph, modifier, gemm_id) else {
                continue;
            };
            apply(graph, modifier, plan);
        }
    }
}

struct FusionPlan {
    gemm_id: NodeId,
    act_value: ValueId,
    weight_value: ValueId,
    scale_value: ValueId,
    axis: TensorIndex,
}

fn match_pattern<T: GraphOp>(graph: &Graph, modifier: &T, gemm_id: NodeId) -> Option<FusionPlan> {
    let gemm_node = &graph.nodes[gemm_id];
    let gemm = match &gemm_node.op {
        Operator::Gemm(g) => g,
        _ => return None,
    };
    // Only fuse the (A @ B.T) shape: trans_a=false, trans_b=true. alpha=1, beta=*.
    if gemm.trans_a || !gemm.trans_b {
        return None;
    }
    if gemm.alpha != 1.0 {
        return None;
    }
    // Bias (C) makes fusion non-trivial — skip for now.
    if gemm_node
        .inputs
        .get(args::GEMM_C)
        .and_then(|x| *x)
        .is_some()
    {
        return None;
    }

    let act_value = gemm_node.inputs[args::GEMM_A]?;
    let dq_value = gemm_node.inputs[args::GEMM_B]?;

    // dq_value must be the unique-consumed output of a DequantizeLinear(axis=0).
    let dequant_id = modifier.defined_node(dq_value)?.0;
    let dequant_node = &graph.nodes[dequant_id];
    let dequant = match &dequant_node.op {
        Operator::DequantizeLinear(d) => d,
        _ => return None,
    };
    if dequant.axis.raw() != 0 {
        return None;
    }
    if modifier.used_node(dq_value).map(|s| s.len()).unwrap_or(0) != 1 {
        return None;
    }

    let weight_value = dequant_node.inputs[args::DEQUANTIZE_X]?;
    let scale_value = dequant_node.inputs[args::DEQUANTIZE_SCALE]?;

    Some(FusionPlan {
        gemm_id,
        act_value,
        weight_value,
        scale_value,
        axis: dequant.axis,
    })
}

fn apply<T: GraphOp>(graph: &mut Graph, modifier: &mut T, plan: FusionPlan) {
    let FusionPlan {
        gemm_id,
        act_value,
        weight_value,
        scale_value,
        axis,
    } = plan;

    let old_output = graph.nodes[gemm_id].outputs[0];
    let ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
    let new_output =
        modifier.register_new_value(graph, format!("DequantGemmFusion_Output_{:?}", gemm_id), ty);
    let new_node = Node {
        inputs: vec![Some(act_value), Some(weight_value), Some(scale_value)],
        outputs: vec![new_output],
        name: format!("DequantGemmFusion_{:?}", gemm_id),
        op: Operator::DequantMatMul(DequantMatMul { axis }),
        meta: NodeMeta::default(),
    };
    modifier.register_new_node(graph, new_node);
    modifier.replace_input_value(graph, old_output, new_output);
}
