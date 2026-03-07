use itertools::Itertools;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::tensor::data::ScalarData;
use crate::tensor::data::TensorData;
use crate::transform::modify::GraphOp;
use crate::transform::pattern::extract_other_binary_input;
use crate::transform::pattern::PatternMatcher;
use crate::transform::utils::TransposeGenerator;
use crate::transform::Pass;

#[derive(Default)]
pub struct AttentionFusion {}

impl<T: GraphOp> Pass<T> for AttentionFusion {
    fn summary(&self) -> &'static str {
        "Attention Fusion"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let matmul_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(&node.op, Operator::MatMul))
            .map(|(id, _)| id)
            .collect();

        for matmul in matmul_nodes {
            let Some(pattern) = match_attention_pattern(graph, modifier, matmul) else {
                continue;
            };
            let AttentionPattern {
                last_node,
                q,
                k,
                v,
                scale,
                is_causal,
                penalty,
            } = pattern;
            let ndim = graph.get_resolved_tensor_type(k).unwrap().dims.ndim();
            let perm = (0..ndim - 2)
                .chain(std::iter::once(ndim - 1))
                .chain(std::iter::once(ndim - 2))
                .collect_vec();
            let k_transposed = TransposeGenerator::default()
                .set_input(k)
                .set_perm(perm)
                .set_node_name(format!("AttentionFusion_K_Transpose_{:?}", matmul))
                .set_value_name(format!("AttentionFusion_K_Transpose_Output_{:?}", matmul))
                .generate(graph, modifier)
                .unwrap();
            let old_output = graph.nodes[last_node].outputs[0];
            let mut inputs = vec![q; 3];
            inputs[args::ATTENTION_Q] = q;
            inputs[args::ATTENTION_K] = k_transposed;
            inputs[args::ATTENTION_V] = v;
            let new_output = modifier.register_new_value(
                graph,
                format!("AttentionFusion_Output{:?}", last_node),
                graph.get_resolved_tensor_type(old_output).unwrap().clone(),
            );
            modifier.register_new_node(
                graph,
                Node::create_node(
                    inputs,
                    vec![new_output],
                    format!("AttentionFusion{:?}", last_node),
                    Operator::Attention(Attention {
                        scale,
                        is_causal,
                        penalty,
                    }),
                ),
            );
            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}

struct AttentionPattern {
    last_node: NodeId,
    q: ValueId,
    k: ValueId,
    v: ValueId,
    scale: f32,
    is_causal: bool,
    penalty: Option<f32>,
}

fn match_attention_pattern<T: GraphOp>(
    graph: &Graph,
    modifier: &T,
    matmul_node: NodeId,
) -> Option<AttentionPattern> {
    let matcher = PatternMatcher::new(graph, modifier, (matmul_node, 0));
    let q = graph.nodes[matmul_node].inputs[0];
    let k = graph.nodes[matmul_node].inputs[1];

    let mut qk = None;
    let mut qk_scaled = None;
    let mut qk_softmax = None;

    let mut scale_node = None;
    let mut causal_node = None;
    let mut mask_node = None;

    let last_node = matcher
        .capture_value(&mut qk)
        .then(|(node, _)| matches!(&node.op, Operator::Mul))?
        .capture_value(&mut qk_scaled)
        .capture_node(&mut scale_node)
        .try_then(|(node, _)| matches!(&node.op, Operator::Mul))
        .map(|res| res.capture_node(&mut causal_node))
        .unwrap_or_else(|matcher| matcher)
        .try_then(|(node, v)| matches!(&node.op, Operator::Sub) && node.inputs[0] == v)
        .map(|res| res.capture_node(&mut mask_node))
        .unwrap_or_else(|matcher| matcher)
        .then(|(node, v)| match &node.op {
            Operator::Softmax(Softmax { axis }) => {
                let Some(ty) = graph.get_resolved_tensor_type(v) else {
                    return false;
                };
                let axis = axis.index(ty.dims.ndim());
                axis + 1 == ty.dims.ndim()
            }
            _ => false,
        })?
        .capture_value(&mut qk_softmax)
        .then(|(node, _)| matches!(&node.op, Operator::MatMul))?
        .last_node();

    let scale = extract_other_binary_input(&graph.nodes[scale_node?], qk?)?;
    let scale = graph.initializer.get(&scale)?.data.to_scalar_data()?;
    let ScalarData::Float(_, scale) = scale else {
        return None;
    };
    let scale = scale as f32;

    let (qk_row, qk_col) = {
        let dims = &graph.get_resolved_tensor_type(qk?)?.dims;
        let ndim = dims.ndim();
        (dims[ndim - 2], dims[ndim - 1])
    };

    if let Some(causal_node) = causal_node {
        let causal_mat = extract_other_binary_input(&graph.nodes[causal_node], qk_scaled?)?;
        let causal = graph.initializer.get(&causal_mat)?;
        let ty = &causal.tensor_type();
        let ndim = ty.dims.ndim();
        if ndim < 2 || ty.dims[ndim - 2] != qk_row || ty.dims[ndim - 1] != qk_col {
            return None;
        }
        let n = qk_row * qk_col;
        let TensorData::Float(_, ref data) = causal.data else {
            return None;
        };
        for i in (0..ty.dims.size()).step_by(n) {
            let slice = &data[i..i + n];
            for r in 0..qk_row {
                for c in 0..qk_col {
                    let idx = r * qk_col + c;
                    let ele = slice[idx];
                    let ok = if c <= r { ele == 1.0 } else { ele == 0.0 };
                    if !ok {
                        return None;
                    }
                }
            }
        }
    }

    let penalty = if let Some(mask_node) = mask_node {
        let mask = &graph.nodes[mask_node].inputs[1];
        let mask = graph.initializer.get(mask)?;
        let ty = &mask.tensor_type();
        let ndim = ty.dims.ndim();
        if ndim < 2 || ty.dims[ndim - 2] != qk_row || ty.dims[ndim - 1] != qk_col {
            return None;
        }
        let n = qk_row * qk_col;
        let TensorData::Float(_, ref data) = mask.data else {
            return None;
        };
        let penalty = data[1];
        for i in (0..ty.dims.size()).step_by(n) {
            let slice = &data[i..i + n];
            for r in 0..qk_row {
                for c in 0..qk_col {
                    let idx = r * qk_col + c;
                    let ele = slice[idx];
                    let ok = if c <= r { ele == 0.0 } else { ele == penalty };
                    if !ok {
                        return None;
                    }
                }
            }
        }
        Some(penalty as f32)
    } else {
        None
    };

    let v = extract_other_binary_input(&graph.nodes[last_node], qk_softmax?)?;

    Some(AttentionPattern {
        last_node,
        q,
        k,
        v,
        scale,
        is_causal: causal_node.is_some(),
        penalty,
    })
}
