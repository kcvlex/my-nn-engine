use itertools::Itertools;

use crate::graph::operator::*;
use crate::graph::Graph;
use crate::graph::Node;
use crate::graph::NodeId;
use crate::graph::ValueId;
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
                mask,
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
            let mut inputs = vec![Some(q); if mask.is_some() { 4 } else { 3 }];
            inputs[args::ATTENTION_Q] = Some(q);
            inputs[args::ATTENTION_K] = Some(k_transposed);
            inputs[args::ATTENTION_V] = Some(v);
            if let Some(mask) = mask {
                inputs[args::ATTENTION_MASK] = Some(mask);
            }
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
                    Operator::Attention(Attention { is_causal, scale }),
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
    mask: Option<ValueId>,
}

// A mask is "causal" if it's a constant additive mask with 0 on the lower
// triangle (c <= r) and a non-zero negative value on the upper triangle.
fn detect_causal_mask(graph: &Graph, mask: ValueId) -> bool {
    let Some(tensor) = graph.get_initializer(mask) else {
        return false;
    };
    let TensorData::Float(_, ref data) = tensor.data else {
        return false;
    };
    let dims = &tensor.dims;
    let ndim = dims.ndim();
    if ndim < 2 {
        return false;
    }
    let (rows, cols) = (dims[ndim - 2], dims[ndim - 1]);
    if rows != cols || rows < 2 {
        return false;
    }
    let n = rows * cols;
    if dims.size() % n != 0 {
        return false;
    }
    for batch_start in (0..dims.size()).step_by(n) {
        for r in 0..rows {
            for c in 0..cols {
                let v = data[batch_start + r * cols + c];
                if c <= r {
                    if v != 0.0 {
                        return false;
                    }
                } else if v >= 0.0 {
                    return false;
                }
            }
        }
    }
    true
}

fn match_attention_pattern<T: GraphOp>(
    graph: &mut Graph,
    modifier: &mut T,
    matmul_node: NodeId,
) -> Option<AttentionPattern> {
    let matcher = PatternMatcher::new(graph, modifier, (matmul_node, 0));
    let q = graph.nodes[matmul_node].inputs[0].unwrap();
    let k = graph.nodes[matmul_node].inputs[1].unwrap();

    let mut qk = None;
    let mut qk_scaled = None;
    let mut qk_softmax = None;
    let mut scale_node = None;
    let mut add_node = None;

    let matcher = matcher
        .capture_value(&mut qk)
        .then(|(node, _)| matches!(&node.op, Operator::Mul))?
        .capture_value(&mut qk_scaled)
        .capture_node(&mut scale_node);

    // Optional Add(qk_scaled, mask) -- additive mask. The other input can be
    // an initializer (BERT/canonicalized GPT-2) or a runtime value (TinyLlama,
    // where the mask is the output of a Where).
    let matcher = matcher
        .try_then(|(node, v)| {
            matches!(&node.op, Operator::Add) &&
                (node.inputs[0].unwrap() == v || node.inputs[1].unwrap() == v)
        })
        .map(|res| res.capture_node(&mut add_node))
        .unwrap_or_else(|matcher| matcher);

    let matcher = matcher.then(|(node, v)| match &node.op {
        Operator::Softmax(Softmax { axis }) => {
            let Some(ty) = graph.get_resolved_tensor_type(v) else {
                return false;
            };
            let axis = axis.index(ty.dims.ndim());
            axis + 1 == ty.dims.ndim()
        }
        _ => false,
    })?;

    // Optional NaN-guard passthrough: Where(IsNaN(softmax), 0_const, softmax).
    let matcher = matcher
        .try_then(|(node, v)| {
            if !matches!(&node.op, Operator::Where) || node.inputs.len() != 3 {
                return false;
            }
            let cond = node.inputs[0].unwrap();
            let t = node.inputs[1].unwrap();
            let f = node.inputs[2].unwrap();
            if f != v {
                return false;
            }
            let Some(tt) = graph.get_initializer(t) else {
                return false;
            };
            let Some(ScalarData::Float(_, val)) = tt.data.to_scalar_data() else {
                return false;
            };
            if val != 0.0 {
                return false;
            }
            let Some(producer) = graph
                .nodes
                .iter()
                .find_map(|(id, n)| n.outputs.contains(&cond).then_some(id))
            else {
                return false;
            };
            matches!(&graph.nodes[producer].op, Operator::IsNaN) &&
                graph.nodes[producer].inputs[0].unwrap() == v
        })
        .unwrap_or_else(|m| m);

    let last_node = matcher
        .capture_value(&mut qk_softmax)
        .then(|(node, _)| matches!(&node.op, Operator::MatMul))?
        .last_node();

    let scale = extract_other_binary_input(&graph.nodes[scale_node?], qk?)?;
    let scale = graph.get_initializer(scale)?.data.to_scalar_data()?;
    let ScalarData::Float(_, scale) = scale else {
        return None;
    };
    let scale = scale as f32;

    let mask = if let Some(add_node) = add_node {
        Some(extract_other_binary_input(
            &graph.nodes[add_node],
            qk_scaled?,
        )?)
    } else {
        None
    };
    let is_causal = mask.map(|m| detect_causal_mask(graph, m)).unwrap_or(false);

    let v = extract_other_binary_input(&graph.nodes[last_node], qk_softmax?)?;

    Some(AttentionPattern {
        last_node,
        q,
        k,
        v,
        scale,
        is_causal,
        mask,
    })
}
