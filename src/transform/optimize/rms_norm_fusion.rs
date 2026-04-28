use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::tensor::data::ScalarData;
use crate::transform::modify::GraphOp;
use crate::transform::pattern::extract_other_binary_input;
use crate::transform::pattern::PatternMatcher;
use crate::transform::Pass;

#[derive(Default)]
pub struct RMSNormFusion {}

fn is_square(node: &Node, graph: &Graph) -> Option<ValueId> {
    match &node.op {
        Operator::Mul => {
            if node.inputs.len() != 2 {
                return None;
            }
            let lhs = node.inputs[0]?;
            let rhs = node.inputs[1]?;
            (lhs == rhs).then_some(lhs)
        }
        Operator::Pow => {
            let exp = node.inputs.get(1).and_then(|x| *x)?;
            let sd = graph.get_initializer(exp)?.data.to_scalar_data()?;
            if matches!(
                sd,
                ScalarData::SInt(_, 2) | ScalarData::UInt(_, 2) | ScalarData::Float(_, 2.0)
            ) {
                node.inputs[0]
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_one(graph: &Graph, value: ValueId) -> bool {
    let Some(t) = graph.get_initializer(value) else {
        return false;
    };
    let Some(sd) = t.data.to_scalar_data() else {
        return false;
    };
    matches!(
        sd,
        ScalarData::SInt(_, 1) | ScalarData::UInt(_, 1) | ScalarData::Float(_, 1.0)
    )
}

fn is_reduce_mean_last_axis(node: &Node, ndim: usize) -> bool {
    let Operator::ReduceMean(Reduce { axes, keepdims }) = &node.op else {
        return false;
    };
    *keepdims && axes.len() == 1 && TensorIndex::new(axes[0] as isize).index(ndim) == ndim - 1
}

impl<T: GraphOp> Pass<T> for RMSNormFusion {
    fn summary(&self) -> &'static str {
        "RMSNormalization Fusion"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let square_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|(_, node)| is_square(node, graph).is_some())
            .map(|(id, _)| id)
            .collect();
        for square_node in square_nodes {
            let Some(pattern) = match_rms_norm_pattern(graph, modifier, square_node) else {
                continue;
            };

            let inputs = vec![Some(pattern.input), Some(pattern.scale)];
            let old_output = graph.nodes[pattern.last_node].outputs[0];
            let new_output = modifier.register_new_value(
                graph,
                format!("RMSNormFusion_Output_{:?}", pattern.last_node),
                graph.get_resolved_tensor_type(old_output).unwrap().clone(),
            );

            modifier.register_new_node(
                graph,
                Node::create_node(
                    inputs,
                    vec![new_output],
                    format!("RMSNormFusion_{:?}", pattern.last_node),
                    Operator::RMSNormalization(RMSNormalization {
                        axis: TensorIndex::new(-1),
                        epsilon: pattern.epsilon,
                    }),
                ),
            );

            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}

struct RMSNormPattern {
    last_node: NodeId,
    input: ValueId,
    scale: ValueId,
    epsilon: f64,
}

// RMSNorm pattern (TinyLlama / HF Llama):
//   XSquared = Pow(X, 2)
//   MeanSq = ReduceMean<axes=[-1], keepdims=true>(XSquared)
//   MeanSqEps = Add(MeanSq, epsilon)
//   StdDev = Sqrt(MeanSqEps)
//   Recip = Div(1, StdDev)            (or Reciprocal(StdDev))
//   Normalized = Mul(X, Recip)
//   [optional Cast]
//   Y = Mul(Normalized, Scale)
fn match_rms_norm_pattern<T: GraphOp>(
    graph: &Graph,
    modifier: &T,
    square_node: NodeId,
) -> Option<RMSNormPattern> {
    let square = &graph.nodes[square_node];
    let x = is_square(square, graph)?;
    let ndim = graph.get_resolved_tensor_type(x)?.dims.ndim();

    let mut reduce_mean_value: Option<ValueId> = None;
    let mut add_node: Option<NodeId> = None;
    let mut recip_value: Option<ValueId> = None;
    let mut pre_scale: Option<ValueId> = None;

    let matcher = PatternMatcher::new(graph, modifier, (square_node, 0))
        .then(|(node, _)| is_reduce_mean_last_axis(node, ndim))?
        .capture_value(&mut reduce_mean_value)
        .then(|(node, _)| matches!(&node.op, Operator::Add))?
        .capture_node(&mut add_node)
        .then(|(node, _)| matches!(&node.op, Operator::Sqrt))?
        .then(|(node, sqrt_v)| match &node.op {
            Operator::Div => {
                node.inputs[1].unwrap() == sqrt_v && is_one(graph, node.inputs[0].unwrap())
            }
            Operator::Reciprocal => true,
            _ => false,
        })?
        .capture_value(&mut recip_value);

    // Canonicalization may rewrite Div(1, sqrt) into Mul(1, Reciprocal(sqrt));
    // accept a transparent Mul-by-one between Reciprocal and the X-normalize Mul.
    let matcher = matcher
        .try_then(|(node, prev)| {
            matches!(&node.op, Operator::Mul) &&
                extract_other_binary_input(node, prev).is_some_and(|other| is_one(graph, other))
        })
        .unwrap_or_else(|m| m);

    let matcher = matcher.then(|(node, recip)| {
        matches!(&node.op, Operator::Mul) && extract_other_binary_input(node, recip) == Some(x)
    })?;

    let matcher = matcher
        .try_then(|(node, _)| matches!(&node.op, Operator::Cast(_)))
        .unwrap_or_else(|m| m);

    let matcher = matcher
        .capture_value(&mut pre_scale)
        .then(|(node, _)| matches!(&node.op, Operator::Mul))?;

    let last_node = matcher.last_node();
    let pre_scale = pre_scale?;
    let scale = extract_other_binary_input(&graph.nodes[last_node], pre_scale)?;

    let add_node = add_node?;
    let reduce_mean_value = reduce_mean_value?;
    let eps_value = extract_other_binary_input(&graph.nodes[add_node], reduce_mean_value)?;
    let epsilon = match graph.get_initializer(eps_value)?.data.to_scalar_data()? {
        ScalarData::Float(_, e) => e,
        _ => return None,
    };

    let _ = recip_value?;
    Some(RMSNormPattern {
        last_node,
        input: x,
        scale,
        epsilon,
    })
}
