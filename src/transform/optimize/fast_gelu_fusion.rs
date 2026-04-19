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
pub struct FastGeLUFusion {}

impl<T: GraphOp> Pass<T> for FastGeLUFusion {
    fn summary(&self) -> &'static str {
        "Fast GeLU Fusion"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let cube_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|(_, node)| is_cube(node, graph))
            .map(|(id, _)| id)
            .collect();

        for cube_node in cube_nodes {
            let Some(pattern) = match_fast_gelu_pattern(graph, modifier, cube_node) else {
                continue;
            };
            let inputs = vec![Some(pattern.input)];

            let old_output = graph.nodes[pattern.last_node].outputs[0];
            let new_output = modifier.register_new_value(
                graph,
                format!("FastGeLUFusion_Output{:?}", pattern.last_node),
                graph.get_resolved_tensor_type(old_output).unwrap().clone(),
            );

            modifier.register_new_node(
                graph,
                Node::create_node(
                    inputs,
                    vec![new_output],
                    format!("FastGeLUFusion{:?}", pattern.last_node),
                    Operator::GeLU(GeLU { approximate: true }),
                ),
            );

            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}

fn is_cube(node: &Node, graph: &Graph) -> bool {
    if !matches!(&node.op, Operator::Pow) {
        return false;
    }
    let Some(exponent) = graph.get_initializer(node.inputs[1].unwrap()) else {
        return false;
    };
    let Some(exponent) = exponent.data.to_scalar_data() else {
        return false;
    };
    matches!(
        exponent,
        ScalarData::SInt(_, 3) | ScalarData::UInt(_, 3) | ScalarData::Float(_, 3.0)
    )
}

struct FastGeLUPattern {
    last_node: NodeId,
    input: ValueId,
}

// From https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
//
//   Mean = ReduceMean<axes=normalized_axes>(X)
//   D = Sub(X, Mean)
//   DD = Mul(D, D)
//   Var = ReduceMean<axes=normalized_axes>(DD)
//   VarEps = Add(Var, epsilon)
//   StdDev = Sqrt(VarEps)
//   Normalized = Div(D, StdDev)
//   NormalizedScaled = Mul(Normalized, Scale)
//   Y = Add(NormalizedScaled, B)
fn match_fast_gelu_pattern<T: GraphOp>(
    graph: &Graph,
    modifier: &T,
    cube_node: NodeId,
) -> Option<FastGeLUPattern> {
    let matcher = PatternMatcher::new(graph, modifier, (cube_node, 0));
    let x = graph.nodes[cube_node].inputs[0].unwrap();

    let is_applied_constant = |node: &Node, known_input: ValueId, expected: f64| {
        let Some(other) = extract_other_binary_input(node, known_input) else {
            return false;
        };
        let Some(other) = graph.get_initializer(other) else {
            return false;
        };
        let Some(other) = other.data.to_scalar_data() else {
            return false;
        };
        let ScalarData::Float(_, other) = other else {
            return false;
        };
        (other - expected).abs() < 1e-6
    };

    let last_node = matcher
        .then(|(node, cube)| {
            if !matches!(&node.op, Operator::Mul) {
                return false;
            }
            is_applied_constant(node, cube, 0.044715)
        })?
        .then(|(node, mul)| {
            if !matches!(&node.op, Operator::Add) {
                return false;
            }
            let Some(other) = extract_other_binary_input(node, mul) else {
                return false;
            };
            other == x
        })?
        .then(|(node, add)| {
            if !matches!(&node.op, Operator::Mul) {
                return false;
            }
            is_applied_constant(node, add, (2.0f64 / std::f64::consts::PI).sqrt())
        })?
        .then(|(node, _)| matches!(&node.op, Operator::Tanh))?
        .then(|(node, tanh)| {
            if !matches!(&node.op, Operator::Add) {
                return false;
            }
            is_applied_constant(node, tanh, 1.0)
        })?
        .then(|(node, add)| {
            if !matches!(&node.op, Operator::Mul) {
                return false;
            }
            let Some(other) = extract_other_binary_input(node, add) else {
                return false;
            };
            // GPT-2 pattern: Mul(add, Mul(x, 0.5))
            if let Some((other_id, _)) = modifier.defined_node(other) {
                let other_node = &graph.nodes[other_id];
                if matches!(&other_node.op, Operator::Mul) &&
                    is_applied_constant(other_node, x, 0.5)
                {
                    return true;
                }
            }
            // BERT pattern: Mul(0.5, add)
            is_applied_constant(node, add, 0.5)
        })?
        // BERT pattern has an extra Mul(x, result) at the end
        .try_then(|(node, mul)| {
            matches!(&node.op, Operator::Mul) && extract_other_binary_input(node, mul) == Some(x)
        })
        .map_or_else(|m| m.last_node(), |m| m.last_node());

    Some(FastGeLUPattern {
        last_node,
        input: x,
    })
}
