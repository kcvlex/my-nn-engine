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
pub struct LayerNormFusion {}

impl<T: GraphOp> Pass<T> for LayerNormFusion {
    fn summary(&self) -> &'static str {
        "LayerNormalization Fusion"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let mean_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                matches!(
                    &node.op,
                    Operator::ReduceMean(Reduce { axes, keepdims, .. })
                    if axes.len() == 1 && axes[0] == -1 && *keepdims
                )
                .then_some(id)
            })
            .collect();

        for mean_node in mean_nodes {
            let Some(pattern) = match_layer_norm_pattern(graph, modifier, mean_node) else {
                continue;
            };
            let mut inputs = [pattern.input; 3];
            inputs[args::LAYER_NORM_SCALE] = pattern.scale;
            inputs[args::LAYER_NORM_BIAS] = pattern.bias;

            let old_output = graph.nodes[pattern.last_node].outputs[0];
            let new_output = modifier.register_new_value(
                graph,
                format!("LayerNormFusion_Output_{:?}", pattern.last_node),
                graph.get_resolved_tensor_type(old_output).unwrap().clone(),
            );

            modifier.register_new_node(
                graph,
                Node::create_node(
                    inputs.to_vec(),
                    vec![new_output],
                    format!("LayerNormFusion_{:?}", pattern.last_node),
                    Operator::LayerNormalization(LayerNormalization {
                        axis: TensorIndex::new(-1),
                        epsilon: pattern.epsilon,
                    }),
                ),
            );

            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}

#[derive(Debug)]
struct LayerNormPattern {
    last_node: NodeId,
    input: ValueId,
    scale: ValueId,
    bias: ValueId,
    epsilon: f64,
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
fn match_layer_norm_pattern<T: GraphOp>(
    graph: &Graph,
    modifier: &T,
    mean_node: NodeId,
) -> Option<LayerNormPattern> {
    let matcher = PatternMatcher::new(graph, modifier, (mean_node, 0));

    // Get input X from the mean node
    let x = graph.nodes[mean_node].inputs[0];

    let mut d: Option<ValueId> = None;
    let mut var: Option<ValueId> = None;
    let mut normalized: Option<ValueId> = None;
    let mut normalized_scale: Option<ValueId> = None;

    let mut var_eps_node: Option<NodeId> = None;
    let mut normalized_scale_node: Option<NodeId> = None;

    let last_node = matcher
        .then(|(node, mean)| {
            matches!(&node.op, Operator::Sub) && node.inputs[0] == x && node.inputs[1] == mean
        })?
        .capture_value(&mut d)
        .then(|(node, d)| match &node.op {
            Operator::Mul => {
                let lhs = node.inputs[0];
                let rhs = node.inputs[1];
                lhs == d && rhs == d
            }
            _ => false,
        })?
        .then(|(node, dd)| match &node.op {
            Operator::ReduceMean(Reduce { axes, .. }) => {
                axes.len() == 1 && axes[0] == -1 && node.inputs[0] == dd
            }
            _ => false,
        })?
        .capture_value(&mut var)
        .then(|(node, _)| matches!(&node.op, Operator::Add))?
        .capture_node(&mut var_eps_node)
        .then(|(node, _)| matches!(&node.op, Operator::Sqrt))?
        .then(|(node, _)| matches!(&node.op, Operator::Reciprocal))?
        .then(|(node, inv_stddev)| {
            let d = d.unwrap();
            matches!(&node.op, Operator::Mul) && node.inputs[0] == d && node.inputs[1] == inv_stddev
        })?
        .capture_value(&mut normalized)
        .then(|(node, _)| matches!(&node.op, Operator::Mul))?
        .capture_value(&mut normalized_scale)
        .capture_node(&mut normalized_scale_node)
        .then(|(node, _)| matches!(&node.op, Operator::Add))?
        .last_node();

    let var = var.unwrap();
    let var_eps_node = &graph.nodes[var_eps_node.unwrap()];
    let epsilon = extract_other_binary_input(var_eps_node, var)?;
    let epsilon = graph.initializer.get(&epsilon)?.data.to_scalar_data()?;
    let ScalarData::Float(_, epsilon) = epsilon else {
        return None;
    };

    let normalized = normalized.unwrap();
    let normalized_scale_node = &graph.nodes[normalized_scale_node.unwrap()];
    let scale = extract_other_binary_input(normalized_scale_node, normalized)?;

    let normalized_scale = normalized_scale.unwrap();
    let y_node = &graph.nodes[last_node];
    let bias = extract_other_binary_input(y_node, normalized_scale)?;

    Some(LayerNormPattern {
        last_node,
        input: x,
        scale,
        bias,
        epsilon,
    })
}
