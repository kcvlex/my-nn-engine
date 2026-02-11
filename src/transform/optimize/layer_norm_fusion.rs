use itertools::Itertools;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::tensor::data::ScalarData;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct LayerNormFusion {}

impl<T: GraphOp> Pass<T> for LayerNormFusion {
    fn summary(&self) -> &'static str {
        "LayerNormalization Fusion"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if let Operator::ReduceMean(_) = &node.op {
                    Some(id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for id in ids {
            let Some(pattern) = match_pattern(graph, modifier, id) else {
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
                Node {
                    inputs: inputs.to_vec(),
                    outputs: vec![new_output],
                    name: format!("LayerNormFusion_{:?}", pattern.last_node),
                    op: Operator::LayerNormalization(LayerNormalization {
                        axis: TensorIndex::new(-1),
                        epsilon: pattern.epsilon,
                    }),
                    meta: NodeMeta::default(),
                },
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
//   InvStdDev = Reciprocal(StdDev)
//   Normalized = Mul(D, InvStdDev)
//   NormalizedScaled = Mul(Normalized, Scale)
//   Y = Add(NormalizedScaled, B)
fn match_pattern<T: GraphOp>(
    graph: &Graph,
    modifier: &T,
    mean_node: NodeId,
) -> Option<LayerNormPattern> {
    let const_scalar = |id: ValueId| {
        let data = graph.initializer.get(&id)?.data.to_scalar_data()?;
        Some(data)
    };

    let just_one_consumer = |id: NodeId| {
        let users = modifier
            .used_node(graph.nodes[id].outputs[0])?
            .iter()
            .map(|(node_id, _)| *node_id)
            .unique()
            .collect::<Vec<_>>();
        if users.len() == 1 {
            Some(users[0])
        } else {
            None
        }
    };

    let just_two_consumers = |id: NodeId| {
        let users = modifier
            .used_node(graph.nodes[id].outputs[0])?
            .iter()
            .map(|(node_id, _)| *node_id)
            .unique()
            .collect::<Vec<_>>();
        if users.len() == 2 {
            Some((users[0], users[1]))
        } else {
            None
        }
    };

    let another_input_of_binop = |node: NodeId, input: ValueId| {
        let n = &graph.nodes[node];
        if n.inputs[0] == input {
            Some(n.inputs[1])
        } else if n.inputs[1] == input {
            Some(n.inputs[0])
        } else {
            None
        }
    };

    let check_reduce_mean = |node: NodeId| {
        // TODO: Should use TensorIndex?
        match &graph.nodes[node].op {
            Operator::ReduceMean(Reduce { axes, .. }) => axes.len() == 1 && axes[0] == -1,
            _ => false,
        }
    };

    macro_rules! match_op {
        ($node_id: expr, $op_variant: pat) => {
            matches!(graph.nodes[$node_id].op, $op_variant)
        };
    }

    let x = graph.nodes[mean_node].inputs[0];
    let mean = graph.nodes[mean_node].outputs[0];
    if !check_reduce_mean(mean_node) {
        return None;
    }

    let d_node = just_one_consumer(mean_node)?;
    if !match_op!(d_node, Operator::Sub) {
        return None;
    }
    let d_node_output = graph.nodes[d_node].outputs[0];

    let d_node_lhs = graph.nodes[d_node].inputs[0];
    let d_node_rhs = graph.nodes[d_node].inputs[1];
    if (d_node_lhs, d_node_rhs) != (x, mean) {
        return None;
    }

    let (dd_node, normalized_node) = just_two_consumers(d_node)?;
    let (dd_node, normalized_node) =
        match (&graph.nodes[dd_node].op, &graph.nodes[normalized_node].op) {
            (_, Operator::Div) => (dd_node, normalized_node),
            (Operator::Div, _) => (normalized_node, dd_node),
            _ => return None,
        };

    {
        let dd_node = &graph.nodes[dd_node];
        match dd_node.op {
            Operator::Mul => {
                let dd_node_lhs = dd_node.inputs[0];
                let dd_node_rhs = dd_node.inputs[1];
                if (dd_node_lhs, dd_node_rhs) != (d_node_output, d_node_output) {
                    return None;
                }
            }
            Operator::Pow => {
                let dd_node_lhs = dd_node.inputs[0];
                let dd_node_rhs = dd_node.inputs[1];
                if dd_node_lhs != d_node_output {
                    return None;
                }
                let dd_node_rhs = const_scalar(dd_node_rhs)?;
                match dd_node_rhs {
                    ScalarData::SInt(_, 2) | ScalarData::UInt(_, 2) | ScalarData::Float(_, 2.0) => {
                        ()
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }
    }

    let var_node = just_one_consumer(dd_node)?;
    if !check_reduce_mean(var_node) {
        return None;
    }

    let var_eps_node = just_one_consumer(var_node)?;
    let epsilon = match &graph.nodes[var_eps_node].op {
        Operator::Add => {
            let var = graph.nodes[var_node].outputs[0];
            let eps = another_input_of_binop(var_eps_node, var)?;
            let eps = const_scalar(eps)?;
            match eps {
                ScalarData::Float(_, v) => v,
                _ => return None,
            }
        }
        _ => return None,
    };

    let std_dev_node = just_one_consumer(var_eps_node)?;
    if !match_op!(std_dev_node, Operator::Sqrt) {
        return None;
    }

    assert!(match_op!(normalized_node, Operator::Div));
    let normalized_lhs = graph.nodes[normalized_node].inputs[0];
    let normalized_rhs = graph.nodes[normalized_node].inputs[1];
    if (normalized_lhs, normalized_rhs) != (d_node_output, graph.nodes[std_dev_node].outputs[0]) {
        return None;
    }

    let normalized_scaled_node = just_one_consumer(normalized_node)?;
    let scale = match &graph.nodes[normalized_scaled_node].op {
        Operator::Mul => {
            let normalized = graph.nodes[normalized_node].outputs[0];
            another_input_of_binop(normalized_scaled_node, normalized)?
        }
        _ => return None,
    };

    let last_node = just_one_consumer(normalized_scaled_node)?;
    let bias = match &graph.nodes[last_node].op {
        Operator::Add => {
            let normalized_scaled = graph.nodes[normalized_scaled_node].outputs[0];
            another_input_of_binop(last_node, normalized_scaled)?
        }
        _ => return None,
    };

    Some(LayerNormPattern {
        last_node,
        input: x,
        scale,
        bias,
        epsilon,
    })
}
