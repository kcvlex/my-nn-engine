use crate::graph::operator::*;
use crate::graph::Graph;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct ConvActivationFusion {}

impl<T: GraphOp> Pass<T> for ConvActivationFusion {
    fn summary(&self) -> &'static str {
        "Conv + Activation Fusion"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let conv_nodes: Vec<_> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(&node.op, Operator::Conv(_)))
            .map(|(id, _)| id)
            .collect();

        for conv_id in conv_nodes {
            // cuDNN's cudnnConvolutionBiasActivationForward requires a bias tensor.
            // Only fuse when bias is present.
            let node = &graph.nodes[conv_id];
            if node.inputs.get(args::CONV_BIAS).is_none() {
                continue;
            }

            let conv_output = node.outputs[0];
            let Some(users) = modifier.used_node(conv_output) else {
                continue;
            };
            if users.len() != 1 {
                continue;
            }
            let (act_id, _) = users[0];
            let activation = match graph.nodes[act_id].op {
                Operator::ReLU => Activation::ReLU,
                _ => continue,
            };

            let Operator::Conv(ref conv) = graph.nodes[conv_id].op else {
                unreachable!();
            };
            if conv.activation != Activation::Identity {
                continue;
            }

            let Operator::Conv(ref mut conv) = graph.nodes[conv_id].op else {
                unreachable!();
            };
            conv.activation = activation;

            let act_output = graph.nodes[act_id].outputs[0];
            modifier.replace_input_value(graph, act_output, conv_output);
        }
    }
}
