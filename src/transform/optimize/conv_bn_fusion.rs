use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::operator::*;
use crate::tensor::data::TensorData;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::Tensor;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct ConvBNFusion {}

impl<T: GraphOp> Pass<T> for ConvBNFusion {
    fn summary(&self) -> &'static str {
        "Conv + BatchNormalization Fusion"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let conv_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(&node.op, Operator::Conv(_)))
            .map(|(id, _)| id)
            .collect();

        for conv_node_id in conv_nodes {
            let Some(bn_node_id) = find_bn_user(graph, modifier, conv_node_id) else {
                continue;
            };

            let conv_node = &graph.nodes[conv_node_id];
            let bn_node = &graph.nodes[bn_node_id];

            let conv_weight_id = conv_node.inputs[args::CONV_WEIGHT].unwrap();
            let has_conv_bias = conv_node.inputs.len() > args::CONV_BIAS;
            let conv_bias_id = has_conv_bias.then(|| conv_node.inputs[args::CONV_BIAS].unwrap());

            let bn_scale_id = bn_node.inputs[args::BATCHNORM_SCALE].unwrap();
            let bn_bias_id = bn_node.inputs[args::BATCHNORM_BIAS].unwrap();
            let bn_mean_id = bn_node.inputs[args::BATCHNORM_MEAN].unwrap();
            let bn_var_id = bn_node.inputs[args::BATCHNORM_VAR].unwrap();
            let Operator::BatchNormalization(BatchNormalization { epsilon, .. }) = &bn_node.op
            else {
                unreachable!()
            };
            let epsilon = *epsilon as f64;

            // All BN parameters must be initializers
            let (Some(bn_scale), Some(bn_bias), Some(bn_mean), Some(bn_var)) = (
                graph.get_initializer(bn_scale_id).map(|t| t.clone()),
                graph.get_initializer(bn_bias_id).map(|t| t.clone()),
                graph.get_initializer(bn_mean_id).map(|t| t.clone()),
                graph.get_initializer(bn_var_id).map(|t| t.clone()),
            ) else {
                continue;
            };

            let Some(conv_weight) = graph.get_initializer(conv_weight_id).map(|t| t.clone()) else {
                continue;
            };

            let conv_bias =
                conv_bias_id.and_then(|id| graph.get_initializer(id).map(|t| t.clone()));

            // Extract float data
            let (Some(scale), Some(bias), Some(mean), Some(var)) = (
                bn_scale.to_1d_floats(),
                bn_bias.to_1d_floats(),
                bn_mean.to_1d_floats(),
                bn_var.to_1d_floats(),
            ) else {
                continue;
            };

            let TensorData::Float(float_ty, ref weight_data) = conv_weight.data else {
                continue;
            };

            let num_channels = scale.len();

            // Compute multiplier: scale / sqrt(var + epsilon)
            let multiplier: Vec<f64> = scale
                .iter()
                .zip(var.iter())
                .map(|(s, v)| s / (v + epsilon).sqrt())
                .collect();

            // new_weight[c_out] = weight[c_out] * multiplier[c_out]
            // Conv weight shape: [C_out, C_in/groups, kH, kW, ...]
            let channel_size = weight_data.len() / num_channels;
            let new_weight_data: Vec<f64> = weight_data
                .chunks(channel_size)
                .zip(multiplier.iter())
                .flat_map(|(chunk, m)| chunk.iter().map(move |w| w * m))
                .collect();

            let new_weight = Tensor::new(
                conv_weight.dims.clone(),
                TensorData::Float(float_ty, new_weight_data),
            )
            .unwrap();

            // new_bias = bn_bias - bn_mean * multiplier [+ conv_bias * multiplier]
            let old_conv_bias = conv_bias.and_then(|t| t.to_1d_floats());
            let new_bias_data: Vec<f64> = (0..num_channels)
                .map(|c| {
                    let base = bias[c] - mean[c] * multiplier[c];
                    if let Some(ref cb) = old_conv_bias {
                        base + cb[c] * multiplier[c]
                    } else {
                        base
                    }
                })
                .collect();

            let new_bias_tensor = Tensor::new(
                ResolvedTensorDims::new(&[num_channels]),
                TensorData::Float(float_ty, new_bias_data),
            )
            .unwrap();

            // Replace conv weight initializer
            let new_weight_id = modifier.register_new_tensor(
                graph,
                new_weight,
                format!("ConvBNFusion_Weight_{:?}", conv_node_id),
            );

            // Create new bias initializer
            let new_bias_id = modifier.register_new_tensor(
                graph,
                new_bias_tensor,
                format!("ConvBNFusion_Bias_{:?}", conv_node_id),
            );

            // Build new Conv node with fused weight and bias
            let conv_node = &graph.nodes[conv_node_id];
            let mut new_inputs = conv_node.inputs.clone();
            new_inputs[args::CONV_WEIGHT] = Some(new_weight_id);
            if has_conv_bias {
                new_inputs[args::CONV_BIAS] = Some(new_bias_id);
            } else {
                new_inputs.push(Some(new_bias_id));
            }

            let bn_output = graph.nodes[bn_node_id].outputs[0];
            let conv_output = graph.nodes[conv_node_id].outputs[0];
            let new_output = modifier.register_new_value(
                graph,
                format!("ConvBNFusion_Output_{:?}", conv_node_id),
                graph.get_resolved_tensor_type(conv_output).unwrap().clone(),
            );

            let conv_node = &graph.nodes[conv_node_id];
            modifier.register_new_node(
                graph,
                Node::create_node(
                    new_inputs,
                    vec![new_output],
                    format!("ConvBNFusion_{:?}", conv_node_id),
                    conv_node.op.clone(),
                ),
            );

            modifier.replace_input_value(graph, bn_output, new_output);
        }
    }
}

fn find_bn_user<T: GraphOp>(graph: &Graph, modifier: &T, conv_node_id: NodeId) -> Option<NodeId> {
    let conv_output = graph.nodes[conv_node_id].outputs[0];
    let users = modifier.used_node(conv_output)?;
    // Conv output must have exactly one user which is BatchNormalization
    if users.len() != 1 {
        return None;
    }
    let (bn_node_id, _) = users.iter().next()?;
    let bn_node = &graph.nodes[*bn_node_id];
    matches!(&bn_node.op, Operator::BatchNormalization(_)).then_some(*bn_node_id)
}
