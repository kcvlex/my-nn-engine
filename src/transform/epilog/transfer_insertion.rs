use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeMeta;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct TransferInsertion {}

impl<T: GraphOp> Pass<T> for TransferInsertion {
    fn summary(&self) -> &'static str {
        "Insert Transfer(HostToDevice/DeviceToHost) for inputs/outputs"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let inputs: Vec<_> = graph
            .inputs
            .iter()
            .map(|&node_id| {
                let Operator::Input(value_id) = graph.nodes[node_id].op else {
                    unreachable!();
                };
                (node_id, value_id)
            })
            .collect();

        let input_host_values: Vec<ValueId> = inputs.iter().map(|(_, v)| *v).collect();

        let initializer_host_values: Vec<ValueId> = graph
            .initializer_ids()
            .into_iter()
            .filter(|&init_id| {
                graph.nodes.iter().any(|(_, node)| {
                    node.inputs.iter().enumerate().any(|(idx, inp)| {
                        *inp == Some(init_id) && !node.op.is_attribute_input(idx)
                    })
                })
            })
            .collect();

        let host_values: Vec<ValueId> = input_host_values
            .into_iter()
            .chain(initializer_host_values)
            .collect();

        for value_id in host_values {
            let ty = graph.get_resolved_tensor_type(value_id).unwrap().clone();
            let device_value = modifier.register_new_value(
                graph,
                format!("Transfer_H2D_{}", value_id.index()),
                ty,
            );
            let h2d_id = modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![Some(value_id)],
                    outputs: vec![device_value],
                    name: format!("Transfer_H2D_{}", value_id.index()),
                    op: Operator::Transfer(TransferKind::HostToDevice),
                    meta: NodeMeta::default(),
                },
            );
            modifier.replace_input_value_if_without_typecheck(
                graph,
                value_id,
                device_value,
                |id, _| id != h2d_id,
            );
        }

        let outputs: Vec<_> = graph
            .outputs
            .iter()
            .map(|&node_id| {
                let Operator::Output(value_id) = graph.nodes[node_id].op else {
                    unreachable!();
                };
                (node_id, value_id)
            })
            .collect();

        for (output_node_id, value_id) in outputs {
            let ty = graph.get_resolved_tensor_type(value_id).unwrap().clone();
            let host_value = modifier.register_new_value(
                graph,
                format!("Transfer_D2H_{}", value_id.index()),
                ty,
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![Some(value_id)],
                    outputs: vec![host_value],
                    name: format!("Transfer_D2H_{}", value_id.index()),
                    op: Operator::Transfer(TransferKind::DeviceToHost),
                    meta: NodeMeta::default(),
                },
            );
            graph.nodes[output_node_id].op = Operator::Output(host_value);
            graph.nodes[output_node_id].inputs = vec![Some(host_value)];
        }
    }
}
