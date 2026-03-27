use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct SinkNHWC2NCHW {}

impl<T: GraphOp> Pass<T> for SinkNHWC2NCHW {
    fn summary(&self) -> &'static str {
        "Sink NHWC2NCHW past elementwise ops toward Conv/MaxPool/Im2Col"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let nhwc2nchw_ids: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.op, Operator::NHWC2NCHW))
            .map(|(id, _)| id)
            .collect();

        for id in nhwc2nchw_ids {
            self.sink_one(graph, modifier, id);
        }
    }
}

impl SinkNHWC2NCHW {
    fn sink_one<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T, nchw_id: NodeId) {
        let mut cur = nchw_id;
        loop {
            let output = graph.nodes[cur].outputs[0];
            let Some(users) = modifier.used_node(output) else {
                break;
            };
            if users.len() != 1 {
                break;
            }
            let &(user_id, user_input_idx) = &users[0];

            match &graph.nodes[user_id].op {
                Operator::NHWC2NCHW => panic!(),
                op if op.is_elementwise() => (),
                _ => break,
            }

            if graph.nodes[user_id].outputs.len() != 1 {
                break;
            }

            // Other inputs must be 4D initializers (constants)
            let other_inputs: Vec<_> = graph.nodes[user_id]
                .inputs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != user_input_idx)
                .map(|(i, v)| (i, *v))
                .collect();
            let can_sink = other_inputs.iter().all(|(_, v)| {
                graph.initializer.contains_key(v) &&
                    graph.get_resolved_tensor_type(*v).unwrap().dims.ndim() == 4
            });
            if !can_sink {
                break;
            }

            // Permute strides of constant 4D inputs: NCHW -> NHWC (perm [0,2,3,1])
            for &(_, v) in &other_inputs {
                let ty = graph
                    .get_resolved_tensor_type(v)
                    .unwrap()
                    .transpose(&[0, 2, 3, 1]);
                modifier.replace_tensor_type(graph, v, ty);
            }

            // Before: nhwc_input -> NHWC2NCHW(cur) -> Elem(user) -> ...
            // After:  nhwc_input -> Elem'          -> NHWC2NCHW' -> ...
            let nhwc_input = graph.nodes[cur].inputs[0];
            let nhwc_input_ty = graph.get_resolved_tensor_type(nhwc_input).unwrap().clone();
            let user_old_output = graph.nodes[user_id].outputs[0];
            let user_old_output_ty = graph
                .get_resolved_tensor_type(user_old_output)
                .unwrap()
                .clone();

            let mut new_elem_inputs = graph.nodes[user_id].inputs.clone();
            new_elem_inputs[user_input_idx] = nhwc_input;
            let new_elem_output = modifier.register_new_value(
                graph,
                format!("SinkNHWC2NCHW_Elem_{}", user_id.index()),
                nhwc_input_ty,
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: new_elem_inputs,
                    outputs: vec![new_elem_output],
                    name: format!("SinkNHWC2NCHW_Elem_{}", user_id.index()),
                    op: graph.nodes[user_id].op.clone(),
                    meta: NodeMeta::default(),
                },
            );

            let new_nchw_output = modifier.register_new_value(
                graph,
                format!("SinkNHWC2NCHW_NCHW_{}", user_id.index()),
                user_old_output_ty,
            );
            cur = modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![new_elem_output],
                    outputs: vec![new_nchw_output],
                    name: format!("SinkNHWC2NCHW_NCHW_{}", user_id.index()),
                    op: Operator::NHWC2NCHW,
                    meta: NodeMeta::default(),
                },
            );

            modifier.replace_input_value(graph, user_old_output, new_nchw_output);
        }
    }
}
