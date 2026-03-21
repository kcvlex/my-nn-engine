use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::transform::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct InsertContiguous {}

impl<T: GraphOp> Pass<T> for InsertContiguous {
    fn summary(&self) -> &'static str {
        "Insert Contiguous nodes where needed"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.op, Operator::MatMul | Operator::Attention(_)))
            .map(|(id, _)| id)
            .collect::<Vec<_>>();

        for id in ids {
            match graph.nodes[id].op {
                Operator::MatMul => self.handle_matmul(graph, modifier, id),
                Operator::Attention(_) => self.handle_attention(graph, modifier, id),
                _ => unreachable!(),
            }
        }
    }
}

fn find_or_create_contiguous<T: GraphOp>(
    graph: &mut Graph,
    modifier: &mut T,
    input: ValueId,
    name: &str,
) -> ValueId {
    // Check if a Contiguous node already exists for this input
    if let Some(users) = modifier.used_node(input) {
        for (user_id, _) in users.iter() {
            let user = &graph.nodes[*user_id];
            if matches!(user.op, Operator::Contiguous(_)) && user.inputs[0] == input {
                return user.outputs[0];
            }
        }
    }

    let input_type = graph.get_resolved_tensor_type(input).unwrap().clone();
    let new_value = modifier.register_new_value(
        graph,
        format!("{}_contiguous", name),
        input_type.contiguous(),
    );
    modifier.register_new_node(
        graph,
        Node {
            inputs: vec![input],
            outputs: vec![new_value],
            op: Operator::Contiguous(Contiguous { ops: vec![] }),
            name: format!("{}_contiguous", name),
            meta: NodeMeta::default(),
        },
    );
    new_value
}

impl InsertContiguous {
    fn handle_matmul<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T, id: NodeId) {
        let node = &graph.nodes[id];
        assert!(matches!(node.op, Operator::MatMul));

        let node_name = node.name.clone();
        for (i, input) in node.inputs.clone().iter().enumerate() {
            let input_type = graph.get_resolved_tensor_type(*input).unwrap().clone();
            if input_type.is_contiguous() {
                continue;
            }

            // TODO: Last two dimensions can be handled by transpose parameter of Gemm.
            let new_value = find_or_create_contiguous(
                graph,
                modifier,
                *input,
                &format!("{}_input_{}", node_name, i),
            );

            modifier.replace_input_value_if_without_typecheck(
                graph,
                *input,
                new_value,
                |id2, _| id == id2,
            );
        }
    }

    fn handle_attention<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T, id: NodeId) {
        let node = &graph.nodes[id];
        let name = node.name.clone();
        assert!(matches!(node.op, Operator::Attention(_)));

        let mut inputs = vec![args::ATTENTION_Q, args::ATTENTION_K, args::ATTENTION_V];
        if node.inputs.len() > args::ATTENTION_MASK {
            inputs.push(args::ATTENTION_MASK);
        }
        for arg in inputs {
            let input = graph.nodes[id].inputs[arg];
            let input_type = &graph.get_resolved_tensor_type(input).unwrap();
            if input_type.is_contiguous() {
                continue;
            }

            let new_value = find_or_create_contiguous(
                graph,
                modifier,
                input,
                &format!("{}_input_{}", name, arg),
            );

            modifier.replace_input_value_if_without_typecheck(graph, input, new_value, |id2, _| {
                id == id2
            });
        }
    }
}
