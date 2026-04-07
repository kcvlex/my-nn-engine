use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::transform::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct ContiguousInsertion {}

impl<T: GraphOp> Pass<T> for ContiguousInsertion {
    fn summary(&self) -> &'static str {
        "Insert Contiguous nodes where needed"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter(|(_, node)| {
                matches!(
                    node.op,
                    Operator::BatchedGemm(_) |
                        Operator::Attention(_) |
                        Operator::AveragePool(_) |
                        Operator::Conv(_) |
                        Operator::MaxPool(_)
                )
            })
            .map(|(id, _)| id)
            .collect::<Vec<_>>();

        for id in ids {
            match graph.nodes[id].op {
                Operator::BatchedGemm(_) => self.handle_batched_gemm(graph, modifier, id),
                Operator::Attention(_) => self.handle_attention(graph, modifier, id),
                Operator::AveragePool(_) | Operator::Conv(_) | Operator::MaxPool(_) => {
                    self.handle_conv_pool(graph, modifier, id)
                }
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
    if let Some(users) = modifier.used_node(input) {
        for (user_id, _) in users.iter() {
            let user = &graph.nodes[*user_id];
            if matches!(user.op, Operator::Contiguous(_)) && user.inputs[0].unwrap() == input {
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
            inputs: vec![Some(input)],
            outputs: vec![new_value],
            op: Operator::Contiguous(Contiguous { ops: vec![] }),
            name: format!("{}_contiguous", name),
            meta: NodeMeta::default(),
        },
    );
    new_value
}

impl ContiguousInsertion {
    fn handle_batched_gemm<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T, id: NodeId) {
        let node = &graph.nodes[id];
        assert!(matches!(node.op, Operator::BatchedGemm(_)));

        let node_name = node.name.clone();
        for (i, input) in node.inputs.clone().iter().enumerate() {
            let input = input.unwrap();
            let new_value = find_or_create_contiguous(
                graph,
                modifier,
                input,
                &format!("{}_input_{}", node_name, i),
            );

            modifier.replace_input_value_if_without_typecheck(graph, input, new_value, |id2, _| {
                id == id2
            });
        }
    }

    fn handle_conv_pool<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T, id: NodeId) {
        let node = &graph.nodes[id];
        let node_name = node.name.clone();
        let input = node.inputs[0].unwrap();

        let new_value =
            find_or_create_contiguous(graph, modifier, input, &format!("{}_input_0", node_name));

        modifier
            .replace_input_value_if_without_typecheck(graph, input, new_value, |id2, _| id == id2);
    }

    fn handle_attention<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T, id: NodeId) {
        let node = &graph.nodes[id];
        let name = node.name.clone();
        assert!(matches!(node.op, Operator::Attention(_)));

        let mut inputs = vec![args::ATTENTION_Q, args::ATTENTION_K, args::ATTENTION_V];
        if node
            .inputs
            .get(args::ATTENTION_MASK)
            .and_then(|x| *x)
            .is_some()
        {
            inputs.push(args::ATTENTION_MASK);
        }
        for arg in inputs {
            let input = graph.nodes[id].inputs[arg].unwrap();
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
