use crate::model::{Graph, Node, NodeId};
use crate::operator::*;
use crate::optimize::optimizer;

fn find_all_patterns(
    graph: &Graph,
    pattern: &[(Operator, usize)],
    modifier: &optimizer::GraphModifier,
) -> Vec<(NodeId, NodeId)> {
    let mut vec = Vec::new();
    for (index, _) in graph.nodes.iter().filter(|(_, node)| !node.mark_as_deleted) {
        let mut cur = Some(index);
        let mut prev = None;
        let mut found = true;
        for pat in pattern.iter().rev() {
            let index = if let Some(x) = cur {
                x
            } else {
                found = false;
                break;
            };
            let (op, value_idx) = pat;
            let node = &graph.nodes[index];
            if node.op != *op || node.mark_as_deleted {
                found = false;
                break;
            }

            prev = cur;
            cur = modifier.defined_node(node.inputs[*value_idx]).map(|x| x.0);
        }

        if found {
            vec.push((prev.unwrap(), index));
        }
    }
    vec
}

#[derive(Default)]
pub struct MatMulAxTB {}

impl optimizer::Pass for MatMulAxTB {
    fn summary(&self) -> &'static str {
        "Convert MatMul(A, Transpose(B)) to MatMulRightTranposed(A, B)"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut optimizer::GraphModifier) {
        let patterns = find_all_patterns(
            graph,
            &[
                (Operator::Transpose, args::TRANSPOSE_DATA),
                (Operator::MatMul, args::MATMUL_RHS),
            ],
            modifier,
        );

        for (index, (transpose, matmul)) in patterns.into_iter().enumerate() {
            let lhs = graph.nodes[matmul].inputs[args::MATMUL_LHS];
            let rhs = graph.nodes[transpose].inputs[args::TRANSPOSE_DATA];
            let old_output = graph.nodes[matmul].outputs[0];
            let ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
            let new_output = modifier.register_new_value(
                graph,
                format!("MatMulRightTransposed_Output_{index}"),
                ty,
            );
            let new_node = Node {
                inputs: vec![lhs, rhs],
                outputs: vec![new_output],
                name: format!("MatMulRightTransposed_{index}"),
                op: Operator::MatMulRightTransposed,
                mark_as_deleted: false,
            };
            modifier.register_new_node(graph, new_node);
            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}
