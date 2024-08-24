use crate::model::{Graph, Node, NodeId};
use crate::operator::*;
use crate::optimize::optimizer;

#[derive(Default)]
pub struct MatMulAxTB {}

impl optimizer::Pass for MatMulAxTB {
    fn summary(&self) -> &'static str {
        "Convert MatMul(A, Transpose(B)) to MatMulRightTranposed(A, B)"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut optimizer::GraphModifier) {
        #[derive(Debug)]
        struct Pattern {
            transpose: NodeId,
            matmul: NodeId,
        }
        let mut res = Vec::new();
        for (id, node) in graph.nodes.iter() {
            let (matmul, rhs) = if matches!(node.op, Operator::MatMul) {
                (id, node.inputs[args::MATMUL_RHS])
            } else {
                continue;
            };

            let rhs = modifier.defined_node(rhs).unwrap().0;
            let transpose = match &graph.nodes[rhs].op {
                Operator::Transpose(perms) if perms.len() == 2 && perms.as_slice() == [1, 0] => rhs,
                _ => continue,
            };
            res.push(Pattern { transpose, matmul });
        }
        for (index, Pattern { transpose, matmul }) in res.into_iter().enumerate() {
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
            };
            modifier.register_new_node(graph, new_node);
            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}
