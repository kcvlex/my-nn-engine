use std::collections::HashSet;

use crate::onnx::model::Graph;
use crate::onnx::model::NodeId;
use crate::onnx::operator::*;
use crate::transform::modify::GraphOp;
use crate::transform::utils::TransposeGenerator;
use crate::transform::Pass;

#[derive(Default)]
pub struct TransposeFusion {
    pub check_strides: bool,
}

impl<T: GraphOp> Pass<T> for TransposeFusion {
    fn summary(&self) -> &'static str {
        "Transpose Fusion"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let transposes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.op, Operator::Transpose(_)))
            .map(|(id, _)| id)
            .collect();

        let mut visited = HashSet::new();
        for transpose_id in transposes {
            if visited.contains(&transpose_id) {
                continue;
            }
            visited.insert(transpose_id);
            let input_value = graph.nodes[transpose_id].inputs[0];
            let (input, mut defs) = modifier.walk_chain_backward(graph, input_value, |node| {
                matches!(node.op, Operator::Transpose(_))
            });
            if defs.is_empty() {
                continue;
            }
            defs.reverse();
            defs.push(transpose_id);
            let old_output = graph.nodes[transpose_id].outputs[0];
            let mut perm = match graph.nodes[defs[0]].op {
                Operator::Transpose(ref p) => p.perm.clone().unwrap(),
                _ => unreachable!(),
            };
            for def in &defs[1..] {
                let p = match graph.nodes[*def].op {
                    Operator::Transpose(ref p) => p,
                    _ => unreachable!(),
                };
                if let Some(p) = &p.perm {
                    perm = p.iter().map(|&i| perm[i]).collect();
                }
            }

            let new_output = TransposeGenerator::default()
                .set_input(input)
                .set_perm(perm)
                .generate(graph, modifier)
                .unwrap();
            if self.check_strides {
                modifier.replace_input_value(graph, old_output, new_output)
            } else {
                modifier.replace_input_value_if_without_typecheck(
                    graph,
                    old_output,
                    new_output,
                    |_, _| true,
                );
            }
        }
    }
}
