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
            let mut defs = follow_defs(transpose_id, graph, modifier);
            if defs.is_empty() {
                continue;
            }
            defs.reverse();
            defs.push(transpose_id);
            let input = graph.nodes[defs[0]].inputs[0];
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

fn follow_defs<T: GraphOp>(id: NodeId, graph: &Graph, modifier: &T) -> Vec<NodeId> {
    let mut res = Vec::new();
    let mut cur = id;
    loop {
        let Some((def, _)) = modifier.defined_node(graph.nodes[cur].inputs[0]) else {
            break;
        };
        if !matches!(graph.nodes[def].op, Operator::Transpose(_)) {
            break;
        }
        cur = def;
        res.push(def);
    }
    res
}
