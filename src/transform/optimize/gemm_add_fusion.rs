use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct GemmAddFusion {}

impl<T: GraphOp> Pass<T> for GemmAddFusion {
    fn summary(&self) -> &'static str {
        "Fuse Gemm and Add into Gemm"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let res = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                let Operator::Gemm(gemm) = &node.op else {
                    return None;
                };

                if node.inputs.get(args::GEMM_C).is_some() {
                    assert!(gemm.beta != 0.0);
                    return None;
                }

                let output = node.outputs[0];
                let used = modifier.used_node(output)?;
                if used.len() != 1 {
                    return None;
                }

                let (used_id, _) = used[0];
                let used = &graph.nodes[used_id];

                if !matches!(used.op, Operator::Add) {
                    return None;
                }

                let lhs = used.inputs[0];
                let rhs = used.inputs[1];
                // TODO?: What happens when both two inputs of Add is the output of Gemm, i.e.,
                //   Gemm ------> Add
                //    \            ^
                //     \          /
                //      +--------+
                if lhs == rhs {
                    return None;
                }

                // For now, bail out if broadcast is necessary for the Add operation.
                if graph.get_resolved_tensor_type(lhs) != graph.get_resolved_tensor_type(rhs) {
                    return None;
                }

                Some((id, used_id))
            })
            .collect::<Vec<_>>();

        for (gemm_id, add_id) in res.into_iter() {
            let gemm_node = &graph.nodes[gemm_id];
            let add_node = &graph.nodes[add_id];
            let mut gemm = match &gemm_node.op {
                Operator::Gemm(g) => g.clone(),
                _ => unreachable!(),
            };
            assert!(gemm_node.inputs.len() == 2);
            assert!(gemm.beta == 0.0);
            assert!(matches!(add_node.op, Operator::Add));

            let gemm_output = gemm_node.outputs[0];
            let add_another = if add_node.inputs[0] == gemm_output {
                add_node.inputs[1]
            } else if add_node.inputs[1] == gemm_output {
                add_node.inputs[0]
            } else {
                unreachable!()
            };
            let add_output = add_node.outputs[0];

            let ty = graph.get_resolved_tensor_type(add_output).unwrap().clone();
            let inputs = vec![
                gemm_node.inputs[args::GEMM_A],
                gemm_node.inputs[args::GEMM_B],
                add_another,
            ];
            gemm.beta = 1.0;

            let new_output = modifier.register_new_value(
                graph,
                format!(
                    "GemmTransComposition_Output_{}_{}",
                    gemm_id.index(),
                    add_id.index()
                ),
                ty,
            );
            let new_node = Node {
                inputs,
                outputs: vec![new_output],
                name: format!("GemmAddFusion_{}_{}", gemm_id.index(), add_id.index()),
                op: Operator::Gemm(gemm),
                meta: NodeMeta::default(),
            };
            modifier.register_new_node(graph, new_node);
            modifier.replace_input_value(graph, add_output, new_output);
        }
    }
}
