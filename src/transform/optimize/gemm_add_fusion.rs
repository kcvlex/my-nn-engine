use crate::graph::operator::*;
use crate::graph::Graph;
use crate::graph::Node;
use crate::graph::NodeMeta;
use crate::tensor::types::broadcast_shape;
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

                let lhs = used.inputs[0].unwrap();
                let rhs = used.inputs[1].unwrap();
                // TODO?: What happens when both two inputs of Add is the output of Gemm, i.e.,
                //   Gemm ------> Add
                //    \            ^
                //     \          /
                //      +--------+
                if lhs == rhs {
                    return None;
                }

                // The bias must be broadcastable to the Gemm output shape.
                let lhs_dims = &graph.get_resolved_tensor_type(lhs).unwrap().dims;
                let rhs_dims = &graph.get_resolved_tensor_type(rhs).unwrap().dims;
                if broadcast_shape(lhs_dims, rhs_dims).is_err() {
                    return None;
                }

                Some((id, used_id))
            })
            .collect::<Vec<_>>();

        for (gemm_id, add_id) in res.into_iter() {
            let mut gemm = match &graph.nodes[gemm_id].op {
                Operator::Gemm(g) => g.clone(),
                _ => unreachable!(),
            };
            assert!(graph.nodes[gemm_id].inputs.len() == 2);
            assert!(gemm.beta == 0.0);
            assert!(matches!(graph.nodes[add_id].op, Operator::Add));

            let gemm_output = graph.nodes[gemm_id].outputs[0];
            let add_another = if graph.nodes[add_id].inputs[0].unwrap() == gemm_output {
                graph.nodes[add_id].inputs[1].unwrap()
            } else if graph.nodes[add_id].inputs[1].unwrap() == gemm_output {
                graph.nodes[add_id].inputs[0].unwrap()
            } else {
                unreachable!()
            };
            let add_output = graph.nodes[add_id].outputs[0];

            let gemm_ty = graph.get_resolved_tensor_type(gemm_output).unwrap().clone();
            let bias_ty = graph.get_resolved_tensor_type(add_another).unwrap().clone();

            // If broadcast is needed, insert a Contiguous node to expand the bias
            // to match the Gemm output shape. ConstantFold will later fold this
            // into a materialized initializer if the bias is constant.
            let c_value = if gemm_ty.dims != bias_ty.dims {
                let broadcast_op = ReinterpretType::Broadcast {
                    before: bias_ty.dims.iter().copied().collect(),
                    after: gemm_ty.dims.iter().copied().collect(),
                };
                let cont_output = modifier.register_new_value(
                    graph,
                    format!(
                        "GemmAddFusion_BiasCont_{}_{}",
                        gemm_id.index(),
                        add_id.index()
                    ),
                    gemm_ty,
                );
                modifier.register_new_node(
                    graph,
                    Node {
                        inputs: vec![Some(add_another)],
                        outputs: vec![cont_output],
                        name: format!(
                            "GemmAddFusion_BiasCont_{}_{}",
                            gemm_id.index(),
                            add_id.index()
                        ),
                        op: Operator::Contiguous(Contiguous {
                            ops: vec![broadcast_op],
                        }),
                        meta: NodeMeta::default(),
                    },
                );
                cont_output
            } else {
                add_another
            };

            let ty = graph.get_resolved_tensor_type(add_output).unwrap().clone();
            let inputs = vec![
                graph.nodes[gemm_id].inputs[args::GEMM_A],
                graph.nodes[gemm_id].inputs[args::GEMM_B],
                Some(c_value),
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
