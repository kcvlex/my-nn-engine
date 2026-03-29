mod cleanup;
pub mod fold_cont;
mod lower_nhwc2nchw;

use itertools::Itertools;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::onnx::utils::simple_topological_order;
use crate::options::*;
use crate::transform::modify::GraphOp;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::Pass;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

#[derive(Default)]
pub struct Ops2Reinterpret {}

fn bundle_reshape_and_transpose<T: GraphOp>(
    graph: &Graph,
    node_id: NodeId,
    modifier: &T,
) -> (Reinterpret, ValueId) {
    let input_value = graph.nodes[node_id].inputs[0];
    let (source, chain) = modifier.walk_chain_backward(graph, input_value, |node| {
        matches!(node.op, Operator::Reshape | Operator::Transpose(_))
    });

    let all_nodes: Vec<NodeId> = chain
        .into_iter()
        .rev()
        .chain(std::iter::once(node_id))
        .collect();
    let ops = all_nodes
        .iter()
        .map(|&id| match &graph.nodes[id].op {
            Operator::Reshape => {
                let input_shape = graph
                    .get_resolved_tensor_type(graph.nodes[id].inputs[0])
                    .unwrap();
                let output_shape = graph
                    .get_resolved_tensor_type(graph.nodes[id].outputs[0])
                    .unwrap();
                ReinterpretType::Reshape {
                    before: input_shape.dims.iter().copied().collect(),
                    after: output_shape.dims.iter().copied().collect(),
                }
            }
            Operator::Transpose(perm) => ReinterpretType::Transpose(perm.clone()),
            _ => unreachable!(),
        })
        .collect();

    (Reinterpret { ops }, source)
}

impl<T: GraphOp> Pass<T> for Ops2Reinterpret {
    fn summary(&self) -> &'static str {
        "Convert Reshape/Transpose to Reinterpret"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = simple_topological_order(graph)
            .into_iter()
            .filter(|id| {
                matches!(
                    graph.nodes[*id].op,
                    Operator::Reshape | Operator::Transpose(_)
                )
            })
            .rev()
            .collect_vec();

        for id in ids.iter() {
            let old_output = graph.nodes[*id].outputs[0];
            let Some(to_bundle) = modifier.used_node(old_output) else {
                continue;
            };
            let to_bundle = to_bundle.iter().any(|(user_id, _)| {
                !matches!(
                    graph.nodes[*user_id].op,
                    Operator::Reshape | Operator::Transpose(_)
                )
            });
            if !to_bundle {
                continue;
            }

            let (re, input) = bundle_reshape_and_transpose(graph, *id, modifier);
            let new_output = modifier.register_new_value(
                graph,
                format!("Reinterpret_{:?}", old_output),
                graph.get_resolved_tensor_type(old_output).unwrap().clone(),
            );
            modifier.register_new_node(
                graph,
                Node::create_node(
                    vec![input],
                    vec![new_output],
                    format!("Reinterpret_{:?}", id),
                    Operator::Reinterpret(re),
                ),
            );
            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}

#[derive(Default)]
pub struct ElimCont {}
impl<T: GraphOp> Pass<T> for ElimCont {
    fn summary(&self) -> &'static str {
        "Eliminate unnecessary Contiguous nodes"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter(|(_, node)| match node.op {
                Operator::Contiguous(_) => {
                    let input_ty = graph.get_resolved_tensor_type(node.inputs[0]).unwrap();
                    let output_ty = graph.get_resolved_tensor_type(node.outputs[0]).unwrap();
                    input_ty.is_contiguous() && input_ty.dims == output_ty.dims
                }
                _ => false,
            })
            .map(|(id, _node)| id)
            .collect::<Vec<_>>();

        for id in ids.iter() {
            let node = &graph.nodes[*id];
            let input = node.inputs[0];
            let output = node.outputs[0];
            modifier.replace_input_value(graph, output, input);
        }
    }
}

pub fn create_epilog_passes(opt: &Options) -> SimplePassManager<SimpleGraphOp> {
    let mut manager = SimplePassManager::new("Epilog".to_string());
    manager.add_pass(Box::new(lower_nhwc2nchw::LowerNHWC2NCHW::default()));
    manager.add_pass(Box::new(Ops2Reinterpret::default()));
    if matches!(opt.target, Target::CUDA) {
        manager.add_pass(Box::new(ElimCont::default()));
    }
    manager.add_pass(Box::new(fold_cont::FoldContiguous::default()));
    manager.add_pass(Box::new(cleanup::CleanupTensors::default()));
    manager
}
