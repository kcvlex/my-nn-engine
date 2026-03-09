mod cleanup;

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
    assert!(matches!(
        graph.nodes[node_id].op,
        Operator::Reshape | Operator::Transpose(_)
    ));
    let mut ops = Vec::new();
    let mut cur = (node_id, 0);
    let mut input = graph.nodes[cur.0].inputs[0];
    loop {
        match &graph.nodes[cur.0].op {
            Operator::Reshape => {
                let output_shape = graph
                    .get_resolved_tensor_type(graph.nodes[cur.0].outputs[cur.1])
                    .unwrap();
                ops.push(ReinterpretType::Reshape(
                    output_shape.dims.iter().map(|d| *d as i64).collect(),
                ));
            }
            Operator::Transpose(perm) => ops.push(ReinterpretType::Transpose(perm.clone())),
            _ => break,
        }
        input = graph.nodes[cur.0].inputs[0];
        cur = match modifier.defined_node(input) {
            Some(v) => v,
            None => break,
        };
    }

    (Reinterpret { ops }, input)
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
                Operator::Contiguous => {
                    let input_shape = graph.get_resolved_tensor_type(node.inputs[0]).unwrap();
                    input_shape.is_contiguous()
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
    manager.add_pass(Box::new(Ops2Reinterpret::default()));
    if matches!(opt.target, Target::CUDA) {
        manager.add_pass(Box::new(ElimCont::default()));
    }
    manager.add_pass(Box::new(cleanup::CleanupTensors::default()));
    manager
}
