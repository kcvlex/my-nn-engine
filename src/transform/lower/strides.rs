use std::collections::HashSet;
use std::collections::VecDeque;

use itertools::zip_eq;

use crate::graph::operator::*;
use crate::graph::Graph;
use crate::graph::Node;
use crate::graph::NodeId;
use crate::graph::NodeMeta;
use crate::graph::UnifyMode;
use crate::graph::ValueId;
use crate::tensor::types::ResolvedTensorType;
use crate::transform::shape::infer_node_output;
use crate::transform::GraphOp;
use crate::transform::Pass;
use crate::transform::Target;

pub struct AssignStrides {
    pub target: Target,
}

impl<T: GraphOp> Pass<T> for AssignStrides {
    fn summary(&self) -> &'static str {
        "Assign strides to all tensors"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let mut impl_ = AssignStridesImpl::new();
        impl_.run(graph, modifier).unwrap();
    }
}

struct AssignStridesImpl {
    computed_values: HashSet<ValueId>,
    visited: HashSet<NodeId>,
    queue: VecDeque<NodeId>,
}

impl AssignStridesImpl {
    fn new() -> Self {
        Self {
            computed_values: HashSet::new(),
            visited: HashSet::new(),
            queue: VecDeque::new(),
        }
    }

    fn run(&mut self, graph: &mut Graph, modifier: &mut impl GraphOp) -> Option<()> {
        for id in graph
            .nodes
            .iter()
            .filter(|(_, node)| node.is_dummy())
            .map(|(id, _)| id)
        {
            self.visited.insert(id);
        }

        let inputs = graph
            .inputs
            .iter()
            .map(|input| match graph.nodes[*input].op {
                Operator::Input(id) => id,
                Operator::SessionState(id) => id,
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        for id in inputs {
            self.complete_value(graph, modifier, id);
        }

        let initializer = graph.initializer_ids();
        for id in initializer {
            self.complete_value(graph, modifier, id);
        }

        let outputs = graph
            .outputs
            .iter()
            .map(|output| match graph.nodes[*output].op {
                Operator::Output(id) => (*output, id, graph.nodes[*output].name.clone()),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        for (node_id, value_id, name) in outputs {
            // TODO: Avoid inserting new node if the output is already contiguous
            let cont_node = self.insert_contiguous(graph, modifier, value_id, &name);
            let cont_value = graph.nodes[cont_node].outputs[0];
            modifier.replace_input_value_if_without_typecheck(
                graph,
                value_id,
                cont_value,
                |id, _| id == node_id,
            );
        }

        while let Some(id) = self.queue.pop_front() {
            let (id, resolved) = self.compute_strides(graph, modifier, id)?;
            let outputs = graph.nodes[id].outputs.clone();
            for (value_id, inferred) in zip_eq(outputs.iter(), resolved.into_iter()) {
                graph
                    .try_unify_type(*value_id, &inferred, UnifyMode::OverwriteStrides)
                    .ok()?;
            }
            for output in outputs {
                self.complete_value(graph, modifier, output);
            }
        }

        Some(())
    }

    fn complete_value(
        &mut self,
        graph: &mut Graph,
        modifier: &mut impl GraphOp,
        value_id: ValueId,
    ) {
        self.computed_values.insert(value_id);
        let used_nodes = modifier
            .used_node(value_id)
            .map(|set| set.into_iter().map(|(id, _)| *id).collect::<Vec<_>>())
            .unwrap_or_default();
        for id in used_nodes {
            if self.visited.contains(&id) {
                continue;
            }

            let mut ok = true;
            for input in graph.nodes[id].inputs.iter() {
                let Some(input) = input else { continue };
                if !self.computed_values.contains(input) {
                    ok = false;
                    break;
                }
            }

            if ok {
                self.queue.push_back(id);
                self.visited.insert(id);
            }
        }
    }

    fn insert_contiguous(
        &self,
        graph: &mut Graph,
        modifier: &mut impl GraphOp,
        input: ValueId,
        name: &str,
    ) -> NodeId {
        let new_shape = graph.get_resolved_tensor_type(input).unwrap().contiguous();
        let new_output =
            modifier.register_new_value(graph, format!("Contiguous_{}", name), new_shape);
        modifier.register_new_node(
            graph,
            Node {
                inputs: vec![Some(input)],
                outputs: vec![new_output],
                op: Operator::Contiguous(Contiguous { ops: vec![] }),
                name: format!("Contiguous_{}", name),
                meta: NodeMeta::default(),
            },
        )
    }

    fn compute_strides(
        &mut self,
        graph: &mut Graph,
        modifier: &mut impl GraphOp,
        node_id: NodeId,
    ) -> Option<(NodeId, Vec<ResolvedTensorType>)> {
        let new_node_id = match &graph.nodes[node_id].op {
            Operator::Reshape => {
                let input = graph.nodes[node_id].inputs[0].unwrap();
                let output = graph.nodes[node_id].outputs[0];
                let input_shape = graph.get_resolved_tensor_type(input)?.clone();
                let output_shape = graph.get_resolved_tensor_type(output)?.clone();

                if input_shape.try_reshape(&output_shape.dims).is_none() {
                    let name = graph.nodes[node_id].name.clone();
                    let cont_node = self.insert_contiguous(graph, modifier, input, &name);
                    let cont_value = graph.nodes[cont_node].outputs[0];
                    let new_output = modifier.register_new_value(
                        graph,
                        format!("Reshape_{}", name),
                        input_shape
                            .contiguous()
                            .try_reshape(&output_shape.dims)
                            .unwrap(),
                    );
                    let mut new_inputs = graph.nodes[node_id].inputs.clone();
                    new_inputs[0] = Some(cont_value);
                    let new_node = modifier.register_new_node(
                        graph,
                        Node {
                            inputs: new_inputs,
                            outputs: vec![new_output],
                            op: Operator::Reshape,
                            name: format!("Reshape_{}", name),
                            meta: NodeMeta::default(),
                        },
                    );
                    modifier.replace_input_value(graph, output, new_output);
                    new_node
                } else {
                    node_id
                }
            }

            _ => node_id,
        };

        let resolved = infer_node_output(graph, new_node_id, UnifyMode::CheckStrides)
            .expect("Invalid strides");
        Some((new_node_id, resolved))
    }
}
