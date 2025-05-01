use crate::onnx::model::{Graph, Node, NodeId, NodeMeta, ValueId};
use crate::onnx::operator::*;
use crate::onnx::utils;
use crate::transform::modify::GraphModifier;
use crate::transform::Pass;
use crate::utils::UnionFind;
use std::collections::HashMap;

#[derive(Default)]
pub struct FuseElementwiseOps {}

impl<T: GraphModifier> Pass<T> for FuseElementwiseOps {
    fn summary(&self) -> &'static str {
        "Fuse elementwise operators"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let fuser = FuseElementwiseOpsImpl::new(graph);
        fuser.run(graph, modifier);
    }
}

struct FuseElementwiseOpsImpl {
    node_ids: Vec<NodeId>,
    id2order: HashMap<NodeId, usize>,
}

impl FuseElementwiseOpsImpl {
    fn new(graph: &Graph) -> Self {
        let node_ids: Vec<_> = utils::simple_topological_order(graph)
            .into_iter()
            .filter(|id| {
                let node = &graph.nodes[*id];

                if !node.op.is_elementwise() {
                    return false;
                }

                // Reject if the node has multiple outputs.
                if node.outputs.len() != 1 {
                    return false;
                }

                // Reject if the node requires broadcast.
                if 1 < node.inputs.len() {
                    let shape0 = &graph
                        .get_resolved_tensor_type(node.inputs[0])
                        .as_ref()
                        .unwrap()
                        .dims;
                    for input in &node.inputs[1..] {
                        let shape = &graph
                            .get_resolved_tensor_type(*input)
                            .as_ref()
                            .unwrap()
                            .dims;
                        if shape != shape0 {
                            return false;
                        }
                    }
                }

                true
            })
            .collect();
        let id2order = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();

        Self { node_ids, id2order }
    }

    fn try_fuse(
        &self,
        id: NodeId,
        graph: &Graph,
        modifier: &impl GraphModifier,
        uf: &mut UnionFind,
    ) -> Option<()> {
        let ord_id = self.id2order.get(&id).copied()?;
        let mut cand = None;
        assert!(graph.nodes[id].outputs.len() == 1);
        let output = graph.nodes[id].outputs[0];
        for (user, _) in modifier.used_node(output)? {
            let user_id = self.id2order.get(user).copied()?;
            let repr = uf.representative(user_id);
            match cand {
                Some(v) if v == repr => (),
                None => cand = Some(repr),
                Some(_) => return None,
            }
        }

        if let Some(cand) = cand {
            uf.merge(cand, ord_id);
        }

        Some(())
    }

    // ids must be sorted.
    fn build_bundled_ops(
        &self,
        ord_ids: &[usize],
        graph: &Graph,
    ) -> (ElementwiseOps, Vec<ValueId>) {
        use std::collections::hash_map::Entry;

        let mut inputs = Vec::new();
        let mut input2idx = HashMap::new();
        let mut intermediates = HashMap::new();
        let mut ops = Vec::with_capacity(ord_ids.len());
        for ord in ord_ids {
            let node_id = self.node_ids[*ord];
            let node = &graph.nodes[node_id];

            let args: Vec<_> = node
                .inputs
                .iter()
                .map(|input| {
                    if let Some(inter) = intermediates.get(input) {
                        ElementwiseOpArg::NthResult(*inter)
                    } else {
                        let idx = match input2idx.entry(input) {
                            Entry::Occupied(entry) => *entry.get(),
                            Entry::Vacant(entry) => {
                                let idx = inputs.len();
                                inputs.push(*input);
                                entry.insert(idx);
                                idx
                            }
                        };
                        ElementwiseOpArg::Input(idx)
                    }
                })
                .collect();

            assert!(node.outputs.len() == 1);
            intermediates.insert(node.outputs[0], intermediates.len());

            ops.push((Box::new(node.op.clone()), args));
        }

        (ElementwiseOps { ops }, inputs)
    }

    fn run(&self, graph: &mut Graph, modifier: &mut impl GraphModifier) {
        let mut uf = UnionFind::new(self.node_ids.len());
        for node_id in self.node_ids.iter().rev() {
            self.try_fuse(*node_id, graph, modifier, &mut uf);
        }

        let mut groups = uf.groups();
        for group in groups.iter_mut().filter(|g| 1 < g.len()) {
            group.sort();
            let last_node_id = group.last().unwrap();
            let last_node_id = self.node_ids[*last_node_id];
            let (op, inputs) = self.build_bundled_ops(group, graph);
            let op = Operator::ElementwiseOps(op);
            let output = graph.nodes[last_node_id].outputs[0];

            let node_name = graph.nodes[last_node_id].name.clone();
            let value_name = graph.values[output].name.clone();
            let output_type = graph.get_resolved_tensor_type(output).unwrap();
            let new_output = modifier.register_new_value(
                graph,
                format!("Fused_{}", value_name),
                output_type.clone(),
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs,
                    outputs: vec![new_output],
                    op,
                    name: format!("Fused_{}", node_name),
                    meta: NodeMeta::default(),
                },
            );
            modifier.replace_input_value(graph, output, new_output);
        }
    }
}
