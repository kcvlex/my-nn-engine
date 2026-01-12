use std::collections::HashMap;

use itertools::zip_eq;

use crate::onnx::model::*;
use crate::onnx::utils;
use crate::schedule::*;
use crate::transform::modify::GraphOp;
use crate::utils::UnionFind;

struct OrderedNodeId {
    ordered: Vec<NodeId>,
    id2order: HashMap<NodeId, usize>,
}

impl OrderedNodeId {
    fn new(ids: Vec<NodeId>) -> Self {
        let id2order = ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        Self {
            ordered: ids,
            id2order,
        }
    }
}

struct KernelsBuilder {
    nodes: OrderedNodeId,
    elementwise_nodes: OrderedNodeId,
}

impl KernelsBuilder {
    fn new(graph: &Graph) -> Self {
        let nodes = utils::simple_topological_order(graph)
            .into_iter()
            .filter(|id| !graph.nodes[*id].is_dummy())
            .collect::<Vec<_>>();
        let nodes = OrderedNodeId::new(nodes);
        let elementwise_nodes: Vec<_> = nodes
            .ordered
            .iter()
            .copied()
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
                // if 1 < node.inputs.len() {
                //     let shape0 = &graph
                //         .get_resolved_tensor_type(node.inputs[0])
                //         .as_ref()
                //         .unwrap()
                //         .dims;
                //     for input in &node.inputs[1..] {
                //         let shape = &graph
                //             .get_resolved_tensor_type(*input)
                //             .as_ref()
                //             .unwrap()
                //             .dims;
                //         if shape != shape0 {
                //             return false;
                //         }
                //     }
                // }

                true
            })
            .collect();
        let elementwise_nodes = OrderedNodeId::new(elementwise_nodes);

        Self {
            nodes,
            elementwise_nodes,
        }
    }

    fn try_fuse(
        &self,
        id: NodeId,
        graph: &Graph,
        graph_op: &impl GraphOp,
        uf: &mut UnionFind,
    ) -> Option<()> {
        let ord_id = self.elementwise_nodes.id2order.get(&id).copied()?;
        let mut cand = None;
        assert!(graph.nodes[id].outputs.len() == 1);
        let output = graph.nodes[id].outputs[0];
        for (user, _) in graph_op.used_node(output)? {
            let user_id = self.elementwise_nodes.id2order.get(user).copied()?;
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
    fn build_bundled_ops(&self, ord_ids: &[usize], graph: &Graph) -> (ElementWises, Vec<ValueId>) {
        use std::collections::hash_map::Entry;

        assert!(ord_ids.is_sorted());

        let mut inputs = Vec::new();
        let mut input2idx = HashMap::new();
        let mut intermediates = HashMap::new();
        let mut ops = Vec::with_capacity(ord_ids.len());
        for ord in ord_ids {
            let node_id = self.elementwise_nodes.ordered[*ord];
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

            ops.push((node.op.clone(), args));
        }

        (ElementWises { ops }, inputs)
    }

    fn elementwise_order2order(&self, ord: usize) -> usize {
        let id = self.elementwise_nodes.ordered[ord];
        self.nodes.id2order[&id]
    }

    fn run(&self, graph: &Graph, graph_op: &impl GraphOp) -> Kernels {
        let mut uf = UnionFind::new(self.elementwise_nodes.ordered.len());
        for node_id in self.elementwise_nodes.ordered.iter().rev() {
            self.try_fuse(*node_id, graph, graph_op, &mut uf);
        }

        #[derive(Clone, Copy, Debug)]
        enum KernelTag {
            Ignore,
            Single,
            ElementwiseLast(usize),
        }

        let mut kernel_tags = vec![KernelTag::Single; self.nodes.ordered.len()];
        let groups = uf
            .groups()
            .into_iter()
            .filter_map(|mut g| {
                if 1 < g.len() {
                    g.sort();
                    Some(g)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for (group_id, group) in groups.iter().enumerate() {
            for g in group.iter() {
                let node_id = self.elementwise_order2order(*g);
                kernel_tags[node_id] = KernelTag::Ignore;
            }
            let last_node_id = self.elementwise_order2order(*group.last().unwrap());
            kernel_tags[last_node_id] = KernelTag::ElementwiseLast(group_id);
        }

        let mut kernels = Kernels::default();
        for (tag, node_id) in zip_eq(kernel_tags.iter(), self.nodes.ordered.iter().copied()) {
            let kernel = match tag {
                KernelTag::Ignore => continue,
                KernelTag::Single => {
                    let node = &graph.nodes[node_id];
                    let op = node.op.clone();
                    let body = if op.is_elementwise() {
                        let ops = vec![(
                            op,
                            (0..node.inputs.len())
                                .map(ElementwiseOpArg::Input)
                                .collect(),
                        )];
                        KernelBody::ElementWises(ElementWises { ops })
                    } else {
                        let body = SingleKernel {
                            op: node.op.clone(),
                        };
                        KernelBody::SingleKernel(body)
                    };
                    Kernel {
                        inputs: graph.nodes[node_id].inputs.clone(),
                        outputs: graph.nodes[node_id].outputs.clone(),
                        body,
                        name: node.name.clone(),
                        mem_alloc: None,
                        omp_info: OmpInfo::default(),
                    }
                }
                KernelTag::ElementwiseLast(group_id) => {
                    let group = &groups[*group_id];
                    let last_node = &graph.nodes[node_id];
                    let outputs = last_node.outputs.clone();
                    let (body, inputs) = self.build_bundled_ops(group, graph);
                    let body = KernelBody::ElementWises(body);
                    let name = format!("Elementwises_{}", last_node.name);
                    Kernel {
                        inputs,
                        outputs,
                        body,
                        name,
                        mem_alloc: None,
                        omp_info: OmpInfo::default(),
                    }
                }
            };
            kernels.0.alloc(kernel);
        }
        kernels
    }
}

pub fn build_kernels(graph: &Graph, graph_op: &impl GraphOp) -> Kernels {
    KernelsBuilder::new(graph).run(graph, graph_op)
}
