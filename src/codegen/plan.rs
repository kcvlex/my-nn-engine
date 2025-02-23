use crate::onnx::model::{Graph, NodeId, Nodes, ValueId};
use crate::onnx::operator::Operator;
use indexmap::{IndexMap, IndexSet};
use std::collections::{HashMap, HashSet};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct AllocateInfo {
    pub value_id: ValueId,
    pub ty: AllocateType,
    pub is_first_use: bool,
}

pub type ChunkId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocateType {
    Chunk(ChunkId),
    Input(ValueId),
    Output(ValueId),
}

impl AllocateType {
    pub fn chunk_id(&self) -> Option<ChunkId> {
        match self {
            AllocateType::Chunk(id) => Some(*id),
            AllocateType::Input(_) | AllocateType::Output(_) => None,
        }
    }
}

// TODO: Make the order deterministic
#[derive(Debug)]
struct DependencyGraph {
    value2defined: IndexMap<ValueId, NodeId>,
    value2used: IndexMap<ValueId, IndexSet<NodeId>>,
    inputs: HashSet<ValueId>,
    outputs: HashSet<ValueId>,
}

impl DependencyGraph {
    fn new(graph: &Graph) -> Self {
        let mut value2defined = IndexMap::new();
        let mut value2used = IndexMap::new();
        let inputs = graph
            .inputs
            .iter()
            .map(|x| match graph.nodes[*x].op {
                Operator::Input(v) => v,
                _ => unreachable!(),
            })
            .chain(graph.initializer.keys().copied())
            .collect::<HashSet<_>>();
        let outputs = graph
            .outputs
            .iter()
            .map(|x| match graph.nodes[*x].op {
                Operator::Output(v) => v,
                _ => unreachable!(),
            })
            .collect::<HashSet<_>>();
        let ignore = |x| inputs.contains(x) || outputs.contains(x);
        for (node_id, node) in graph.nodes.iter() {
            if node.is_dummy() {
                continue;
            }
            for &input in node.inputs.iter().filter(|x| !ignore(x)) {
                value2used
                    .entry(input)
                    .or_insert_with(IndexSet::new)
                    .insert(node_id);
            }
            for &output in node.outputs.iter().filter(|x| !ignore(x)) {
                value2defined.insert(output, node_id);
            }
        }
        Self {
            value2defined,
            value2used,
            inputs,
            outputs,
        }
    }
}

#[derive(Default)]
struct Chunks {
    free_list: Vec<ChunkId>,
    slot: usize,
}

impl Chunks {
    fn allocate(&mut self) -> ChunkId {
        let id = self.slot;
        self.slot += 1;
        id
    }

    fn free(&mut self, id: ChunkId) {
        self.free_list.push(id);
    }

    fn reuse_or_new(&mut self) -> ChunkId {
        self.free_list.pop().unwrap_or_else(|| self.allocate())
    }
}

struct MemoryPlanner<'graph> {
    graph: &'graph Graph,
    deps: DependencyGraph,
    liveness_counter: IndexMap<ChunkId, usize>,
    chunks: Chunks,
    allocations: HashMap<ValueId, AllocateType>,
}

impl<'graph> MemoryPlanner<'graph> {
    fn new(graph: &'graph Graph) -> Self {
        let deps = DependencyGraph::new(graph);

        MemoryPlanner {
            graph,
            deps,
            liveness_counter: IndexMap::new(),
            chunks: Chunks::default(),
            allocations: HashMap::new(),
        }
    }

    // ref: https://arxiv.org/pdf/1604.06174
    fn run(&mut self, order: &[NodeId]) -> Vec<Vec<AllocateInfo>> {
        for input in self.deps.inputs.iter() {
            self.allocations.insert(*input, AllocateType::Input(*input));
        }
        for output in self.deps.outputs.iter() {
            self.allocations
                .insert(*output, AllocateType::Output(*output));
        }

        for &node_id in order.iter() {
            let allocated = self.run_node(node_id);
            for (value_id, allocated) in allocated {
                self.allocations.insert(value_id, allocated);
            }
        }

        self.coalesce_output(order);

        let mut last_user = vec![None; self.chunks.slot];
        let mut info_v = Vec::with_capacity(order.len());
        for &node_id in order.iter().rev() {
            let tmp = self.graph.nodes[node_id]
                .outputs
                .iter()
                .map(|output| {
                    let chunk = self.allocations.get(output).unwrap();
                    if let AllocateType::Chunk(chunk) = chunk {
                        last_user[*chunk] = Some(*output);
                    }
                    AllocateInfo {
                        value_id: *output,
                        ty: *chunk,
                        is_first_use: false,
                    }
                })
                .collect::<Vec<_>>();
            info_v.push(tmp);
        }

        info_v.reverse();

        for vec in info_v.iter_mut() {
            for info in vec.iter_mut() {
                if let AllocateType::Chunk(chunk) = info.ty {
                    if last_user[chunk] == Some(info.value_id) {
                        info.is_first_use = true;
                    }
                }
            }
        }
        info_v
    }

    fn coalesce_output(&mut self, order: &[NodeId]) {
        let mut output_set: HashMap<_, _> = self.deps.outputs.iter().map(|v| (*v, *v)).collect();

        for node_id in order.iter().rev() {
            for (i, output) in self.graph.nodes[*node_id].outputs.iter().enumerate() {
                if let Some(v) = output_set.get(output) {
                    *self.allocations.get_mut(output).unwrap() = AllocateType::Output(*v);

                    // TODO: Support other patterns
                    if i == 0 && matches!(self.graph.nodes[*node_id].op, Operator::Identity) {
                        output_set.insert(self.graph.nodes[*node_id].inputs[0], *v);
                    }
                }
            }
        }
    }

    fn run_node(&mut self, node_id: NodeId) -> Vec<(ValueId, AllocateType)> {
        let mut res = Vec::new();
        let node = &self.graph.nodes[node_id];
        for output in node.outputs.iter() {
            let chunk = match node.op {
                // Split is a special case.
                // TODO: When the input is `Input` or initializer.
                Operator::Split(_) => {
                    let res = *self.allocations.get(&node.inputs[0]).unwrap();
                    // assert!(matches!(res, AllocateType::Chunk(_)));
                    res
                }
                _ => {
                    if self.deps.outputs.contains(output) {
                        AllocateType::Output(*output)
                    } else {
                        match self.try_in_place(*output) {
                            Some(prev) => *self.allocations.get(&prev).unwrap(),
                            None => AllocateType::Chunk(self.chunks.reuse_or_new()),
                        }
                    }
                }
            };
            res.push((*output, chunk));
            if let AllocateType::Chunk(chunk) = chunk {
                // It is possible that the value is not used by any other nodes, e.g., the output
                // of splitted one.
                if let Some(used) = self.deps.value2used.get(output) {
                    *self.liveness_counter.entry(chunk).or_insert(0) += used.len();
                }
            }
        }

        for input in self.graph.nodes[node_id]
            .inputs
            .iter()
            .filter(|x| !self.deps.inputs.contains(x))
        {
            let chunk_id = match self.allocations.get(input).and_then(|x| x.chunk_id()) {
                Some(v) => v,
                None => continue,
            };
            let counter = self.liveness_counter.get_mut(&chunk_id).unwrap();
            *counter -= 1;
            if *counter == 0 {
                self.chunks.free(chunk_id);
            }
        }
        res
    }

    // TODO: Identity assumes that the computation MUST be in-place.
    // If the source is not contiguous, we need to allocate a new chunk.
    fn try_in_place(&self, value_id: ValueId) -> Option<ValueId> {
        if !self.can_in_place(value_id) {
            return None;
        }

        let node_id = self.deps.value2defined[&value_id];
        for input in self.graph.nodes[node_id].inputs.iter() {
            let is_input = match self.allocations.get(input) {
                Some(AllocateType::Input(_)) => true,
                Some(AllocateType::Chunk(chunk_id)) => {
                    if self.liveness_counter[chunk_id] != 1 {
                        continue;
                    }
                    false
                }
                Some(_) | None => continue,
            };

            // TODO: correct?
            if matches!(self.graph.nodes[node_id].op, Operator::Identity) {
                return Some(*input);
            }

            if is_input {
                continue;
            }

            if self
                .graph
                .get_resolved_tensor_type(*input)
                .as_ref()
                .unwrap() ==
                self.graph
                    .get_resolved_tensor_type(value_id)
                    .as_ref()
                    .unwrap()
            {
                return Some(*input);
            }
        }

        None
    }

    fn can_in_place(&self, value_id: ValueId) -> bool {
        let node_id = self.deps.value2defined[&value_id];
        let op = &self.graph.nodes[node_id].op;
        matches!(op, Operator::Identity) || op.is_elementwise()
    }
}

fn simple_topological_order(graph: &Graph) -> Vec<NodeId> {
    fn dfs(
        node: NodeId,
        res: &mut Vec<NodeId>,
        visited: &mut HashSet<NodeId>,
        adj: &HashMap<NodeId, HashSet<NodeId>>,
        nodes: &Nodes,
    ) {
        if visited.contains(&node) {
            return;
        }
        visited.insert(node);
        if let Some(neighbors) = adj.get(&node) {
            for &next in neighbors.iter() {
                dfs(next, res, visited, adj, nodes);
            }
        }
        if !nodes[node].is_dummy() {
            res.push(node);
        }
    }

    let mut res = Vec::new();
    let mut visited = HashSet::new();
    let mut defined = HashMap::new();
    for (id, node) in graph.nodes.iter() {
        for value in node.outputs.iter() {
            defined.insert(value, id);
        }
    }
    let mut adj = HashMap::new();
    for (id, node) in graph.nodes.iter() {
        for value in node.inputs.iter() {
            if let Some(defines) = defined.get(value) {
                adj.entry(*defines).or_insert(HashSet::new()).insert(id);
            } else {
                assert!(graph.initializer.contains_key(value));
            }
        }
    }

    for (id, _) in graph.nodes.iter().filter(|(_, node)| !node.is_dummy()) {
        if !visited.contains(&id) {
            dfs(id, &mut res, &mut visited, &adj, &graph.nodes);
        }
    }
    res.reverse();
    res
}

pub fn plan(graph: &Graph) -> Vec<(NodeId, Vec<AllocateInfo>)> {
    let order = simple_topological_order(graph);
    MemoryPlanner::new(graph)
        .run(&order)
        .into_iter()
        .zip(order)
        .map(|(x, y)| (y, x))
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::onnx::load::*;
    use crate::onnx::model::Model;
    use crate::transform::modify::SimpleGraphModifier;
    use crate::transform::{create_infer_passes, PassManager};
    use std::io::{Error, Result};
    use std::path::PathBuf;

    fn load_model(path: &str) -> Result<Model> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/test/plan")
            .join(path);
        let mut model =
            Model::load_from_path(path).map_err(|e| Error::other(format!("{:?}", e)))?;
        let passes = create_infer_passes();
        let mut modifier = SimpleGraphModifier::new(&model.graph);
        passes.run(&mut model.graph, &mut modifier);
        Ok(model)
    }

    fn to_node_order(graph: &Graph, order: &[&str]) -> Result<Vec<NodeId>> {
        let mut name2id: HashMap<&str, _> = HashMap::new();
        for (id, node) in graph.nodes.iter() {
            if name2id.insert(&node.name, id).is_some() {
                return Err(Error::other("duplicated node name"));
            }
        }

        let mut res = Vec::with_capacity(order.len());
        for name in order.iter() {
            let id = name2id
                .get(name)
                .ok_or_else(|| Error::other("node not found"))?;
            res.push(*id);
        }
        Ok(res)
    }

    // TODO: Insert Contiguous operator
    // Graph:
    //
    //               +-- 1.Sigmoid -- 3.Pool --+
    //              /                           \
    // Input -- 0.Sigmoid                      4.Add -- 5.Transpose -- Output
    //              \                           /
    //               +-- 2.Pool ---------------+
    //
    #[test]
    fn diamond() -> Result<()> {
        #[derive(Debug, PartialEq)]
        struct Test {
            ty: AllocateType,
            is_first_use: bool,
        }

        let model = load_model("diamond.onnx")?;
        let order = [
            "/layer1/Sigmoid",
            "/layer2/layer2.0/Sigmoid",
            "/layer3/MaxPool",
            "/layer2/layer2.1/MaxPool",
            "/Add",
            "/Transpose",
            "Contiguous_Output_7",
        ];
        let order = to_node_order(&model.graph, &order)?;
        let mem = MemoryPlanner::new(&model.graph)
            .run(&order)
            .iter()
            .flatten()
            .map(
                |AllocateInfo {
                     ty, is_first_use, ..
                 }| Test {
                    ty: *ty,
                    is_first_use: *is_first_use,
                },
            )
            .collect::<Vec<_>>();
        let output_id = match model.graph.nodes[model.graph.outputs[0]].op {
            Operator::Output(id) => id,
            _ => unreachable!(),
        };
        assert_eq!(
            mem,
            &[
                Test {
                    ty: AllocateType::Chunk(0),
                    is_first_use: true
                },
                Test {
                    ty: AllocateType::Chunk(1),
                    is_first_use: true
                },
                Test {
                    ty: AllocateType::Chunk(2),
                    is_first_use: true
                },
                Test {
                    ty: AllocateType::Chunk(0),
                    is_first_use: false
                },
                Test {
                    ty: AllocateType::Chunk(0),
                    is_first_use: false
                },
                Test {
                    ty: AllocateType::Chunk(2),
                    is_first_use: false
                },
                Test {
                    ty: AllocateType::Output(output_id),
                    is_first_use: false
                },
            ]
        );
        Ok(())
    }
}
