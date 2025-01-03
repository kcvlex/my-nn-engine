use crate::model::{Graph, NodeId, Nodes, ValueId};
use crate::operator::Operator;
use crate::optimize::opinfo::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct AllocateInfo {
    pub ty: AllocateType,
    pub is_first_use: bool,
}

pub type ChunkId = usize;

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug)]
struct DependencyGraph {
    preds: HashMap<NodeId, Vec<NodeId>>,
    succs: HashMap<NodeId, Vec<NodeId>>,
    // value2defined: HashMap<ValueId, NodeId>,
}

impl DependencyGraph {
    fn new(graph: &Graph) -> Self {
        let mut preds: HashMap<_, Vec<_>> = HashMap::new();
        let mut succs: HashMap<_, Vec<_>> = HashMap::new();
        let mut value2defined = HashMap::new();
        for (node_id, node) in graph.nodes.iter() {
            if node.is_dummy() {
                continue;
            }
            for &output in node.outputs.iter() {
                value2defined.insert(output, node_id);
            }
        }
        for (node_id, node) in graph.nodes.iter() {
            if node.is_dummy() {
                continue;
            }
            preds.insert(node_id, Vec::new());
            for &input in node.inputs.iter() {
                if let Some(pred_id) = value2defined.get(&input) {
                    preds.entry(node_id).or_default().push(*pred_id);
                    succs.entry(*pred_id).or_default().push(node_id);
                }
            }
        }

        Self { preds, succs }
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
    liveness_counter: HashMap<NodeId, usize>,
    chunks: Chunks,
    allocations: HashMap<NodeId, AllocateType>,
}

impl<'graph> MemoryPlanner<'graph> {
    fn new(graph: &'graph Graph) -> Self {
        let deps = DependencyGraph::new(graph);
        let liveness_counter: HashMap<NodeId, usize> =
            deps.succs.iter().map(|(k, v)| (*k, v.len())).collect();

        MemoryPlanner {
            graph,
            deps,
            liveness_counter,
            chunks: Chunks::default(),
            allocations: HashMap::new(),
        }
    }

    // ref: https://arxiv.org/pdf/1604.06174
    fn run(&mut self, order: &[NodeId]) -> Vec<AllocateInfo> {
        for &node_id in order.iter() {
            let allocated = self.run_node(node_id);
            self.allocations.insert(node_id, allocated);
        }

        self.coalesce_input(order);
        self.coalesce_output(order);

        let mut last_user = vec![None; self.chunks.slot];
        let mut info_v = Vec::with_capacity(order.len());
        for &node_id in order.iter().rev() {
            let chunk = self.allocations.get(&node_id).unwrap();
            if let AllocateType::Chunk(chunk) = chunk {
                    last_user[*chunk] = Some(node_id);
            }
            info_v.push(AllocateInfo {
                ty: *chunk,
                is_first_use: false,
            });
        }

        info_v.reverse();

        for (i, node_id) in order.iter().enumerate() {
            if let AllocateType::Chunk(chunk) = info_v[i].ty {
                if last_user[chunk] == Some(*node_id) {
                    info_v[i].is_first_use = true;
                }
            }
        }

        info_v
    }

    fn coalesce_input(&mut self, order: &[NodeId]) {
        let mut input_set: HashMap<_, _> = self
            .graph
            .inputs
            .iter()
            .map(|v| match self.graph.nodes[*v].op {
                Operator::Input(v) => (v, v),
                _ => unreachable!(),
            })
            .chain(self.graph.initializer.keys().map(|k| (*k, *k)))
            .collect();

        for node_id in order.iter().copied() {
            if !self.graph.nodes[node_id].op.is_identity() {
                continue;
            }
            let input = self.graph.nodes[node_id].inputs[0];
            if let Some(v) = input_set.get(&input) {
                *self.allocations.get_mut(&node_id).unwrap() = AllocateType::Input(*v);
                input_set.insert(self.graph.nodes[node_id].outputs[0], *v);
            }
        }
    }

    fn coalesce_output(&mut self, order: &[NodeId]) {
        let mut output_set: HashMap<_, _> = self
            .graph
            .outputs
            .iter()
            .map(|v| match self.graph.nodes[*v].op {
                Operator::Output(v) => (v, v),
                _ => unreachable!(),
            })
            .collect();

        for node_id in order.iter().rev() {
            let output = self.graph.nodes[*node_id].outputs[0];
            if let Some(v) = output_set.get(&output) {
                *self.allocations.get_mut(node_id).unwrap() = AllocateType::Output(*v);

                // TODO: Support other patterns
                if self.graph.nodes[*node_id].op.is_identity() {
                    output_set.insert(self.graph.nodes[*node_id].inputs[0], *v);
                }
            }
        }
    }

    fn run_node(&mut self, node_id: NodeId) -> AllocateType {
        let in_place = self.try_in_place(node_id);
        let in_place_prev_id = in_place.map(|i| self.deps.preds[&node_id][i]);
        let chunk = match in_place_prev_id {
            Some(prev) => *self.allocations.get(&prev).unwrap(),
            None => AllocateType::Chunk(self.chunks.reuse_or_new()),
        };

        for &pred_id in self.deps.preds.get(&node_id).unwrap() {
            let counter = self.liveness_counter.get_mut(&pred_id).unwrap();
            *counter -= 1;
            if *counter == 0 {
                let used_now = in_place_prev_id.map(|x| x == pred_id).unwrap_or(false);
                if !used_now {
                    let chunk_id = self
                        .allocations
                        .get(&pred_id)
                        .and_then(|x| x.chunk_id())
                        .unwrap();
                    self.chunks.free(chunk_id);
                }
            }
        }

        chunk
    }

    fn try_in_place(&self, node_id: NodeId) -> Option<usize> {
        if !self.can_in_place(node_id) {
            return None;
        }

        for (i, &pred_id) in self.deps.preds.get(&node_id).unwrap().iter().enumerate() {
            // Last use
            if self.liveness_counter[&pred_id] == 1 {
                return Some(i);
            }
        }

        None
    }

    fn can_in_place(&self, node_id: NodeId) -> bool {
        let op = &self.graph.nodes[node_id].op;
        // op.is_elementwise() || op.is_identity() || matches!(op, Operator::Gemm(_))
        op.is_elementwise() || op.is_identity()
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

pub fn plan(graph: &Graph) -> Vec<(NodeId, AllocateInfo)> {
    let order = simple_topological_order(graph);
    let mut planner = MemoryPlanner::new(graph);
    planner
        .run(&order)
        .into_iter()
        .zip(order)
        .map(|(x, y)| (y, x))
        .collect()
}
