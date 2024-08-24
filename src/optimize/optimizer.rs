use crate::model::{Graph, Node, NodeId, ValueId, ValueInfo};
use crate::operator::Operator;
use crate::tensor::tensor::{ResolvedTensorType, TensorType};
use std::collections::{HashMap, HashSet};

pub trait Pass {
    fn summary(&self) -> &'static str;
    fn run(&self, graph: &mut Graph, modifier: &mut GraphModifier);
}

pub struct Optimizer {
    pub name: String,
    pub passes: Vec<Box<dyn Pass>>,
    //model: &mut Model,
}

impl Optimizer {
    pub fn new(name: String) -> Self {
        Optimizer {
            name,
            passes: Vec::new(),
        }
    }

    pub fn run(&self, graph: &mut Graph) {
        let mut modifier = GraphModifier::new(graph);
        for opt in self.passes.iter() {
            opt.run(graph, &mut modifier);
            modifier.delete_nodes(graph);
        }

        graph.delete_nodes(|&(id, _)| modifier.makred_as_deleted.contains(&id));
    }
}

#[derive(Default)]
pub struct GraphModifier {
    value2defined: HashMap<ValueId, (NodeId, usize)>,
    value2used: HashMap<ValueId, HashSet<(NodeId, usize)>>,

    to_delete_nodes: Vec<NodeId>,
    makred_as_deleted: HashSet<NodeId>,
    outdegrees: HashMap<NodeId, usize>,
}

impl GraphModifier {
    fn new(graph: &Graph) -> Self {
        let mut modifier = GraphModifier::default();
        for (node_id, node) in graph.nodes.iter() {
            for (index, &value) in node.inputs.iter().enumerate() {
                modifier
                    .value2used
                    .entry(value)
                    .or_insert_with(HashSet::new)
                    .insert((node_id, index));
            }
            for (index, &value) in node.outputs.iter().enumerate() {
                modifier.value2defined.insert(value, (node_id, index));
            }
        }
        for (value, (node, _)) in modifier.value2defined.iter() {
            *modifier.outdegrees.entry(*node).or_insert(0) +=
                modifier.value2used.get(value).map_or(0, |x| x.len());
        }
        modifier
    }

    pub fn register_new_value(
        &mut self,
        graph: &mut Graph,
        name: String,
        ty: ResolvedTensorType,
    ) -> ValueId {
        let ty = Some(TensorType::Resolved(ty));
        graph.values.alloc(ValueInfo { name, ty })
    }

    pub fn register_new_node(&mut self, graph: &mut Graph, v: Node) -> NodeId {
        let inputs = v.inputs.clone();
        let outputs = v.outputs.clone();
        let res = graph.nodes.alloc(v);
        for (index, value) in inputs.iter().enumerate() {
            self.value2used
                .entry(*value)
                .or_default()
                .insert((res, index));
            if let Some((defined, _)) = self.value2defined.get(value) {
                self.incr_outdegree(*defined, 1);
            }
        }
        for (index, value) in outputs.iter().enumerate() {
            self.value2defined.insert(*value, (res, index));
            self.incr_outdegree(res, self.value2used.get(value).map_or(0, |x| x.len()));
        }
        res
    }

    fn incr_outdegree(&mut self, node: NodeId, v: usize) {
        *self.outdegrees.entry(node).or_insert(0) += v;
    }

    fn decr_outdegree(&mut self, node: NodeId, v: usize) {
        if let Some(deg) = self.outdegrees.get_mut(&node) {
            *deg -= v;
            if *deg == 0 {
                self.to_delete_nodes.push(node);
            }
        }
    }

    pub fn replace_input_value(
        &mut self,
        graph: &mut Graph,
        old_value: ValueId,
        new_value: ValueId,
    ) {
        let old_defines = self.value2defined.remove(&old_value);
        let new_defines = self.value2defined.get(&new_value).cloned();

        if let Some(used) = self.value2used.remove(&old_value) {
            for &(node, index) in used.iter() {
                let node = &mut graph.nodes[node];
                if let Operator::Output(v) = node.op {
                    assert!(index == 0);
                    if v == old_value {
                        node.op = Operator::Output(new_value);
                    }
                }
                node.inputs[index] = new_value;
            }
            let len = used.len();
            self.value2used.insert(new_value, used);
            if let Some((defines, _)) = old_defines {
                self.decr_outdegree(defines, len);
            }
            if let Some((new_defines, _)) = new_defines {
                self.incr_outdegree(new_defines, len);
            }
        }
    }

    pub fn defined_node(&self, value: ValueId) -> Option<(NodeId, usize)> {
        self.value2defined.get(&value).cloned()
    }

    pub fn used_node(&self, value: ValueId) -> Option<&HashSet<(NodeId, usize)>> {
        self.value2used.get(&value)
    }

    fn delete_nodes_dfs(&mut self, node_id: NodeId, graph: &Graph) {
        if self.makred_as_deleted.contains(&node_id) {
            return;
        }
        if graph.nodes[node_id].is_dummy() {
            return;
        }
        self.makred_as_deleted.insert(node_id);
        for value in graph.nodes[node_id].inputs.iter() {
            if let Some((defined, _)) = self.value2defined.get(value).cloned() {
                let deg = self.outdegrees.get_mut(&defined).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    self.delete_nodes_dfs(defined, graph);
                }
            }
        }
    }

    fn delete_nodes(&mut self, graph: &mut Graph) {
        let mut vec = Vec::new();
        std::mem::swap(&mut vec, &mut self.to_delete_nodes);
        for node in vec {
            self.delete_nodes_dfs(node, graph);
        }
    }
}
