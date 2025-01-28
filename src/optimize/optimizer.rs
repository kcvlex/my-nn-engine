use crate::onnx::model::{Graph, Node, NodeId, ValueId, ValueInfo};
use crate::onnx::operator::Operator;
use crate::tensor::types::{ResolvedTensorType, TensorType};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

pub trait GraphModifier {
    fn new(graph: &Graph) -> Self;

    fn register_new_value(
        &mut self,
        graph: &mut Graph,
        name: String,
        ty: ResolvedTensorType,
    ) -> ValueId {
        let ty = Some(TensorType::Resolved(ty));
        graph.values.alloc(ValueInfo { name, ty })
    }

    fn register_new_node(&mut self, graph: &mut Graph, v: Node) -> NodeId;

    fn defined_node(&self, value: ValueId) -> Option<(NodeId, usize)>;

    fn used_node(&self, value: ValueId) -> Option<&HashSet<(NodeId, usize)>>;

    fn replace_input_value_if<P>(
        &mut self,
        graph: &mut Graph,
        old_value: ValueId,
        new_value: ValueId,
        pred: P,
    ) where
        P: Fn(NodeId, &Node) -> bool;

    fn replace_input_value(&mut self, graph: &mut Graph, old_value: ValueId, new_value: ValueId) {
        self.replace_input_value_if(graph, old_value, new_value, |_, _| true);
    }

    fn replace_op(&mut self, graph: &mut Graph, node_id: NodeId, op: Operator) {
        graph.nodes[node_id].op = op;
    }
}

trait NodeDelete {
    fn update_deleted_nodes(&mut self, graph: &mut Graph);
}

pub trait Pass<T: GraphModifier> {
    fn summary(&self) -> &'static str;
    fn run(&self, graph: &mut Graph, modifier: &mut T);
}

#[allow(private_bounds)]
pub struct Optimizer<T: GraphModifier + NodeDelete> {
    pub name: String,
    pub passes: Vec<Box<dyn Pass<T>>>,
    //model: &mut Model,
}

#[allow(private_bounds)]
impl<T: GraphModifier + NodeDelete> Optimizer<T> {
    pub fn new(name: String) -> Self {
        Optimizer {
            name,
            passes: Vec::new(),
        }
    }

    pub fn run(&self, graph: &mut Graph) {
        let mut modifier = T::new(graph);
        for opt in self.passes.iter() {
            println!("Running pass: {}", opt.summary());
            opt.run(graph, &mut modifier);
            modifier.update_deleted_nodes(graph);
        }

        graph.delete_nodes();
    }
}

type Value2Defined = HashMap<ValueId, (NodeId, usize)>;
type Value2Used = HashMap<ValueId, HashSet<(NodeId, usize)>>;

fn calc_value2xx(graph: &Graph) -> (Value2Defined, Value2Used) {
    let mut value2defined = HashMap::new();
    let mut value2used = HashMap::new();
    for (node_id, node) in graph.nodes.iter() {
        for (index, &value) in node.inputs.iter().enumerate() {
            value2used
                .entry(value)
                .or_insert_with(HashSet::new)
                .insert((node_id, index));
        }
        for (index, &value) in node.outputs.iter().enumerate() {
            value2defined.insert(value, (node_id, index));
        }
    }
    (value2defined, value2used)
}

pub struct SimpleGraphModifier {
    value2defined: HashMap<ValueId, (NodeId, usize)>,
    value2used: HashMap<ValueId, HashSet<(NodeId, usize)>>,
}

impl GraphModifier for SimpleGraphModifier {
    fn new(graph: &Graph) -> Self {
        let (value2defined, value2used) = calc_value2xx(graph);
        SimpleGraphModifier {
            value2defined,
            value2used,
        }
    }

    fn register_new_node(&mut self, graph: &mut Graph, v: Node) -> NodeId {
        let inputs = v.inputs.clone();
        let outputs = v.outputs.clone();
        let res = graph.nodes.alloc(v);
        for (index, value) in inputs.iter().enumerate() {
            self.value2used
                .entry(*value)
                .or_default()
                .insert((res, index));
        }
        for (index, value) in outputs.iter().enumerate() {
            self.value2defined.insert(*value, (res, index));
        }
        res
    }

    fn replace_input_value_if<P>(
        &mut self,
        graph: &mut Graph,
        old_value: ValueId,
        new_value: ValueId,
        pred: P,
    ) where
        P: Fn(NodeId, &Node) -> bool,
    {
        let mut changed = HashSet::new();
        if let Some(used) = self.value2used.get(&old_value) {
            for &(node, index) in used.iter() {
                if pred(node, &graph.nodes[node]) {
                    changed.insert((node, index));
                }
            }
        }

        if changed.is_empty() {
            return;
        }

        if let Entry::Occupied(mut old) = self.value2used.entry(old_value) {
            for v in changed.iter() {
                old.get_mut().remove(v);
            }
            if old.get().is_empty() {
                old.remove();
            }
        }

        for (node, index) in changed.iter() {
            let node = &mut graph.nodes[*node];
            if let Operator::Output(v) = node.op {
                assert!(v == old_value);
                node.op = Operator::Output(new_value);
            }
            node.inputs[*index] = new_value;
        }

        self.value2used
            .entry(new_value)
            .or_default()
            .extend(changed);
    }

    fn defined_node(&self, value: ValueId) -> Option<(NodeId, usize)> {
        self.value2defined.get(&value).cloned()
    }

    fn used_node(&self, value: ValueId) -> Option<&HashSet<(NodeId, usize)>> {
        self.value2used.get(&value)
    }
}

impl NodeDelete for SimpleGraphModifier {
    fn update_deleted_nodes(&mut self, graph: &mut Graph) {
        let mut visited = HashSet::new();
        for node in graph.outputs.iter() {
            self.used_nodes_dfs(graph, *node, &mut visited);
        }

        for (id, node) in graph
            .nodes
            .iter_mut()
            .filter(|(id, _)| !visited.contains(id))
        {
            if node.is_dummy() {
                // TODO: correct?
                continue;
            }
            node.meta.mark_as_deleted = true;
            for (i, used) in node.inputs.iter().enumerate() {
                if let Entry::Occupied(mut e) = self.value2used.entry(*used) {
                    e.get_mut().remove(&(id, i));
                    if e.get().is_empty() {
                        e.remove();
                    }
                } else {
                    unreachable!();
                }
            }
        }

        for (_, node) in graph
            .nodes
            .iter_mut()
            .filter(|(id, _)| !visited.contains(id))
        {
            for defined in node.outputs.iter() {
                self.value2defined.remove(defined);
                // assert!(!self.value2used.contains_key(defined));
            }
        }
    }
}

impl SimpleGraphModifier {
    fn used_nodes_dfs(&mut self, graph: &Graph, node_id: NodeId, visited: &mut HashSet<NodeId>) {
        if visited.contains(&node_id) {
            return;
        }

        visited.insert(node_id);

        assert!(!graph.nodes[node_id].meta.mark_as_deleted);

        if matches!(graph.nodes[node_id].op, Operator::Input(_)) {
            return;
        }
        for value in graph.nodes[node_id]
            .inputs
            .iter()
            .filter(|v| !graph.initializer.contains_key(v))
        {
            let (defined, _) = self.value2defined.get(value).unwrap();
            self.used_nodes_dfs(graph, *defined, visited);
        }
    }
}

#[derive(Default)]
pub struct ExperimentalGraphModifier {
    value2defined: HashMap<ValueId, (NodeId, usize)>,
    value2used: HashMap<ValueId, HashSet<(NodeId, usize)>>,

    to_delete_nodes: Vec<NodeId>,
    outdegrees: HashMap<NodeId, usize>,
}

impl GraphModifier for ExperimentalGraphModifier {
    fn new(graph: &Graph) -> Self {
        let mut modifier = ExperimentalGraphModifier::default();
        let (value2defined, value2used) = calc_value2xx(graph);
        modifier.value2defined = value2defined;
        modifier.value2used = value2used;
        for (value, (node, _)) in modifier.value2defined.iter() {
            *modifier.outdegrees.entry(*node).or_insert(0) +=
                modifier.value2used.get(value).map_or(0, |x| x.len());
        }
        modifier
    }

    fn register_new_node(&mut self, graph: &mut Graph, v: Node) -> NodeId {
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

    fn replace_input_value_if<P>(
        &mut self,
        graph: &mut Graph,
        old_value: ValueId,
        new_value: ValueId,
        pred: P,
    ) where
        P: Fn(NodeId, &Node) -> bool,
    {
        let old_defines = self.value2defined.get(&old_value).cloned();
        let new_defines = self.value2defined.get(&new_value).cloned();

        if let Some(used) = self.value2used.get(&old_value).cloned() {
            let mut replaced = HashSet::new();
            for &(node, index) in used.iter() {
                if !pred(node, &graph.nodes[node]) {
                    continue;
                }
                replaced.insert((node, index));
                let node = &mut graph.nodes[node];
                if let Operator::Output(v) = node.op {
                    assert!(index == 0);
                    if v == old_value {
                        node.op = Operator::Output(new_value);
                    }
                }
                node.inputs[index] = new_value;
            }
            let len = replaced.len();
            self.value2used.insert(new_value, replaced);
            if let Some((defines, _)) = old_defines {
                self.decr_outdegree(defines, len);
            }
            if let Some((new_defines, _)) = new_defines {
                self.incr_outdegree(new_defines, len);
            }
        }
    }

    fn defined_node(&self, value: ValueId) -> Option<(NodeId, usize)> {
        self.value2defined.get(&value).cloned()
    }

    fn used_node(&self, value: ValueId) -> Option<&HashSet<(NodeId, usize)>> {
        self.value2used.get(&value)
    }
}

impl NodeDelete for ExperimentalGraphModifier {
    fn update_deleted_nodes(&mut self, graph: &mut Graph) {
        let mut vec = Vec::new();
        std::mem::swap(&mut vec, &mut self.to_delete_nodes);
        for node in vec {
            self.delete_nodes_dfs(node, graph);
        }
    }
}

impl ExperimentalGraphModifier {
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

    fn delete_nodes_dfs(&mut self, node_id: NodeId, graph: &mut Graph) {
        if graph.nodes[node_id].meta.mark_as_deleted {
            return;
        }
        if graph.nodes[node_id].is_dummy() {
            return;
        }
        graph.nodes[node_id].meta.mark_as_deleted = true;
        for value in graph.nodes[node_id].outputs.iter() {
            self.value2defined.remove(value);
            self.value2used.remove(value);
        }
        for value in graph.nodes[node_id].inputs.clone().iter() {
            if let Some((defined, _)) = self.value2defined.get(value).cloned() {
                let deg = self.outdegrees.get_mut(&defined).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    self.delete_nodes_dfs(defined, graph);
                }
            }
        }
    }
}
