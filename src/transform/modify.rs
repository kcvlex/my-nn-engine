use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::collections::HashSet;

use indexmap::IndexSet;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::ValueId;
use crate::onnx::model::ValueInfo;
use crate::onnx::operator::Operator;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::TensorType;
use crate::tensor::Tensor;

pub trait GraphOp {
    // fn new(graph: &Graph) -> Self;

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

    fn register_new_tensor(&mut self, graph: &mut Graph, tensor: Tensor, name: String) -> ValueId {
        let value_id = self.register_new_value(graph, name, tensor.tensor_type());
        graph.initializer.insert(value_id, tensor);
        value_id
    }

    fn defined_node(&self, value: ValueId) -> Option<(NodeId, usize)>;

    fn used_node(&self, value: ValueId) -> Option<&IndexSet<(NodeId, usize)>>;

    fn replace_input_value_if<P>(
        &mut self,
        graph: &mut Graph,
        old_value: ValueId,
        new_value: ValueId,
        pred: P,
    ) where
        P: Fn(NodeId, &Node) -> bool,
    {
        if graph.get_resolved_tensor_type(old_value).unwrap() !=
            graph.get_resolved_tensor_type(new_value).unwrap()
        {
            dbg!(graph.get_resolved_tensor_type(old_value));
            dbg!(graph.get_resolved_tensor_type(new_value));
            panic!("Type mismatch");
        }
        self.replace_input_value_if_without_typecheck(graph, old_value, new_value, pred);
    }

    fn replace_input_value_if_without_typecheck<P>(
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

    fn replace_tensor_type(&self, graph: &mut Graph, value_id: ValueId, ty: ResolvedTensorType) {
        graph.values[value_id].ty = Some(TensorType::Resolved(ty));
    }

    fn replace_tensor(&self, graph: &mut Graph, value_id: ValueId, tensor: Tensor) {
        self.replace_tensor_type(graph, value_id, tensor.tensor_type());
        *graph.initializer.get_mut(&value_id).unwrap() = tensor;
    }

    fn set_node_input(&mut self, graph: &mut Graph, node_id: NodeId, index: usize, value: ValueId);

    fn drop_node_input(&mut self, graph: &mut Graph, node_id: NodeId, index: usize);

    /// Returns `(source_value, chain)` where `source_value` is the input of the last
    /// matched node (i.e., the value just before the chain), and `chain` is the list of
    /// matched `NodeId`s in backward order (closest to `value` first).
    fn walk_chain_backward<F>(
        &self,
        graph: &Graph,
        value: ValueId,
        pred: F,
    ) -> (ValueId, Vec<NodeId>)
    where
        F: Fn(&Node) -> bool,
    {
        let mut chain = Vec::new();
        let mut cur = value;

        loop {
            let Some((node_id, _)) = self.defined_node(cur) else {
                break;
            };
            let node = &graph.nodes[node_id];
            if !pred(node) {
                break;
            }
            chain.push(node_id);
            cur = node.inputs[0].unwrap();
        }

        (cur, chain)
    }

    /// Returns `(final_value, chain)` where `final_value` is the output of the last
    /// matched node (i.e., the value just after the chain), and `chain` is the list of
    /// matched `NodeId`s in forward order (closest to `value` first).
    fn walk_chain_forward<F>(
        &self,
        graph: &Graph,
        value: ValueId,
        pred: F,
    ) -> (ValueId, Vec<NodeId>)
    where
        F: Fn(&Node) -> bool,
    {
        let mut chain = Vec::new();
        let mut cur = value;

        loop {
            let Some(users) = self.used_node(cur) else {
                break;
            };
            if users.len() != 1 {
                break;
            }
            let (node_id, _) = *users.iter().next().unwrap();
            let node = &graph.nodes[node_id];
            if !pred(node) {
                break;
            }
            chain.push(node_id);
            cur = node.outputs[0];
        }

        (cur, chain)
    }
}

pub trait NodeDelete {
    fn update_deleted_nodes(&mut self, graph: &mut Graph);
}

type Value2Defined = HashMap<ValueId, (NodeId, usize)>;
type Value2Used = HashMap<ValueId, IndexSet<(NodeId, usize)>>;

fn calc_value2xx(graph: &Graph) -> (Value2Defined, Value2Used) {
    let mut value2defined = HashMap::new();
    let mut value2used = HashMap::new();
    for (node_id, node) in graph.nodes.iter() {
        for (index, value) in node.inputs.iter().enumerate() {
            if let Some(value) = value {
                value2used
                    .entry(*value)
                    .or_insert_with(IndexSet::new)
                    .insert((node_id, index));
            }
        }
        for (index, &value) in node.outputs.iter().enumerate() {
            value2defined.insert(value, (node_id, index));
        }
    }
    (value2defined, value2used)
}

pub struct SimpleGraphOp {
    value2defined: HashMap<ValueId, (NodeId, usize)>,
    value2used: HashMap<ValueId, IndexSet<(NodeId, usize)>>,
}

impl SimpleGraphOp {
    pub fn new(graph: &Graph) -> Self {
        let (value2defined, value2used) = calc_value2xx(graph);
        SimpleGraphOp {
            value2defined,
            value2used,
        }
    }
}

impl GraphOp for SimpleGraphOp {
    fn register_new_node(&mut self, graph: &mut Graph, v: Node) -> NodeId {
        let inputs = v.inputs.clone();
        let outputs = v.outputs.clone();
        let res = graph.nodes.alloc(v);
        for (index, value) in inputs.iter().enumerate() {
            if let Some(value) = value {
                self.value2used
                    .entry(*value)
                    .or_default()
                    .insert((res, index));
            }
        }
        for (index, value) in outputs.iter().enumerate() {
            self.value2defined.insert(*value, (res, index));
        }
        res
    }

    fn replace_input_value_if_without_typecheck<P>(
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
                old.get_mut().shift_remove(v);
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
            node.inputs[*index] = Some(new_value);
        }

        self.value2used
            .entry(new_value)
            .or_default()
            .extend(changed);
    }

    fn defined_node(&self, value: ValueId) -> Option<(NodeId, usize)> {
        self.value2defined.get(&value).cloned()
    }

    fn used_node(&self, value: ValueId) -> Option<&IndexSet<(NodeId, usize)>> {
        self.value2used.get(&value)
    }

    fn set_node_input(&mut self, graph: &mut Graph, node_id: NodeId, index: usize, value: ValueId) {
        while graph.nodes[node_id].inputs.len() <= index {
            graph.nodes[node_id].inputs.push(None);
        }
        if let Some(old_value) = graph.nodes[node_id].inputs[index] {
            if let Entry::Occupied(mut entry) = self.value2used.entry(old_value) {
                entry.get_mut().shift_remove(&(node_id, index));
                if entry.get().is_empty() {
                    entry.remove();
                }
            }
        }
        graph.nodes[node_id].inputs[index] = Some(value);
        self.value2used
            .entry(value)
            .or_default()
            .insert((node_id, index));
    }

    fn drop_node_input(&mut self, graph: &mut Graph, node_id: NodeId, index: usize) {
        let Some(old_value) = graph.nodes[node_id].inputs[index] else {
            return;
        };
        graph.nodes[node_id].inputs[index] = None;
        let Entry::Occupied(mut old) = self.value2used.entry(old_value) else {
            panic!("Inconsistent value2used");
        };
        old.get_mut().shift_remove(&(node_id, index));
        if old.get().is_empty() {
            old.remove();
        }
    }
}

impl NodeDelete for SimpleGraphOp {
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
                if let Some(used) = used {
                    if let Entry::Occupied(mut e) = self.value2used.entry(*used) {
                        e.get_mut().shift_remove(&(id, i));
                        if e.get().is_empty() {
                            e.remove();
                        }
                    } else {
                        unreachable!();
                    }
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

impl SimpleGraphOp {
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
            .filter_map(|v| v.as_ref())
            .filter(|v| !graph.initializer.contains_key(v))
        {
            let (defined, _) = self.value2defined.get(value).unwrap();
            self.used_nodes_dfs(graph, *defined, visited);
        }
    }
}
