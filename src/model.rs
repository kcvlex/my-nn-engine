use crate::operator::Operator;
use crate::tensor::{
    dimensions::Dimension,
    tensor::{ResolvedTensorType, Tensor, TensorType, UnresolvedTensorType},
};
use id_arena::{Arena, Id};
use std::collections::{HashMap, HashSet};
use std::ops::{Deref, Index, IndexMut};

#[derive(Debug)]
pub struct Model {
    pub graph: Graph,
}

#[derive(Debug)]
pub struct Graph {
    pub nodes: Nodes,
    pub name: String,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub values: Values,
    pub initializer: HashMap<ValueId, Tensor>,
}

impl Graph {
    pub fn get_resolved_tensor_type(&self, id: ValueId) -> Option<&ResolvedTensorType> {
        self.values[id].ty.as_ref()?.as_resolved()
    }

    pub fn input_values(&self) -> Vec<ValueId> {
        self.inputs
            .iter()
            .map(|&n| match self.nodes[n].op {
                Operator::Input(v) => v,
                _ => unreachable!("not input"),
            })
            .collect()
    }

    pub fn output_values(&self) -> Vec<ValueId> {
        self.outputs
            .iter()
            .map(|&n| match self.nodes[n].op {
                Operator::Output(v) => v,
                _ => unreachable!("not input"),
            })
            .collect()
    }

    pub fn delete_nodes<T>(&mut self, pred: T)
    where
        T: Fn(&(NodeId, &Node)) -> bool,
    {
        let mut nodes = Nodes::default();
        for (_, node) in self.nodes.iter().filter(|v| {
            let to_delete = pred(v);
            if to_delete && v.1.is_dummy() {
                panic!("cannot delete input/output node");
            }
            !to_delete
        }) {
            nodes.alloc(node.clone());
        }
        for vec in [&mut self.inputs, &mut self.outputs] {
            *vec = vec
                .iter()
                .map(|&n| self.nodes[n].clone())
                .map(|x| nodes.alloc(x))
                .collect();
        }
        self.nodes = nodes;
    }

    pub fn topological_order(&self) -> Vec<NodeId> {
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
        for (id, node) in self.nodes.iter() {
            for value in node.outputs.iter() {
                defined.insert(value, id);
            }
        }
        let mut adj = HashMap::new();
        for (id, node) in self.nodes.iter() {
            for value in node.inputs.iter() {
                let defines = defined.get(value).unwrap();
                adj.entry(*defines).or_insert(HashSet::new()).insert(id);
            }
        }

        println!("{:?}", adj);
        for (id, _) in self.nodes.iter().filter(|(_, node)| !node.is_dummy()) {
            if !visited.contains(&id) {
                dfs(id, &mut res, &mut visited, &adj, &self.nodes);
            }
        }
        res.reverse();
        res
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub name: String,
    pub op: Operator,
}

impl Node {
    pub fn is_dummy(&self) -> bool {
        matches!(self.op, Operator::Input(_) | Operator::Output(_))
    }
}

#[derive(Default, Debug)]
pub struct Nodes(Arena<Node>);
pub type NodeId = Id<Node>;

impl Nodes {
    pub fn alloc(&mut self, v: Node) -> NodeId {
        self.0.alloc(v)
    }
}

impl Index<NodeId> for Nodes {
    type Output = Node;
    fn index(&self, index: NodeId) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<NodeId> for Nodes {
    fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub name: String,
    pub ty: Option<TensorType>,
}

#[derive(Default, Debug)]
pub struct Values(Arena<ValueInfo>);

pub type ValueId = Id<ValueInfo>;

impl Index<ValueId> for Values {
    type Output = ValueInfo;
    fn index(&self, index: ValueId) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<ValueId> for Values {
    fn index_mut(&mut self, index: ValueId) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Values {
    pub fn alloc(&mut self, v: ValueInfo) -> ValueId {
        self.0.alloc(v)
    }
}

impl Deref for Nodes {
    type Target = Arena<Node>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

type GraphvizValue = (String, Option<TensorType>);

struct GraphvizNode {
    name: String,
    op: String,
    inputs: Vec<GraphvizValue>,
}

struct GraphvizGraph {
    nodes: Vec<GraphvizNode>,
}

impl TensorType {
    fn to_dot(&self) -> String {
        let vec = match &self {
            Self::Resolved(ty) => ty.dims.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
            Self::Unresolved(UnresolvedTensorType {
                dims: Some(dims), ..
            }) => dims
                .inner()
                .iter()
                .map(|d| match d {
                    Dimension::Const(x) => x.to_string(),
                    Dimension::Param(x) => x.clone(),
                })
                .collect::<Vec<_>>(),
            Self::Unresolved(_) => vec!["?".to_string()],
        };
        format!("[{}]", vec.join(" x "))
    }
}

impl GraphvizGraph {
    fn new(graph: &Graph) -> Self {
        let mut value2node = HashMap::new();
        for (_, node) in graph.nodes.iter() {
            for defined in &node.outputs {
                let name = graph.values[*defined].name.clone();
                value2node.insert(name, node.name.clone());
            }
        }
        for name in graph.initializer.keys() {
            let name = graph.values[*name].name.clone();
            value2node.insert(name.clone(), name.clone());
        }

        let nodes = graph
            .nodes
            .iter()
            .map(|(_, node)| {
                let inputs = node
                    .inputs
                    .iter()
                    .map(|&v| {
                        let value = &graph.values[v];
                        let from = value2node.get(&value.name).expect("not found").clone();
                        let ty = value.ty.clone();
                        (from, ty)
                    })
                    .collect();
                GraphvizNode {
                    name: node.name.clone(),
                    op: node.op.name().to_string(),
                    inputs,
                }
            })
            .collect();
        Self { nodes }
    }
}

impl GraphvizGraph {
    fn to_dot(&self) -> String {
        let mut res = "digraph {\n".to_string();
        for node in self.nodes.iter() {
            res.push_str(&format!("  {} [label=\"{}\"];\n", node.name, node.op));
        }
        for node in self.nodes.iter() {
            for input in node.inputs.iter() {
                let shape = input.1.as_ref().map_or(String::from("?"), |x| x.to_dot());
                res.push_str(&format!(
                    "  {} -> {} [label=\"{}\"];\n",
                    input.0, node.name, shape
                ));
            }
        }
        res.push_str("}\n");
        res
    }
}

impl Graph {
    pub fn to_dot(&self) -> String {
        GraphvizGraph::new(self).to_dot()
    }
}
