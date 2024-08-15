use crate::operator::Operator;
use crate::tensor::{
    dimensions::Dimension,
    tensor::{Tensor, TensorType},
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
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub values: Values,
    pub initializer: HashMap<ValueId, Tensor>,
}

#[derive(Debug)]
pub struct Node {
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub name: String,
    pub op: Operator,
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
    outputs: Vec<GraphvizValue>, // Only deal with output of the graph
}

struct GraphizGraph {
    nodes: Vec<GraphvizNode>,
    inputs: Vec<GraphvizValue>,
    outputs: Vec<GraphvizValue>,
}

impl TensorType {
    fn to_dot(&self) -> String {
        if let Some(ref dims) = self.dims {
            let tmp = dims
                .inner()
                .iter()
                .map(|d| match d {
                    Dimension::Const(x) => x.to_string(),
                    Dimension::Param(x) => x.clone(),
                })
                .collect::<Vec<_>>()
                .join(" x ");
            format!("[{}]", tmp)
        } else {
            "[?]".to_string()
        }
    }
}

impl GraphizGraph {
    fn new(graph: &Graph) -> Self {
        let inputs = graph
            .inputs
            .iter()
            .map(|&v| (graph.values[v].name.clone(), graph.values[v].ty.clone()))
            .collect();
        let outputs: Vec<_> = graph
            .outputs
            .iter()
            .map(|&v| (graph.values[v].name.clone(), graph.values[v].ty.clone()))
            .collect();

        let output_names: HashSet<_> = outputs.iter().map(|(name, _)| name.clone()).collect();

        let mut value2node = HashMap::new();
        for (_, node) in graph.nodes.iter() {
            for &output in node.outputs.iter() {
                let name = graph.values[output].name.clone();
                value2node.insert(name, node.name.clone());
            }
        }
        for name in graph.initializer.keys() {
            let name = graph.values[*name].name.clone();
            value2node.insert(name.clone(), name.clone());
        }
        for input in graph.inputs.iter() {
            let name = graph.values[*input].name.clone();
            value2node.insert(name.clone(), name.clone());
        }
        for output in graph.outputs.iter() {
            let name = graph.values[*output].name.clone();
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
                let outputs = node
                    .outputs
                    .iter()
                    .map(|&v| {
                        let value = &graph.values[v];
                        let from = value2node.get(&value.name).expect("not found").clone();
                        let ty = value.ty.clone();
                        (from, ty)
                    })
                    .filter(|(name, _)| output_names.contains(name))
                    .collect();
                GraphvizNode {
                    name: node.name.clone(),
                    op: node.op.name().to_string(),
                    inputs,
                    outputs,
                }
            })
            .collect();
        Self {
            nodes,
            inputs,
            outputs,
        }
    }
}

impl GraphizGraph {
    fn to_dot(&self) -> String {
        let mut res = "digraph {\n".to_string();
        let outputs: HashSet<_> = self.outputs.iter().map(|(name, _)| name.clone()).collect();
        for input in self.inputs.iter() {
            res.push_str(&format!("  {} [label=\"{}\"];\n", input.0, input.0));
        }
        for node in self.nodes.iter() {
            res.push_str(&format!("  {} [label=\"{}\"];\n", node.name, node.op));
        }
        for output in self.outputs.iter() {
            res.push_str(&format!("  {} [label=\"{}\"];\n", output.0, output.0));
        }
        for node in self.nodes.iter() {
            for input in node.inputs.iter() {
                let shape = input.1.as_ref().map_or(String::from("?"), |x| x.to_dot());
                res.push_str(&format!(
                    "  {} -> {} [label=\"{}\"];\n",
                    input.0, node.name, shape
                ));
            }
            for output in node.outputs.iter() {
                let shape = output.1.as_ref().map_or(String::from("?"), |x| x.to_dot());
                res.push_str(&format!(
                    "  {} -> {} [label=\"{}\"];\n",
                    node.name, output.0, shape
                ));
            }
        }
        res.push_str("}\n");
        res
    }
}

impl Graph {
    pub fn to_dot(&self) -> String {
        GraphizGraph::new(self).to_dot()
    }
}
