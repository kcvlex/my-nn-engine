use crate::operator::Operator;
use crate::types::{Tensor, TensorType};
use id_arena::{Arena, Id};
use std::collections::HashMap;
use std::ops::{Index, IndexMut};

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
pub struct ValueInfo {
    pub name: String,
    pub ty: Option<TensorType>,
}

#[derive(Debug)]
pub struct Node {
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub name: String,
    pub op: Operator,
}

#[derive(Default, Debug)]
pub struct Values(Arena<ValueInfo>);
pub type ValueId = Id<ValueInfo>;

impl Values {
    pub fn alloc(&mut self, v: ValueInfo) -> ValueId {
        self.0.alloc(v)
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
