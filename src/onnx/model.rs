use crate::onnx::operator::Operator;
use crate::tensor::{
    dimensions::{Dimension, ParamKey},
    resolved_dimensions::ResolvedTensorDims,
    tensor::{ResolvedTensorType, Tensor, TensorType, TypeError, UnresolvedTensorType},
};
use id_arena::{Arena, Id};
use std::collections::{BTreeMap, HashMap};
use std::ops::{Index, IndexMut};

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
    pub initializer: BTreeMap<ValueId, Tensor>,

    pub(crate) resolved_params: HashMap<ParamKey, usize>,
}

fn unify_types(
    lhs: &[Dimension],
    rhs: &[usize],
    params: &mut HashMap<ParamKey, usize>,
) -> Option<ResolvedTensorDims> {
    if lhs.len() != rhs.len() {
        return None;
    }

    let mut res = Vec::new();
    for (ld, rd) in lhs.iter().zip(rhs.iter()) {
        let mut ld = ld.clone();
        if let Dimension::Param(ref k) = ld {
            if let Some(x) = params.get(k) {
                ld = Dimension::Const(*x);
            }
        }
        match ld {
            Dimension::Const(x) => {
                if x != *rd {
                    return None;
                }
            }
            Dimension::Param(k) => {
                params.insert(k.clone(), *rd);
            }
        };
        res.push(*rd);
    }

    Some(res.into())
}

impl Graph {
    pub fn resolve_input_types(
        &mut self,
        input_tys: &[&ResolvedTensorDims],
    ) -> Result<(), TypeError> {
        let ids = self
            .inputs
            .iter()
            .map(|&n| match self.nodes[n].op {
                Operator::Input(v) => v,
                _ => unreachable!(),
            })
            .filter(|&v| !self.initializer.contains_key(&v))
            .collect::<Vec<_>>();
        if input_tys.len() != ids.len() {
            return Err(TypeError::InconsistentInput);
        }

        for (i, id) in ids.iter().enumerate() {
            match self.values[*id].ty.as_ref().unwrap() {
                TensorType::Resolved(_) => (), // TODO: check if consistent
                TensorType::Unresolved(ref ty) => {
                    let ty = ty.clone();
                    let res = unify_types(
                        ty.dims.unwrap().inner().as_slice(),
                        input_tys[i].as_slice(),
                        &mut self.resolved_params,
                    )
                    .ok_or(TypeError::InconsistentInput)?;
                    self.values[*id].ty = Some(TensorType::Resolved(ResolvedTensorType::new(
                        ty.elem_type,
                        res,
                    )));
                }
            };
        }
        Ok(())
    }

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

    pub fn delete_nodes(&mut self) {
        let mut nodes = Nodes::default();
        for (_, node) in self.nodes.iter().filter(|(_, node)| {
            if node.meta.mark_as_deleted && node.is_dummy() {
                panic!("cannot delete input/output node");
            }
            !node.meta.mark_as_deleted && !node.is_dummy()
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

    pub fn is_output_value(&self, id: ValueId) -> bool {
        self.outputs.iter().any(|&n| match self.nodes[n].op {
            Operator::Output(v) => v == id,
            _ => false,
        })
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct NodeMeta {
    pub(crate) mark_as_deleted: bool,
    pub(crate) omp_parallel: Option<usize>,
    pub(crate) omp_for: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub name: String,
    pub op: Operator,

    pub(crate) meta: NodeMeta,
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

    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.0.iter().filter(|(_, v)| !v.meta.mark_as_deleted)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (NodeId, &mut Node)> {
        self.0.iter_mut().filter(|(_, v)| !v.meta.mark_as_deleted)
    }
}

impl Index<NodeId> for Nodes {
    type Output = Node;
    fn index(&self, index: NodeId) -> &Self::Output {
        let res = &self.0[index];
        assert!(!res.meta.mark_as_deleted);
        res
    }
}

impl IndexMut<NodeId> for Nodes {
    fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
        let res = &mut self.0[index];
        assert!(!res.meta.mark_as_deleted);
        res
    }
}

#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub name: String,
    pub ty: Option<TensorType>,
}

#[derive(Default, Debug)]
pub struct Values(Arena<ValueInfo>);

impl Values {
    pub fn inner(&self) -> &Arena<ValueInfo> {
        &self.0
    }
}

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

/*
impl Deref for Nodes {
    type Target = Arena<Node>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
*/

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
                    Dimension::Param(x) => x.to_string(),
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
