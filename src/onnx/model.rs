use std::collections::BTreeMap;
use std::collections::HashMap;
use std::ops::Index;
use std::ops::IndexMut;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use id_arena::Arena;
use id_arena::Id;
use itertools::zip_eq;

use crate::onnx::load::ModelLoadError;
use crate::onnx::operator::Operator;
use crate::onnx::operator::OperatorType;
use crate::tensor::data::TensorData;
use crate::tensor::types::DataType;
use crate::tensor::types::Dimension;
use crate::tensor::types::ParamKey;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::TensorType;
use crate::tensor::types::TypeError;
use crate::tensor::types::UnresolvedTensorType;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct OpsetImport {
    pub domain: String,
    pub version: i64,
}

#[derive(Debug)]
pub struct Model {
    pub ir_version: i64,
    pub opset_import: Vec<OpsetImport>,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub graph: Graph,
}

#[derive(Debug)]
pub struct Graph {
    pub nodes: Nodes,
    pub name: String,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub values: Values,
    pub(super) initializer: BTreeMap<ValueId, Tensor>,
    pub(super) external_refs: BTreeMap<ValueId, ExternalTensorRef>,
    mmap_cache: Mutex<HashMap<PathBuf, Arc<memmap2::Mmap>>>,

    pub(crate) resolved_params: HashMap<ParamKey, usize>,
}

#[derive(Debug, Clone)]
pub struct ExternalTensorRef {
    pub path: std::path::PathBuf,
    pub offset: u64,
    pub length: Option<u64>,
    pub elem_type: DataType,
    pub dims: ResolvedTensorDims,
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

    Some(ResolvedTensorDims::new(&res))
}

#[derive(Debug, Clone, Copy)]
pub enum UnifyMode {
    CheckStrides,
    OverwriteStrides,
    IgnoreStrides,
}

impl Graph {
    pub fn resolve_input_types(
        &mut self,
        input_tys: &[ResolvedTensorType],
    ) -> Result<(), TypeError> {
        let ids = self
            .inputs
            .iter()
            .map(|&n| match self.nodes[n].op {
                Operator::Input(v) => v,
                _ => unreachable!(),
            })
            .filter(|&v| !self.has_initializer(v))
            .collect::<Vec<_>>();
        if input_tys.len() != ids.len() {
            return Err(TypeError::InconsistentInput);
        }

        for (id, ty) in zip_eq(ids.iter(), input_tys.iter()) {
            self.try_unify_type(*id, ty, UnifyMode::IgnoreStrides)?;
        }
        Ok(())
    }

    pub fn try_unify_type(
        &mut self,
        value_id: ValueId,
        resolved: &ResolvedTensorType,
        mode: UnifyMode,
    ) -> Result<(), TypeError> {
        match &self.values[value_id].ty.as_ref() {
            Some(TensorType::Resolved(ref ty)) => {
                if ty.dims != resolved.dims {
                    return Err(TypeError::InconsistentInput);
                }

                match mode {
                    UnifyMode::CheckStrides => {
                        if ty.strides() != resolved.strides() {
                            return Err(TypeError::InconsistentInput);
                        }
                    }
                    UnifyMode::OverwriteStrides => {
                        self.values[value_id].ty = Some(TensorType::Resolved(resolved.clone()));
                    }
                    UnifyMode::IgnoreStrides => {}
                };

                Ok(())
            }

            Some(TensorType::Unresolved(ref ty)) => {
                if matches!(mode, UnifyMode::CheckStrides) {
                    return Err(TypeError::InconsistentInput);
                }

                // dbg!(&self.values[value_id]);
                let ty = ty.clone();

                // E.g., Outputs of yolov4
                if let Some(ref dims) = &ty.dims {
                    let _ =
                        unify_types(dims.inner(), &resolved.dims[..], &mut self.resolved_params)
                            .ok_or(TypeError::InconsistentInput)?;
                }
                self.values[value_id].ty = Some(TensorType::Resolved(resolved.clone()));
                Ok(())
            }
            None => {
                self.values[value_id].ty = Some(TensorType::Resolved(resolved.clone()));
                Ok(())
            }
        }
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

    pub fn empty_graph(name: String) -> Self {
        Self {
            nodes: Nodes::default(),
            name,
            inputs: vec![],
            outputs: vec![],
            values: Values::default(),
            initializer: BTreeMap::new(),
            external_refs: BTreeMap::new(),
            mmap_cache: Mutex::new(HashMap::new()),
            resolved_params: HashMap::new(),
        }
    }

    pub fn get_initializer(&self, value_id: ValueId) -> Option<Tensor> {
        if let Some(t) = self.initializer.get(&value_id) {
            return Some(t.clone());
        }
        let r = self.external_refs.get(&value_id)?;
        self.load_external(r).ok()
    }

    pub fn get_inline_initializer(&self, value_id: ValueId) -> Option<&Tensor> {
        self.initializer.get(&value_id)
    }

    pub fn with_external_bytes<R>(
        &self,
        value_id: ValueId,
        f: impl FnOnce(DataType, &[u8]) -> R,
    ) -> Option<Result<R, ModelLoadError>> {
        let r = self.external_refs.get(&value_id)?;
        let mmap = match self.mmap_for(&r.path) {
            Ok(m) => m,
            Err(e) => return Some(Err(e)),
        };
        let start = r.offset as usize;
        let end = r
            .length
            .map(|len| start + len as usize)
            .unwrap_or(mmap.len());
        Some(Ok(f(r.elem_type, &mmap[start..end])))
    }

    fn mmap_for(&self, path: &Path) -> Result<Arc<memmap2::Mmap>, ModelLoadError> {
        let mut cache = self.mmap_cache.lock().unwrap();
        if let Some(m) = cache.get(path) {
            return Ok(Arc::clone(m));
        }
        let file = std::fs::File::open(path).map_err(ModelLoadError::FileRead)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(ModelLoadError::FileRead)?;
        let arc = Arc::new(mmap);
        cache.insert(path.to_path_buf(), Arc::clone(&arc));
        Ok(arc)
    }

    fn load_external(&self, r: &ExternalTensorRef) -> Result<Tensor, ModelLoadError> {
        let mmap = self.mmap_for(&r.path)?;
        let start = r.offset as usize;
        let end = r
            .length
            .map(|len| start + len as usize)
            .unwrap_or(mmap.len());
        let data = TensorData::from_bytes(r.elem_type, &mmap[start..end]);
        Tensor::new(r.dims.clone(), data).map_err(|e| {
            ModelLoadError::Unexpected(format!("external data shape mismatch: {:?}", e))
        })
    }

    pub fn set_initializer(&mut self, value_id: ValueId, tensor: Tensor) {
        self.external_refs.remove(&value_id);
        self.initializer.insert(value_id, tensor);
    }

    pub fn has_initializer(&self, value_id: ValueId) -> bool {
        self.initializer.contains_key(&value_id) || self.external_refs.contains_key(&value_id)
    }

    pub fn initializer_ids(&self) -> Vec<ValueId> {
        self.initializer
            .keys()
            .copied()
            .chain(self.external_refs.keys().copied())
            .collect()
    }

    pub fn remove_initializer<F: Fn(&ValueId) -> bool>(&mut self, remove: F) {
        self.initializer.retain(|k, _| !remove(k));
        self.external_refs.retain(|k, _| !remove(k));
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct NodeMeta {
    pub(crate) mark_as_deleted: bool,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub inputs: Vec<Option<ValueId>>,
    pub outputs: Vec<ValueId>,
    pub name: String,
    pub op: Operator,

    pub(crate) meta: NodeMeta,
}

impl Node {
    pub fn is_dummy(&self) -> bool {
        self.op.operator_type() == OperatorType::Dummy
    }

    pub fn create_node(
        inputs: Vec<Option<ValueId>>,
        outputs: Vec<ValueId>,
        name: String,
        op: Operator,
    ) -> Self {
        Self {
            inputs,
            outputs,
            name,
            op,
            meta: NodeMeta::default(),
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
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
        match &self {
            Self::Resolved(ty) => {
                let dims = ty
                    .dims
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" x ");
                let strides = ty
                    .strides()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" x ");
                format!("[{dims}] / [{strides}]")
            }
            Self::Unresolved(UnresolvedTensorType {
                dims: Some(dims), ..
            }) => dims
                .inner()
                .iter()
                .map(|d| match d {
                    Dimension::Const(x) => x.to_string(),
                    Dimension::Param(x) => x.to_string(),
                })
                .collect::<Vec<_>>()
                .join(" x "),
            Self::Unresolved(_) => ["?".to_string()].join(" x "),
        }
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
                    .filter_map(|v| *v)
                    .map(|v| {
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
                    input.0.replace(".", "_"),
                    node.name.replace(".", "_"),
                    shape
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
