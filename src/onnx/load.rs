use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use itertools::Itertools;
use prost::DecodeError;
use prost::Message;

use crate::graph::operator::*;
use crate::graph::ExternalTensorRef;
use crate::graph::Graph;
use crate::graph::Node;
use crate::graph::NodeMeta;
use crate::graph::Nodes;
use crate::graph::ValueId;
use crate::graph::ValueInfo;
use crate::graph::Values;
use crate::onnx::Model;
use crate::onnx::OpsetImport;
use crate::tensor::data::ScalarData;
use crate::tensor::data::TensorData;
use crate::tensor::types::DataType;
use crate::tensor::types::Dimension;
use crate::tensor::types::FloatType;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::SIntType;
use crate::tensor::types::TensorType;
use crate::tensor::types::TypeError;
use crate::tensor::types::UIntType;
use crate::tensor::types::UnresolvedTensorDims;
use crate::tensor::types::UnresolvedTensorType;
use crate::tensor::Tensor;

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

#[derive(Debug)]
pub enum ModelLoadError {
    FileRead(std::io::Error),
    Decode(DecodeError),
    ElemTypeUnspecified,
    NoGraph,
    UnsupportedElemType(tensor_proto::DataType),
    UnsupportedValueType(type_proto::Value),
    UnsupportedAttributeType(attribute_proto::AttributeType),
    UnsupportedOp(String),
    NegativeDimension(i64),
    TypeError(TypeError),
    Required(String),
    Unexpected(String),
}

type LoadResult<T> = Result<T, ModelLoadError>;

pub trait LoadProto {
    fn load_from_path<P: AsRef<Path>>(p: P) -> LoadResult<Self>
    where
        Self: Sized;
}

impl LoadProto for Model {
    fn load_from_path<P: AsRef<Path>>(p: P) -> LoadResult<Self> {
        let base_dir = p.as_ref().parent().map(|p| p.to_path_buf());
        let model = std::fs::read(p).map_err(ModelLoadError::FileRead)?;
        let model = ModelProto::decode(&*model).map_err(ModelLoadError::Decode)?;
        let opset_import = model
            .opset_import
            .iter()
            .map(|o| OpsetImport {
                domain: o.domain.clone(),
                version: o.version,
            })
            .collect();
        let graph = model.graph.ok_or(ModelLoadError::NoGraph)?;
        let graph = GraphLoader::default().load_graph(graph, base_dir.as_deref())?;
        Ok(Model {
            ir_version: model.ir_version,
            opset_import,
            producer_name: model.producer_name,
            producer_version: model.producer_version,
            domain: model.domain,
            model_version: model.model_version,
            doc_string: model.doc_string,
            graph,
        })
    }
}

impl LoadProto for Tensor {
    fn load_from_path<P: AsRef<Path>>(p: P) -> LoadResult<Self> {
        let tensor = std::fs::read(p).map_err(ModelLoadError::FileRead)?;
        let tensor = TensorProto::decode(&*tensor).map_err(ModelLoadError::Decode)?;
        load_tensor(tensor, None)
    }
}

impl Tensor {
    /// Convert Tensor to ONNX TensorProto bytes
    pub fn to_proto_bytes(&self) -> Vec<u8> {
        let proto = tensor_to_proto(self);
        proto.encode_to_vec()
    }

    /// Create Tensor from ONNX TensorProto bytes
    pub fn from_proto_bytes(bytes: &[u8]) -> LoadResult<Self> {
        let proto = TensorProto::decode(bytes).map_err(ModelLoadError::Decode)?;
        load_tensor(proto, None)
    }
}

#[derive(Default)]
struct GraphLoader {
    entries: HashMap<String, ValueId>,
    values: Values,
    external_refs: BTreeMap<ValueId, ExternalTensorRef>,
}

#[allow(dead_code)]
#[derive(Debug)]
enum Attribute {
    Float(f32),
    Int(i64),
    Ints(Vec<i64>),
    Str(String),
    Strings(Vec<String>),
    Tensor(Tensor),
}

#[allow(dead_code)]
impl Attribute {
    fn f(&self) -> LoadResult<f32> {
        match self {
            Attribute::Float(x) => Ok(*x),
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
    }

    fn i(&self) -> LoadResult<i64> {
        match self {
            Attribute::Int(x) => Ok(*x),
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
    }

    fn index(&self) -> LoadResult<TensorIndex> {
        self.i().map(|x| TensorIndex::new(x as isize))
    }

    fn b(&self) -> LoadResult<bool> {
        self.i().map(|x| x != 0)
    }

    fn ints<T>(&self) -> LoadResult<Vec<T>>
    where
        T: TryFrom<i64>,
    {
        match self {
            Attribute::Ints(x) => {
                let mut vec = Vec::with_capacity(x.len());
                for i in x.iter() {
                    vec.push(
                        T::try_from(*i).map_err(|_| ModelLoadError::Unexpected("".to_string()))?,
                    );
                }
                Ok(vec)
            }
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
    }

    fn indexes(&self) -> LoadResult<Vec<TensorIndex>> {
        let ints = self.ints()?;
        Ok(ints.into_iter().map(TensorIndex::new).collect())
    }

    fn s(&self) -> LoadResult<&str> {
        match self {
            Attribute::Str(x) => Ok(x),
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
    }

    fn ty(&self) -> LoadResult<DataType> {
        self.i().and_then(|x| DataType::try_from(x as i32))
    }

    fn strings(&self) -> LoadResult<Vec<String>> {
        match self {
            Attribute::Strings(x) => Ok(x.clone()),
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
    }

    fn tensor(&self) -> LoadResult<Tensor> {
        match self {
            Attribute::Tensor(x) => Ok(x.clone()),
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
    }
}

type Attributes = HashMap<String, Attribute>;

impl GraphLoader {
    fn load_graph(mut self, graph: GraphProto, base_dir: Option<&Path>) -> LoadResult<Graph> {
        let graph_name = graph.name.clone();
        let input = {
            let initializer_names: HashSet<_> =
                graph.initializer.iter().map(|x| x.name.as_str()).collect();
            let input: Vec<_> = graph
                .input
                .into_iter()
                .filter(|x| !initializer_names.contains(x.name.as_str()))
                .collect();
            input
        };
        let loaded_initializers = self.load_initializer(graph.initializer, base_dir)?;
        let inputs = self.load_value_info_vec(input)?;
        let outputs = self.load_value_info_vec(graph.output)?;
        let mut nodes = self.load_nodes(graph.node, base_dir)?;
        let inputs = {
            let mut res = Vec::with_capacity(inputs.len());
            for &x in inputs.iter() {
                let node = Node {
                    name: self.values[x].name.clone(),
                    inputs: Vec::new(),
                    outputs: vec![x],
                    op: Operator::Input(x),
                    meta: NodeMeta::default(),
                };
                res.push(nodes.alloc(node));
            }
            res
        };
        let outputs = outputs
            .into_iter()
            .map(|x| {
                let node = Node {
                    name: self.values[x].name.clone(),
                    inputs: vec![Some(x)],
                    outputs: Vec::new(),
                    op: Operator::Output(x),
                    meta: NodeMeta::default(),
                };
                nodes.alloc(node)
            })
            .collect();

        // Fill dummy values
        let defined: HashSet<ValueId> = loaded_initializers
            .keys()
            .copied()
            .chain(self.external_refs.keys().copied())
            .chain(inputs.iter().map(|&x| nodes[x].outputs[0]))
            .chain(
                nodes
                    .iter()
                    .flat_map(|(_, node)| node.outputs.iter().copied()),
            )
            .collect();
        let tensor = Tensor::new(
            ResolvedTensorDims::new(&[]),
            TensorData::Float(FloatType::F32, vec![-42.0]),
        )
        .unwrap();
        let mut dummy_initializers: Vec<(ValueId, Tensor)> = Vec::new();
        for value_id in nodes
            .iter()
            .flat_map(|(_, node)| node.inputs.iter())
            .filter_map(|x| x.as_ref())
            .filter(|&x| !defined.contains(x))
            .unique()
        {
            dummy_initializers.push((*value_id, tensor.clone()));
            self.values[*value_id].ty = Some(TensorType::Resolved(tensor.tensor_type()));
        }

        let mut graph = Graph::empty_graph(graph_name);
        graph.nodes = nodes;
        graph.inputs = inputs;
        graph.outputs = outputs;
        graph.values = self.values;
        for (id, t) in loaded_initializers {
            graph.set_initializer(id, t);
        }
        graph.external_refs = self.external_refs;
        for (id, t) in dummy_initializers {
            graph.set_initializer(id, t);
        }
        Ok(graph)
    }

    fn load_value_info_vec(&mut self, v: Vec<ValueInfoProto>) -> LoadResult<Vec<ValueId>> {
        let mut res = Vec::new();
        for info in v.into_iter() {
            let name = info.name;
            let ty = info.r#type.ok_or(ModelLoadError::Unexpected(
                "ValueInfo.type must be specified".to_string(),
            ))?;
            let ty = load_type(ty)?;
            let id = self
                .entries
                .entry(name.clone())
                .or_insert_with(|| self.values.alloc(ValueInfo { name, ty: Some(ty) }));
            res.push(*id);
        }
        Ok(res)
    }

    fn load_initializer(
        &mut self,
        v: Vec<TensorProto>,
        base_dir: Option<&Path>,
    ) -> LoadResult<BTreeMap<ValueId, Tensor>> {
        let mut res = BTreeMap::new();
        for tensor in v.into_iter() {
            let name = tensor.name.clone();
            if !tensor.external_data.is_empty() {
                let external_ref = make_external_ref(&tensor, base_dir)?;
                let ty = ResolvedTensorType::new(external_ref.elem_type, external_ref.dims.clone());
                let id = *self.entries.entry(name.clone()).or_insert_with(|| {
                    self.values.alloc(ValueInfo {
                        name,
                        ty: Some(TensorType::Resolved(ty)),
                    })
                });
                self.external_refs.insert(id, external_ref);
                continue;
            }
            let tensor = load_tensor(tensor, base_dir)?;
            let id = self.entries.entry(name.clone()).or_insert_with(|| {
                self.values.alloc(ValueInfo {
                    name,
                    ty: Some(TensorType::Resolved(tensor.tensor_type())),
                })
            });
            res.insert(*id, tensor);
        }
        Ok(res)
    }

    fn load_nodes(&mut self, nodes: Vec<NodeProto>, base_dir: Option<&Path>) -> LoadResult<Nodes> {
        let mut resolve = |x: String| -> ValueId {
            *self.entries.entry(x.clone()).or_insert_with(|| {
                self.values.alloc(ValueInfo {
                    name: x.clone(),
                    ty: None,
                })
            })
        };
        let mut res = Nodes::default();
        for node in nodes.into_iter() {
            let name = node.name;
            let inputs: Vec<Option<ValueId>> = node
                .input
                .into_iter()
                .map(|x| if x.is_empty() { None } else { Some(resolve(x)) })
                .collect();
            let outputs: Vec<ValueId> = node.output.into_iter().map(|x| resolve(x)).collect();
            let attributes = load_attributes(node.attribute, base_dir)?;
            let op = load_op(&node.op_type, &attributes)?;
            res.alloc(Node {
                name,
                inputs,
                outputs,
                op,
                meta: NodeMeta::default(),
            });
        }
        Ok(res)
    }
}

fn parse_external_data(
    tensor: &TensorProto,
    base_dir: Option<&Path>,
) -> LoadResult<(std::path::PathBuf, u64, Option<u64>)> {
    let ed: HashMap<&str, &str> = tensor
        .external_data
        .iter()
        .map(|kv| (kv.key.as_str(), kv.value.as_str()))
        .collect();
    let location = ed
        .get("location")
        .ok_or_else(|| ModelLoadError::Unexpected("external_data missing location".to_string()))?;
    let offset = ed
        .get("offset")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let length = ed.get("length").and_then(|s| s.parse::<u64>().ok());
    let base = base_dir
        .ok_or_else(|| ModelLoadError::Unexpected("external_data requires base_dir".to_string()))?;
    Ok((base.join(location), offset, length))
}

fn make_external_ref(
    tensor: &TensorProto,
    base_dir: Option<&Path>,
) -> LoadResult<ExternalTensorRef> {
    let elem_type = DataType::try_from(tensor.data_type)?;
    let (path, offset, length) = parse_external_data(tensor, base_dir)?;
    let dims =
        ResolvedTensorDims::new(&tensor.dims.iter().map(|&x| x as usize).collect::<Vec<_>>());
    Ok(ExternalTensorRef {
        path,
        offset,
        length,
        elem_type,
        dims,
    })
}

fn load_tensor(tensor: TensorProto, base_dir: Option<&Path>) -> LoadResult<Tensor> {
    use std::io::Read;
    use std::io::Seek;
    use std::io::SeekFrom;

    let elem_type = DataType::try_from(tensor.data_type)?;
    let data = if !tensor.external_data.is_empty() {
        let (path, offset, length) = parse_external_data(&tensor, base_dir)?;
        let mut file = std::fs::File::open(&path).map_err(ModelLoadError::FileRead)?;
        file.seek(SeekFrom::Start(offset))
            .map_err(ModelLoadError::FileRead)?;
        let raw = if let Some(len) = length {
            let mut buf = vec![0u8; len as usize];
            file.read_exact(&mut buf)
                .map_err(ModelLoadError::FileRead)?;
            buf
        } else {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)
                .map_err(ModelLoadError::FileRead)?;
            buf
        };
        TensorData::from_bytes(elem_type, &raw)
    } else if tensor.raw_data.is_empty() {
        match elem_type {
            DataType::Bool => TensorData::Bool(
                tensor
                    .int32_data
                    .into_iter()
                    .map(|v| if v != 0 { 1u8 } else { 0u8 })
                    .collect(),
            ),
            DataType::SInt(ty @ SIntType::I8) => {
                TensorData::SInt(ty, tensor.int32_data.into_iter().map(i64::from).collect())
            }
            DataType::SInt(ty @ SIntType::I32) => {
                TensorData::SInt(ty, tensor.int32_data.into_iter().map(i64::from).collect())
            }
            DataType::SInt(ty @ SIntType::I64) => TensorData::SInt(ty, tensor.int64_data),
            DataType::UInt(ty @ UIntType::U8) => TensorData::UInt(
                ty,
                tensor.int32_data.into_iter().map(|v| v as u64).collect(),
            ),
            DataType::UInt(ty @ UIntType::U64) => TensorData::UInt(ty, tensor.uint64_data),
            DataType::Float(ty @ FloatType::F32) => {
                TensorData::Float(ty, tensor.float_data.into_iter().map(f64::from).collect())
            }
            DataType::Float(ty @ FloatType::F64) => TensorData::Float(ty, tensor.double_data),
            DataType::Float(FloatType::BF16) => {
                return Err(ModelLoadError::TypeError(
                    crate::tensor::types::TypeError::InferError(
                        "inline BF16 data without raw_data is unsupported".to_string(),
                    ),
                ));
            }
        }
    } else {
        TensorData::from_bytes(elem_type, tensor.raw_data.as_slice())
    };
    let mut dims = Vec::new();
    for dim in tensor.dims.into_iter() {
        let dim = usize::try_from(dim).map_err(|_| ModelLoadError::NegativeDimension(dim))?;
        dims.push(dim);
    }
    let dims = ResolvedTensorDims::new(&dims);
    Tensor::new(dims, data).map_err(ModelLoadError::TypeError)
}

pub(crate) fn tensor_to_proto(tensor: &Tensor) -> TensorProto {
    let data_type = match tensor.data.elem_type() {
        DataType::Bool => tensor_proto::DataType::Bool as i32,
        DataType::SInt(SIntType::I8) => tensor_proto::DataType::Int8 as i32,
        DataType::SInt(SIntType::I32) => tensor_proto::DataType::Int32 as i32,
        DataType::SInt(SIntType::I64) => tensor_proto::DataType::Int64 as i32,
        DataType::UInt(UIntType::U8) => tensor_proto::DataType::Uint8 as i32,
        DataType::UInt(UIntType::U64) => tensor_proto::DataType::Uint64 as i32,
        DataType::Float(FloatType::BF16) => tensor_proto::DataType::Bfloat16 as i32,
        DataType::Float(FloatType::F32) => tensor_proto::DataType::Float as i32,
        DataType::Float(FloatType::F64) => tensor_proto::DataType::Double as i32,
    };

    let dims: Vec<i64> = tensor.dims.iter().map(|&d| d as i64).collect();

    // BF16 has no inline field in TensorProto; encode as little-endian raw_data.
    if let TensorData::Float(FloatType::BF16, v) = &tensor.data {
        let raw_data: Vec<u8> = v
            .iter()
            .flat_map(|&x| f64_to_bf16_bits(x).to_le_bytes())
            .collect();
        return TensorProto {
            dims,
            data_type,
            raw_data,
            ..Default::default()
        };
    }

    let (float_data, double_data, int32_data, int64_data, uint64_data) = match &tensor.data {
        TensorData::Bool(v) => (
            vec![],
            vec![],
            v.iter().map(|&b| b as i32).collect(),
            vec![],
            vec![],
        ),
        TensorData::Float(FloatType::F32, v) => (
            v.iter().map(|&x| x as f32).collect(),
            vec![],
            vec![],
            vec![],
            vec![],
        ),
        TensorData::Float(FloatType::F64, v) => (vec![], v.clone(), vec![], vec![], vec![]),
        TensorData::Float(FloatType::BF16, _) => unreachable!("BF16 handled above as raw_data"),
        TensorData::SInt(SIntType::I8, v) => (
            vec![],
            vec![],
            v.iter().map(|&x| x as i32).collect(),
            vec![],
            vec![],
        ),
        TensorData::SInt(SIntType::I32, v) => (
            vec![],
            vec![],
            v.iter().map(|&x| x as i32).collect(),
            vec![],
            vec![],
        ),
        TensorData::SInt(SIntType::I64, v) => (vec![], vec![], vec![], v.clone(), vec![]),
        TensorData::UInt(UIntType::U8, v) => (
            vec![],
            vec![],
            v.iter().map(|&x| x as i32).collect(),
            vec![],
            vec![],
        ),
        TensorData::UInt(UIntType::U64, v) => (vec![], vec![], vec![], vec![], v.clone()),
    };

    TensorProto {
        dims,
        data_type,
        float_data,
        double_data,
        int32_data,
        int64_data,
        uint64_data,
        ..Default::default()
    }
}

fn f64_to_bf16_bits(x: f64) -> u16 {
    let x = x as f32;
    if x.is_nan() {
        // Canonical bf16 quiet NaN.
        return 0x7FC0;
    }
    let bits = x.to_bits();
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

fn load_type(ty: TypeProto) -> LoadResult<TensorType> {
    let ty = ty.value.ok_or(ModelLoadError::Unexpected(
        "TypeProto.value must be specified".to_string(),
    ))?;
    match ty {
        type_proto::Value::TensorType(tensor) => load_tensor_type(tensor),
        x => Err(ModelLoadError::UnsupportedValueType(x)),
    }
}

fn load_tensor_type(tensor: type_proto::Tensor) -> LoadResult<TensorType> {
    let elem_type = DataType::try_from(tensor.elem_type)?;
    if let Some(dims) = tensor.shape {
        let mut shape = Vec::new();
        for dim in dims.dim.into_iter() {
            let dim: Dimension = dim
                .value
                .ok_or(ModelLoadError::Unexpected(
                    "Dimension.value must be specified".to_string(),
                ))?
                .try_into()?;
            shape.push(dim);
        }
        let mut ty = TensorType::Unresolved(UnresolvedTensorType {
            elem_type,
            dims: Some(UnresolvedTensorDims::new(&shape)),
        });
        ty.normalize();
        Ok(ty)
    } else {
        Ok(TensorType::Unresolved(UnresolvedTensorType {
            elem_type,
            dims: None,
        }))
    }
}

impl TryFrom<i32> for DataType {
    type Error = ModelLoadError;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let value = tensor_proto::DataType::try_from(value)
            .map_err(|e| ModelLoadError::Unexpected(format!("Invalid DataType: {:?}", e)))?;
        match value {
            tensor_proto::DataType::Bool => Ok(DataType::Bool),
            tensor_proto::DataType::Bfloat16 => Ok(FloatType::BF16.into()),
            tensor_proto::DataType::Float => Ok(FloatType::F32.into()),
            tensor_proto::DataType::Double => Ok(FloatType::F64.into()),
            tensor_proto::DataType::Int8 => Ok(SIntType::I8.into()),
            tensor_proto::DataType::Int32 => Ok(SIntType::I32.into()),
            tensor_proto::DataType::Int64 => Ok(SIntType::I64.into()),
            tensor_proto::DataType::Uint64 => Ok(UIntType::U64.into()),
            tensor_proto::DataType::Undefined => Err(ModelLoadError::ElemTypeUnspecified),
            x => Err(ModelLoadError::UnsupportedElemType(x)),
        }
    }
}

impl TryFrom<tensor_shape_proto::dimension::Value> for Dimension {
    type Error = ModelLoadError;
    fn try_from(value: tensor_shape_proto::dimension::Value) -> Result<Self, Self::Error> {
        match value {
            tensor_shape_proto::dimension::Value::DimValue(x) => {
                let x = x
                    .try_into()
                    .map_err(|_| ModelLoadError::NegativeDimension(x))?;
                Ok(Dimension::Const(x))
            }
            tensor_shape_proto::dimension::Value::DimParam(x) => Ok(Dimension::Param(x.into())),
        }
    }
}

fn load_utf8(v: Vec<u8>) -> LoadResult<String> {
    String::from_utf8(v).map_err(|err| ModelLoadError::Unexpected(err.to_string()))
}

fn load_attributes(v: Vec<AttributeProto>, base_dir: Option<&Path>) -> LoadResult<Attributes> {
    let mut res = HashMap::new();
    for attr in v.into_iter() {
        let name = attr.name;
        let ty = attribute_proto::AttributeType::try_from(attr.r#type)
            .map_err(|err| ModelLoadError::Unexpected(err.to_string()))?;
        let value = match ty {
            attribute_proto::AttributeType::Float => Ok(Attribute::Float(attr.f)),
            attribute_proto::AttributeType::Int => Ok(Attribute::Int(attr.i)),
            attribute_proto::AttributeType::Ints => Ok(Attribute::Ints(attr.ints)),
            attribute_proto::AttributeType::String => load_utf8(attr.s).map(Attribute::Str),
            attribute_proto::AttributeType::Strings => attr
                .strings
                .into_iter()
                .map(load_utf8)
                .collect::<Result<Vec<_>, _>>()
                .map(Attribute::Strings),
            attribute_proto::AttributeType::Tensor => {
                load_tensor(attr.t.unwrap(), base_dir).map(Attribute::Tensor)
            }
            x => Err(ModelLoadError::UnsupportedAttributeType(x)),
        }?;
        res.insert(name, value);
    }
    Ok(res)
}

trait OptionalVecExt<T: Clone + Copy + PartialEq> {
    fn with_default(self, default: T) -> OptionalVec<T>;
}

impl<T: Clone + Copy + PartialEq> OptionalVecExt<T> for Option<Vec<T>> {
    fn with_default(self, default: T) -> OptionalVec<T> {
        OptionalVec::new(self, default)
    }
}

trait RequiredAttr {
    fn required(&self, name: &str) -> LoadResult<&Attribute>;
}

impl RequiredAttr for Attributes {
    fn required(&self, name: &str) -> LoadResult<&Attribute> {
        self.get(name)
            .ok_or(ModelLoadError::Required(name.to_string()))
    }
}

impl BatchNormalization {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let epsilon = attributes
            .get("epsilon")
            .map(|x| x.f())
            .transpose()?
            .unwrap_or(1e-5);
        let momentum = attributes
            .get("momentum")
            .map(|x| x.f())
            .transpose()?
            .unwrap_or(0.9);
        Ok(BatchNormalization { epsilon, momentum })
    }
}

impl Attention {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let is_causal = attributes
            .get("is_causal")
            .map(|x| x.b())
            .transpose()?
            .unwrap_or(false);
        let scale = attributes.get("scale").map(|x| x.f()).transpose()?.unwrap();
        Ok(Self { is_causal, scale })
    }
}

impl Cast {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        // TODO: saturate
        let to = attributes.required("to")?.ty()?;
        Ok(Cast { to })
    }
}

impl Concat {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axis = attributes.required("axis")?.index()?;
        Ok(Concat { axis })
    }
}

impl DequantizeLinear {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axis = attributes
            .get("axis")
            .map(|a| a.index())
            .transpose()?
            .unwrap_or(TensorIndex::new(1));
        Ok(DequantizeLinear { axis })
    }
}

impl DequantMatMul {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axis = attributes
            .get("axis")
            .map(|a| a.index())
            .transpose()?
            .unwrap_or(TensorIndex::new(0));
        Ok(DequantMatMul { axis })
    }
}

impl Constant {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let value = attributes.required("value")?.tensor()?;
        Ok(Constant { value })
    }
}

impl ConstantOfShape {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let value = attributes
            .get("value")
            .map(|x| x.tensor())
            .transpose()?
            .and_then(|tensor| tensor.data.try_into().ok())
            .unwrap_or(ScalarData::Float(FloatType::F32, 0.0));
        Ok(ConstantOfShape { value })
    }
}

impl ConvPad {
    fn load(attrs: &Attributes) -> LoadResult<Self> {
        let auto_pad = attrs.get("auto_pad").map_or(Ok("NOTSET"), |x| x.s())?;
        let pads = attrs
            .get("pads")
            .map(|x| x.ints::<usize>())
            .transpose()?
            .map(|v| {
                if v.len() % 2 != 0 {
                    Err(ModelLoadError::Unexpected("Invalid pads".to_string()))
                } else {
                    let half = v.len() / 2;
                    let mut res = Vec::with_capacity(half);
                    for i in 0..half {
                        res.push((v[i], v[i + half]));
                    }
                    Ok(res)
                }
            })
            .transpose()?;
        let pad = match (auto_pad, pads) {
            ("NOTSET", pads) => Self::NotSet(OptionalVec::new(pads, (0, 0))),
            ("SAME_UPPER", None) => Self::SameUpper,
            ("SAME_LOWER", None) => Self::SameLower,
            ("VALID", None) => Self::Valid,
            _ => return Err(ModelLoadError::Unexpected("Invalid padding".to_string())),
        };
        Ok(pad)
    }
}

impl Conv {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let dilations = attributes
            .get("dilations")
            .map(|x| x.ints())
            .transpose()?
            .with_default(1);
        let group = attributes.get("group").map_or(Ok(1), |x| x.i())?;
        let kernel_shape = attributes
            .get("kernel_shape")
            .ok_or(ModelLoadError::Unexpected(
                "TODO: support inference of kernel_shape".to_string(),
            ))?
            .ints()?
            .as_slice()
            .into();
        let strides = attributes
            .get("strides")
            .map(|x| x.ints())
            .transpose()?
            .with_default(1);
        let pad = ConvPad::load(attributes)?;
        Ok(Conv {
            pad,
            dilations,
            group: group as usize,
            kernel_shape,
            strides,
            input_layout: Layout::NCHW,
            output_layout: Layout::NCHW,
            activation: Activation::default(),
        })
    }
}

impl Gather {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axis = attributes
            .get("axis")
            .map(|x| x.index())
            .transpose()?
            .unwrap_or(TensorIndex::new(0));
        Ok(Gather { axis })
    }
}

impl GeLU {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let approximate = attributes
            .get("approximate")
            .map(|x| x.s())
            .transpose()?
            .unwrap_or("none");
        let approximate = match approximate {
            "none" => false,
            "tanh" => true,
            _ => {
                return Err(ModelLoadError::Unexpected(
                    "Invalid approximate".to_string(),
                ))
            }
        };
        Ok(GeLU { approximate })
    }
}

impl Gemm {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let trans_a = attributes
            .get("transA")
            .map(|x| x.b())
            .transpose()?
            .unwrap_or(false);
        let trans_b = attributes
            .get("transB")
            .map(|x| x.b())
            .transpose()?
            .unwrap_or(false);
        let alpha = attributes
            .get("alpha")
            .map(|x| x.f())
            .transpose()?
            .unwrap_or(1.0)
            .into();
        let beta = attributes
            .get("beta")
            .map(|x| x.f())
            .transpose()?
            .unwrap_or(1.0)
            .into();
        Ok(Gemm {
            trans_a,
            trans_b,
            alpha,
            beta,
        })
    }
}

impl LayerNormalization {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axis = attributes
            .get("axis")
            .map(|x| x.index())
            .transpose()?
            .unwrap_or(TensorIndex::new(-1));
        let epsilon = attributes
            .get("epsilon")
            .map(|x| x.f())
            .transpose()?
            .unwrap_or(1e-5)
            .into();
        Ok(LayerNormalization { axis, epsilon })
    }
}

impl LeakyReLU {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let alpha = attributes
            .get("alpha")
            .map(|x| x.f())
            .transpose()?
            .unwrap_or(0.01)
            .into();
        Ok(LeakyReLU { alpha })
    }
}

impl Pooling {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let dilations = attributes
            .get("dilations")
            .map(|x| x.ints())
            .transpose()?
            .with_default(1);
        let ceil_mode = attributes
            .get("ceil_mode")
            .map(|x| x.b())
            .transpose()?
            .unwrap_or(false);
        let kernel_shape = attributes
            .required("kernel_shape")?
            .ints()?
            .as_slice()
            .into();
        let strides = attributes
            .get("strides")
            .map(|x| x.ints())
            .transpose()?
            .with_default(1);
        let pad = ConvPad::load(attributes)?;
        Ok(Pooling {
            pad,
            ceil_mode,
            dilations,
            kernel_shape,
            strides,
            layout: Layout::NCHW,
        })
    }
}

impl OneHot {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axis = attributes
            .get("axis")
            .map(|x| x.i())
            .transpose()?
            .unwrap_or(-1) as isize;
        Ok(OneHot {
            axis,

            depth: None,
            on_value: None,
            off_value: None,
        })
    }
}

impl Reduce {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let keepdims = attributes
            .get("keepdims")
            .map(|x| x.b())
            .transpose()?
            .unwrap_or(true);
        let axes = attributes
            .get("axes")
            .map(|x| x.ints())
            .unwrap_or(Ok(Vec::new()))?;
        Ok(Self { keepdims, axes })
    }
}

impl ResizeNearestMode {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let nearest_mode = attributes.get("nearest_mode").map(|x| x.s()).transpose()?;
        let nearest_mode = match nearest_mode {
            Some("round_prefer_floor") | None => Self::RoundPreferFloor,
            Some("round_prefer_ceil") => Self::RoundPreferCeil,
            Some("floor") => Self::Floor,
            Some("ceil") => Self::Ceil,
            Some(_) => {
                return Err(ModelLoadError::Unexpected(
                    "Invalid nearest_mode".to_string(),
                ))
            }
        };
        Ok(nearest_mode)
    }
}

impl ResizeMode {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let mode = attributes
            .get("mode")
            .map(|x| x.s())
            .transpose()?
            .unwrap_or("nearest");
        match mode {
            "nearest" => Ok(Self::Nearest(ResizeNearestMode::load(attributes)?)),
            _ => unimplemented!(),
        }
    }
}

impl ResizeCoordinateTransformationMode {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let mode = attributes
            .get("coordinate_transformation_mode")
            .map(|x| x.s())
            .transpose()?;
        let mode = match mode {
            Some("half_pixel") | None => Self::HalfPixel,
            Some("pytorch_half_pixel") | Some("align_corners") | Some("asymmetric") => {
                unimplemented!()
            }
            Some(_) => {
                return Err(ModelLoadError::Unexpected(
                    "Invalid coordinate_transformation_mode".to_string(),
                ))
            }
        };
        Ok(mode)
    }
}

impl ResizeKeepAspectRatioPolicy {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let policy = attributes
            .get("keep_aspect_ratio")
            .map(|x| x.s())
            .transpose()?;
        let policy = match policy {
            Some("stretch") | None => Self::Stretch,
            Some("not_larger") => Self::NotLarger,
            Some("not_smaller") => Self::NotSmaller,
            Some(_) => {
                return Err(ModelLoadError::Unexpected(
                    "Invalid keep_aspect_ratio".to_string(),
                ))
            }
        };
        Ok(policy)
    }
}

impl Resize {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axes = attributes.get("axes").map(|x| x.indexes()).transpose()?;
        let coordinate_transformation_mode = ResizeCoordinateTransformationMode::load(attributes)?;
        let keep_aspect_ratio_policy = ResizeKeepAspectRatioPolicy::load(attributes)?;
        let mode = ResizeMode::load(attributes)?;
        Ok(Self {
            axes,
            coordinate_transformation_mode,
            keep_aspect_ratio_policy,
            mode,

            scale: None,
        })
    }
}

impl Shape {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let start = attributes
            .get("start")
            .map(|x| x.index())
            .transpose()?
            .unwrap_or(TensorIndex::new(0));
        let end = attributes.get("end").map(|x| x.index()).transpose()?;
        Ok(Shape { start, end })
    }
}

impl Softmax {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axis = attributes
            .get("axis")
            .map(|x| x.index())
            .transpose()?
            .unwrap_or(TensorIndex::new(-1));
        Ok(Softmax { axis })
    }
}

impl SplitOutputs {
    // TODO: Support version 18 (num_outputs).
    // TODO: When `split` is given as an input, not as an attribute.
    fn load(attributes: &Attributes) -> LoadResult<Option<Self>> {
        let split = attributes
            .get("split")
            .map(|x| x.ints())
            .transpose()?
            .map(|x| Self::Split(x.into_iter().collect()));
        Ok(split)
    }
}

impl Split {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axis = attributes
            .get("axis")
            .map(|x| x.index())
            .transpose()?
            .unwrap_or(TensorIndex::new(0));
        let outputs = SplitOutputs::load(attributes)?;
        Ok(Split { axis, outputs })
    }
}

impl Squeeze {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axes = attributes.get("axes").map(|x| x.indexes()).transpose()?;
        Ok(Self { axes })
    }
}

impl Swish {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let alpha = attributes
            .get("alpha")
            .map(|x| x.f())
            .transpose()?
            .unwrap_or(1.0);
        Ok(Self { alpha })
    }
}

impl Transpose {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let perm = attributes.get("perm").map(|x| x.ints()).transpose()?;
        Ok(Transpose { perm })
    }
}

impl Unsqueeze {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axes = attributes
            .get("axes")
            .map(|x| x.indexes())
            .transpose()?
            .unwrap_or_default();
        Ok(Unsqueeze { axes })
    }
}

fn load_op(op: &str, attributes: &Attributes) -> LoadResult<Operator> {
    match op {
        "Add" => Ok(Operator::Add),
        "And" => Ok(Operator::And),
        "Attention" => Ok(Operator::Attention(Attention::load(attributes)?)),
        "BatchNormalization" => Ok(Operator::BatchNormalization(BatchNormalization::load(
            attributes,
        )?)),
        "Cast" => Ok(Operator::Cast(Cast::load(attributes)?)),
        "Clip" => Ok(Operator::Clip(Clip {
            min: None,
            max: None,
        })),
        "Concat" => Ok(Operator::Concat(Concat::load(attributes)?)),
        "Constant" => Ok(Operator::Constant(Constant::load(attributes)?)),
        "ConstantOfShape" => Ok(Operator::ConstantOfShape(ConstantOfShape::load(
            attributes,
        )?)),
        "AveragePool" => Ok(Operator::AveragePool(Pooling::load(attributes)?)),
        "Conv" => Ok(Operator::Conv(Conv::load(attributes)?)),
        "Cos" => Ok(Operator::Cos),
        "DequantizeLinear" => Ok(Operator::DequantizeLinear(DequantizeLinear::load(
            attributes,
        )?)),
        "DequantMatMul" => Ok(Operator::DequantMatMul(DequantMatMul::load(attributes)?)),
        "Div" => Ok(Operator::Div),
        "Equal" => Ok(Operator::Equal),
        "Expand" => Ok(Operator::Expand),
        "Exp" => Ok(Operator::Exp),
        "Flatten" => {
            let axis = attributes
                .get("axis")
                .map(|x| x.i())
                .transpose()?
                .unwrap_or(1);
            Ok(Operator::Flatten(Flatten {
                axis: TensorIndex::new(axis as isize),
            }))
        }
        "Gather" => Ok(Operator::Gather(Gather::load(attributes)?)),
        "Gelu" => Ok(Operator::GeLU(GeLU::load(attributes)?)),
        "Gemm" => Ok(Operator::Gemm(Gemm::load(attributes)?)),
        "GlobalAveragePool" => Ok(Operator::GlobalAveragePool),
        "LayerNormalization" => Ok(Operator::LayerNormalization(LayerNormalization::load(
            attributes,
        )?)),
        "LeakyRelu" => Ok(Operator::LeakyReLU(LeakyReLU::load(attributes)?)),
        "LessOrEqual" => Ok(Operator::LessOrEqual),
        "Log" => Ok(Operator::Log),
        "Identity" => Ok(Operator::Identity),
        "IsNaN" => Ok(Operator::IsNaN),
        "MatMul" => Ok(Operator::MatMul),
        "MaxPool" => Ok(Operator::MaxPool(Pooling::load(attributes)?)),
        "Mul" => Ok(Operator::Mul),
        "Neg" => Ok(Operator::Neg),
        "NonZero" => Ok(Operator::NonZero),
        "OneHot" => Ok(Operator::OneHot(OneHot::load(attributes)?)),
        "Pow" => Ok(Operator::Pow),
        "Range" => Ok(Operator::Range),
        "Reciprocal" => Ok(Operator::Reciprocal),
        "ReduceMax" => Ok(Operator::ReduceMax(Reduce::load(attributes)?)),
        "ReduceMean" => Ok(Operator::ReduceMean(Reduce::load(attributes)?)),
        "ReduceSum" => Ok(Operator::ReduceSum(Reduce::load(attributes)?)),
        "Relu" => Ok(Operator::ReLU),
        "Reshape" => Ok(Operator::Reshape),
        "Resize" => Ok(Operator::Resize(Resize::load(attributes)?)),
        "Sigmoid" => Ok(Operator::Sigmoid),
        "Sin" => Ok(Operator::Sin),
        "Shape" => Ok(Operator::Shape(Shape::load(attributes)?)),
        "Slice" => Ok(Operator::Slice),
        "Softmax" => Ok(Operator::Softmax(Softmax::load(attributes)?)),
        "Split" => Ok(Operator::Split(Split::load(attributes)?)),
        "Sqrt" => Ok(Operator::Sqrt),
        "Squeeze" => Ok(Operator::Squeeze(Squeeze::load(attributes)?)),
        "Sub" => Ok(Operator::Sub),
        "Swish" => Ok(Operator::Swish(Swish::load(attributes)?)),
        "Tanh" => Ok(Operator::Tanh),
        "Transpose" => Ok(Operator::Transpose(Transpose::load(attributes)?)),
        "Unsqueeze" => Ok(Operator::Unsqueeze(Unsqueeze::load(attributes)?)),
        "Where" => Ok(Operator::Where),

        // Custom
        "KVCacheUpdate" => Ok(Operator::KVCacheUpdate),
        "QuantizingKVCacheUpdate" => Ok(Operator::QuantizingKVCacheUpdate),
        x => Err(ModelLoadError::UnsupportedOp(x.to_string())),
    }
}
