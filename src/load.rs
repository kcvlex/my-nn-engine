use crate::model::{Graph, Model, Node, Nodes, ValueId, ValueInfo, Values};
use crate::operator::*;
use crate::tensor::{
    dimensions::{Dimension, UnresolvedTensorDims},
    resolved_dimensions::ResolvedTensorDims,
    tensor::{DataType, Tensor, TensorData, TensorType, TypeError, UnresolvedTensorType},
};
use prost::{DecodeError, Message};
use std::collections::HashMap;
use std::path::Path;
include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

type TensorDataTypeProto = tensor_proto::DataType;

#[derive(Debug)]
pub enum ModelLoadError {
    FileRead(std::io::Error),
    Decode(DecodeError),
    ElemTypeUnspecified,
    NoGraph,
    UnsupportedElemType(TensorDataTypeProto),
    UnsupportedValueType(type_proto::Value),
    UnsupportedAttributeType(attribute_proto::AttributeType),
    UnsupportedOp(String),
    NegativeDimension(i64),
    TypeError(TypeError),
    Unexpected(String),
}

type LoadResult<T> = Result<T, ModelLoadError>;

impl Model {
    pub fn load_from_path<P: AsRef<Path>>(p: P) -> LoadResult<Model> {
        let model = std::fs::read(p).map_err(ModelLoadError::FileRead)?;
        let model = ModelProto::decode(&*model).map_err(ModelLoadError::Decode)?;
        let graph = model.graph.ok_or(ModelLoadError::NoGraph)?;
        let graph = GraphLoader::default().load_graph(graph)?;
        Ok(Model { graph })
    }
}

#[derive(Default)]
struct GraphLoader {
    entries: HashMap<String, ValueId>,
    values: Values,
}

#[allow(dead_code)]
#[derive(Debug)]
enum Attribute {
    Float(f32),
    Int(i64),
    Ints(Vec<i64>),
    Str(String),
}

impl Attribute {
    // fn float(&self) -> LoadResult<f32> {
    //     match self {
    //         Attribute::Float(x) => Ok(*x),
    //         x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
    //     }
    // }

    fn int(&self) -> LoadResult<i64> {
        match self {
            Attribute::Int(x) => Ok(*x),
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
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

    fn str(&self) -> LoadResult<&str> {
        match self {
            Attribute::Str(x) => Ok(x),
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
    }
}

type Attributes = HashMap<String, Attribute>;

impl GraphLoader {
    fn load_graph(mut self, graph: GraphProto) -> LoadResult<Graph> {
        let initializer = self.load_initializer(graph.initializer)?;
        let inputs = self.load_value_info_vec(graph.input)?;
        let outputs = self.load_value_info_vec(graph.output)?;
        let mut nodes = self.load_nodes(graph.node)?;
        let inputs = {
            let mut res = Vec::with_capacity(inputs.len());
            for &x in inputs.iter() {
                let node = Node {
                    name: self.values[x].name.clone(),
                    inputs: Vec::new(),
                    outputs: vec![x],
                    op: Operator::Input(x),
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
                    inputs: vec![x],
                    outputs: Vec::new(),
                    op: Operator::Output(x),
                };
                nodes.alloc(node)
            })
            .collect();
        Ok(Graph {
            name: graph.name,
            initializer,
            inputs,
            outputs,
            values: self.values,
            nodes,
        })
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

    fn load_initializer(&mut self, v: Vec<TensorProto>) -> LoadResult<HashMap<ValueId, Tensor>> {
        let mut res = HashMap::new();
        for tensor in v.into_iter() {
            let name = tensor.name.clone();
            let tensor = load_tensor(tensor)?;
            let id = self.entries.entry(name.clone()).or_insert_with(|| {
                self.values.alloc(ValueInfo {
                    name,
                    ty: Some(tensor.tensor_type()),
                })
            });
            res.insert(*id, tensor);
        }
        Ok(res)
    }

    fn load_nodes(&mut self, nodes: Vec<NodeProto>) -> LoadResult<Nodes> {
        macro_rules! io {
            ($v: expr) => {{
                $v.into_iter()
                    .map(|x| {
                        *self.entries.entry(x.clone()).or_insert_with(|| {
                            self.values.alloc(ValueInfo {
                                name: x.clone(),
                                ty: None,
                            })
                        })
                    })
                    .collect()
            }};
        }
        let mut res = Nodes::default();
        for node in nodes.into_iter() {
            let name = node.name;
            let inputs = io!(node.input);
            let outputs = io!(node.output);
            let attributes = load_attributes(node.attribute)?;
            let op = load_op(&node.op_type, &attributes)?;
            res.alloc(Node {
                name,
                inputs,
                outputs,
                op,
            });
        }
        Ok(res)
    }
}

fn load_tensor(tensor: TensorProto) -> LoadResult<Tensor> {
    let elem_type = DataType::try_from(tensor.data_type)?;
    let data = if tensor.raw_data.is_empty() {
        match elem_type {
            DataType::I64 => TensorData::I64(tensor.int64_data),
            DataType::F32 => TensorData::F32(tensor.float_data),
            DataType::F64 => TensorData::F64(tensor.double_data),
        }
    } else {
        TensorData::from_bytes(elem_type, tensor.raw_data.as_slice())
    };
    let mut dims = Vec::new();
    for dim in tensor.dims.into_iter() {
        let dim = usize::try_from(dim).map_err(|_| ModelLoadError::NegativeDimension(dim))?;
        dims.push(dim);
    }
    let dims = ResolvedTensorDims::new(dims);
    Tensor::new(dims, data).map_err(ModelLoadError::TypeError)
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
            dims: Some(UnresolvedTensorDims::new(shape)),
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
            TensorDataTypeProto::Float => Ok(DataType::F32),
            TensorDataTypeProto::Double => Ok(DataType::F64),
            TensorDataTypeProto::Int64 => Ok(DataType::I64),
            TensorDataTypeProto::Undefined => Err(ModelLoadError::ElemTypeUnspecified),
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
            tensor_shape_proto::dimension::Value::DimParam(x) => Ok(Dimension::Param(x)),
        }
    }
}

fn load_attributes(v: Vec<AttributeProto>) -> LoadResult<Attributes> {
    let mut res = HashMap::new();
    for attr in v.into_iter() {
        let name = attr.name;
        let ty = attribute_proto::AttributeType::try_from(attr.r#type)
            .map_err(|err| ModelLoadError::Unexpected(err.to_string()))?;
        let value = match ty {
            attribute_proto::AttributeType::Float => Ok(Attribute::Float(attr.f)),
            attribute_proto::AttributeType::Int => Ok(Attribute::Int(attr.i)),
            attribute_proto::AttributeType::Ints => Ok(Attribute::Ints(attr.ints)),
            attribute_proto::AttributeType::String => String::from_utf8(attr.s)
                .map(Attribute::Str)
                .map_err(|err| ModelLoadError::Unexpected(err.to_string())),
            x => Err(ModelLoadError::UnsupportedAttributeType(x)),
        }?;
        res.insert(name, value);
    }
    Ok(res)
}

fn load_pad(attrs: &Attributes) -> LoadResult<ConvPad> {
    let auto_pad = attrs.get("auto_pad").map_or(Ok("NOTSET"), |x| x.str())?;
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
        ("NOTSET", pads) => ConvPad::NotSet(OptionalVec::new(pads, (0, 0))),
        ("SAME_UPPER", None) => ConvPad::SameUpper,
        ("SAME_LOWER", None) => ConvPad::SameLower,
        ("VALID", None) => ConvPad::Valid,
        _ => return Err(ModelLoadError::Unexpected("Invalid padding".to_string())),
    };
    Ok(pad)
}

trait OptionalVecExt<T: Clone + Copy> {
    fn with_default(self, default: T) -> OptionalVec<T>;
}

impl<T: Clone + Copy> OptionalVecExt<T> for Option<Vec<T>> {
    fn with_default(self, default: T) -> OptionalVec<T> {
        OptionalVec::new(self, default)
    }
}

fn load_op(op: &str, attributes: &Attributes) -> LoadResult<Operator> {
    match op {
        "Add" => Ok(Operator::Add),
        "Relu" => Ok(Operator::ReLU),
        "MatMul" => Ok(Operator::MatMul),
        "Reshape" => Ok(Operator::Reshape),
        "Transpose" => {
            let perm = attributes
                .get("perm")
                .map(|x| x.ints())
                .unwrap_or(Ok(Vec::new()))?;
            Ok(Operator::Transpose(perm))
        }
        "Conv" => {
            let dilations = attributes
                .get("dilations")
                .map(|x| x.ints())
                .transpose()?
                .with_default(1);
            let groups = attributes.get("groups").map_or(Ok(1), |x| x.int())?;
            let kernel_shape = attributes
                .get("kernel_shape")
                .ok_or(ModelLoadError::Unexpected(
                    "TODO: support inference of kernel_shape".to_string(),
                ))?
                .ints()?
                .into();
            let strides = attributes
                .get("strides")
                .map(|x| x.ints())
                .transpose()?
                .with_default(1);
            let pad = load_pad(attributes)?;
            Ok(Operator::Conv(Conv {
                pad,
                dilations,
                groups: groups as usize,
                kernel_shape,
                strides,
            }))
        }
        "MaxPool" => {
            let dilations = attributes
                .get("dilations")
                .map(|x| x.ints())
                .transpose()?
                .with_default(1);
            let ceil_mode = attributes
                .get("ceil_mode")
                .map(|x| x.int())
                .transpose()?
                .map_or(false, |x| x != 0);
            let kernel_shape = attributes
                .get("kernel_shape")
                .ok_or(ModelLoadError::Unexpected(
                    "kernel_shape is required".to_string(),
                ))?
                .ints()?
                .into();
            let strides = attributes
                .get("strides")
                .map(|x| x.ints())
                .transpose()?
                .with_default(1);
            let pad = load_pad(attributes)?;
            Ok(Operator::MaxPool(Pooling {
                pad,
                ceil_mode,
                dilations,
                kernel_shape,
                strides,
            }))
        }
        x => Err(ModelLoadError::UnsupportedOp(x.to_string())),
    }
}
