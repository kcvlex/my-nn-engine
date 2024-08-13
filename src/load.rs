use crate::model::{Graph, Model, Node, Nodes, ValueId, ValueInfo, Values};
use crate::operator::Operator;
use crate::tensor::{
    dimensions::{Dimension, TensorDims},
    resolved_dimensions::ResolvedTensorDims,
    tensor::{DataType, Tensor, TensorData, TensorType, TypeError},
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
    UnsupportedOp(String),
    NegativeDimension(i64),
    TypeError(TypeError),
    Unexpected(String),
}

type LoadResult<T> = Result<T, ModelLoadError>;

pub fn load_from_path<P: AsRef<Path>>(p: P) -> LoadResult<Model> {
    let model = std::fs::read(p).map_err(ModelLoadError::FileRead)?;
    let model = ModelProto::decode(&*model).map_err(ModelLoadError::Decode)?;
    let graph = model.graph.ok_or(ModelLoadError::NoGraph)?;
    let graph = GraphLoader::default().load_graph(graph)?;
    Ok(Model { graph })
}

#[derive(Default)]
struct GraphLoader {
    entries: HashMap<String, ValueId>,
    values: Values,
}

impl GraphLoader {
    fn load_graph(mut self, graph: GraphProto) -> LoadResult<Graph> {
        let initializer = self.load_initializer(graph.initializer)?;
        let inputs = self.load_value_info_vec(graph.input)?;
        let outputs = self.load_value_info_vec(graph.output)?;
        let nodes = self.load_nodes(graph.node)?;
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
            let op = match node.op_type.as_str() {
                "Add" => Ok(Operator::Add),
                x => Err(ModelLoadError::UnsupportedOp(x.to_string())),
            }?;
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
            DataType::F32 => TensorData::F32(tensor.float_data),
            DataType::F64 => TensorData::F64(tensor.double_data),
        }
    } else {
        TensorData::from_raw_data(elem_type, tensor.raw_data)
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
    let dims = if let Some(dims) = tensor.shape {
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
        Ok(Some(shape))
    } else {
        Ok(None)
    }?
    .map(TensorDims::new);
    Ok(TensorType { elem_type, dims })
}

impl TryFrom<i32> for DataType {
    type Error = ModelLoadError;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let value = tensor_proto::DataType::try_from(value)
            .map_err(|e| ModelLoadError::Unexpected(format!("Invalid DataType: {:?}", e)))?;
        match value {
            TensorDataTypeProto::Float => Ok(DataType::F32),
            TensorDataTypeProto::Double => Ok(DataType::F64),
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
