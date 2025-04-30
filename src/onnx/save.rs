use crate::onnx::model::*;
use crate::onnx::operator::*;
use crate::onnx::utils;
use crate::tensor::{
    data::TensorData,
    dimensions::Dimension,
    types::{
        DataType, FloatType, ResolvedTensorType, SIntType, TensorType, UIntType,
        UnresolvedTensorType,
    },
    Tensor,
};
use itertools::Itertools;
use prost::Message;
use std::collections::HashSet;
include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

// FIXME: This module is not tested at all!

impl From<SIntType> for tensor_proto::DataType {
    fn from(ty: SIntType) -> Self {
        match ty {
            SIntType::I32 => tensor_proto::DataType::Int32,
            SIntType::I64 => tensor_proto::DataType::Int64,
        }
    }
}

impl From<UIntType> for tensor_proto::DataType {
    fn from(ty: UIntType) -> Self {
        match ty {
            UIntType::U64 => tensor_proto::DataType::Uint64,
        }
    }
}

impl From<FloatType> for tensor_proto::DataType {
    fn from(ty: FloatType) -> Self {
        match ty {
            FloatType::F32 => tensor_proto::DataType::Float,
            FloatType::F64 => tensor_proto::DataType::Double,
        }
    }
}

impl From<DataType> for tensor_proto::DataType {
    fn from(ty: DataType) -> Self {
        match ty {
            DataType::SInt(s) => s.into(),
            DataType::UInt(u) => u.into(),
            DataType::Float(f) => f.into(),
        }
    }
}

impl From<DataType> for i32 {
    fn from(ty: DataType) -> Self {
        tensor_proto::DataType::from(ty).into()
    }
}

impl Tensor {
    fn to_proto(&self) -> TensorProto {
        let mut res = TensorProto {
            data_type: self.data.elem_type().into(),
            ..Default::default()
        };
        match &self.data {
            TensorData::SInt(SIntType::I32, buf) => {
                res.int32_data = buf.iter().copied().map(|x| x.try_into().unwrap()).collect();
            }
            TensorData::SInt(SIntType::I64, buf) => {
                res.int64_data = buf.to_vec();
            }
            TensorData::UInt(UIntType::U64, buf) => {
                res.uint64_data = buf.to_vec();
            }
            TensorData::Float(FloatType::F32, buf) => {
                res.float_data = buf.iter().copied().map(|x| x as f32).collect();
            }
            TensorData::Float(FloatType::F64, buf) => {
                res.double_data = buf.to_vec();
            }
        }
        res.dims = self.dims.iter().map(|x| *x as i64).collect();
        res
    }
}

impl AttributeProto {
    fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    fn with_f(mut self, f: f32) -> Self {
        self.r#type = attribute_proto::AttributeType::Float.into();
        self.f = f;
        self
    }

    fn with_i(mut self, i: i64) -> Self {
        self.r#type = attribute_proto::AttributeType::Int.into();
        self.i = i;
        self
    }

    fn with_index(self, index: TensorIndex) -> Self {
        self.with_i(index.raw() as i64)
    }

    fn with_s(mut self, s: &str) -> Self {
        self.r#type = attribute_proto::AttributeType::String.into();
        self.s = s.as_bytes().to_vec();
        self
    }

    fn with_ints(mut self, ints: &[i64]) -> Self {
        self.r#type = attribute_proto::AttributeType::Ints.into();
        self.ints = ints.to_vec();
        self
    }

    fn with_indexes(mut self, indexes: &[TensorIndex]) -> Self {
        self.r#type = attribute_proto::AttributeType::Ints.into();
        self.ints = indexes.iter().map(|x| x.raw() as i64).collect();
        self
    }

    fn with_ty(self, ty: DataType) -> Self {
        let ty: tensor_proto::DataType = ty.into();
        let ty: i32 = ty.into();
        self.with_i(ty.into())
    }
}

impl BatchNormalization {
    fn to_proto(&self) -> Vec<AttributeProto> {
        vec![
            AttributeProto::default()
                .with_name("epsilon")
                .with_f(self.epsilon),
            AttributeProto::default()
                .with_name("momentum")
                .with_f(self.momentum),
        ]
    }
}

impl Cast {
    fn to_proto(&self) -> Vec<AttributeProto> {
        vec![AttributeProto::default().with_name("to").with_ty(self.to)]
    }
}

impl Concat {
    fn to_proto(&self) -> Vec<AttributeProto> {
        vec![AttributeProto::default()
            .with_name("axis")
            .with_index(self.axis)]
    }
}

impl Node {
    fn to_proto(&self, values: &Values) -> NodeProto {
        let (opname, attrs) = match &self.op {
            Operator::Add => ("Add", vec![]),
            Operator::BatchNormalization(attrs) => ("BatchNormalization", attrs.to_proto()),
            _ => todo!(),
        };
        NodeProto {
            input: self
                .inputs
                .iter()
                .map(|x| values[*x].name.clone())
                .collect(),
            output: self
                .outputs
                .iter()
                .map(|x| values[*x].name.clone())
                .collect(),
            op_type: opname.to_string(),
            attribute: attrs,
            ..Default::default()
        }
    }
}

impl From<usize> for tensor_shape_proto::dimension::Value {
    fn from(dim: usize) -> Self {
        Self::DimValue(dim as i64)
    }
}

impl From<Dimension> for tensor_shape_proto::dimension::Value {
    fn from(dim: Dimension) -> Self {
        match dim {
            Dimension::Const(v) => Self::DimValue(v as i64),
            Dimension::Param(v) => Self::DimParam(v.to_string()),
        }
    }
}

impl<T: Into<tensor_shape_proto::dimension::Value> + Clone> From<&[T]> for TensorShapeProto {
    fn from(dims: &[T]) -> Self {
        Self {
            dim: dims
                .iter()
                .map(|x| tensor_shape_proto::Dimension {
                    value: Some(x.clone().into()),
                    ..Default::default()
                })
                .collect(),
        }
    }
}

impl TensorType {
    fn to_proto(&self) -> type_proto::Tensor {
        let mut res = type_proto::Tensor::default();
        let (elem_type, shape): (_, TensorShapeProto) = match self {
            TensorType::Unresolved(UnresolvedTensorType { elem_type, dims }) => {
                (elem_type, dims.as_ref().unwrap().inner().as_slice().into())
            }
            TensorType::Resolved(ResolvedTensorType {
                elem_type, dims, ..
            }) => (elem_type, dims.inner().as_slice().into()),
        };
        res.elem_type = (*elem_type).into();
        res.shape = Some(shape);
        res
    }
}

impl ValueInfo {
    fn to_proto(&self) -> ValueInfoProto {
        ValueInfoProto {
            name: self.name.clone(),
            r#type: self.ty.as_ref().map(|ty| TypeProto {
                value: Some(type_proto::Value::TensorType(ty.to_proto())),
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}

impl Graph {
    fn to_proto(&self) -> GraphProto {
        let mut dummy = HashSet::new();

        macro_rules! io {
            ($iter: expr) => {{
                $iter
                    .inspect(|v| {
                        dummy.insert(*v);
                    })
                    .map(|v| &self.values[v])
                    .sorted_by_key(|x| &x.name)
                    .map(|v| v.to_proto())
                    .collect()
            }};
        }

        let name = self.name.clone();
        let input = io!(self.inputs.iter().map(|x| match self.nodes[*x].op {
            Operator::Input(v) => v,
            _ => unreachable!(),
        }));
        let output = io!(self.outputs.iter().map(|x| match self.nodes[*x].op {
            Operator::Output(v) => v,
            _ => unreachable!(),
        }));
        let initializer = self
            .initializer
            .iter()
            .filter(|(k, _)| !dummy.contains(k))
            .sorted_by_key(|(k, _)| *k)
            .map(|(_, v)| v.to_proto())
            .collect();

        let node: Vec<_> = utils::simple_topological_order(self)
            .iter()
            .map(|x| self.nodes[*x].to_proto(&self.values))
            .collect();

        assert!(node.len() == self.nodes.iter().filter(|(_, v)| !v.is_dummy()).count());

        GraphProto {
            name,
            input,
            output,
            initializer,
            node,
            ..Default::default()
        }
    }
}

impl Model {
    pub fn to_proto(&self) -> ModelProto {
        ModelProto {
            graph: Some(self.graph.to_proto()),
            ..Default::default()
        }
    }
}
