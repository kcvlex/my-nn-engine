use crate::onnx::model::{Graph, Model, Node, NodeMeta, Nodes, ValueId, ValueInfo, Values};
use crate::onnx::operator::*;
use crate::tensor::{
    data::TensorData,
    dimensions::{Dimension, ResolvedTensorDims, UnresolvedTensorDims},
    types::{DataType, FloatType, SIntType, TensorType, TypeError, UIntType, UnresolvedTensorType},
    Tensor,
};
use itertools::Itertools;
use prost::{DecodeError, Message};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

impl Tensor {
    fn save(&self) -> TensorProto {
        let mut res = TensorProto::default();
        match &self.data {
            TensorData::SInt(SIntType::I32, buf) => {
                res.data_type = tensor_proto::DataType::Int32.into();
                res.int32_data = buf.iter().copied().map(|x| x.try_into().unwrap()).collect();
            },
            TensorData::SInt(SIntType::I64, buf) => {
                res.data_type = tensor_proto::DataType::Int64.into();
                res.int64_data = buf.to_vec();
            },
            TensorData::UInt(UIntType::U64, buf) => {
                res.data_type = tensor_proto::DataType::Uint64.into();
                res.uint64_data = buf.to_vec();
            },
            TensorData::Float(FloatType::F32, buf) => {
                res.data_type = tensor_proto::DataType::Float.into();
                res.float_data = buf.iter().copied().map(|x| x as f32).collect();
            },
            TensorData::Float(FloatType::F64, buf) => {
                res.data_type = tensor_proto::DataType::Double.into();
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
        let ty = match ty {
            DataType::SInt(SIntType::I32) => tensor_proto::DataType::Int32,
            DataType::SInt(SIntType::I64) => tensor_proto::DataType::Int64,
            DataType::UInt(UIntType::U64) => tensor_proto::DataType::Uint64,
            DataType::Float(FloatType::F32) => tensor_proto::DataType::Float,
            DataType::Float(FloatType::F64) => tensor_proto::DataType::Double,
        };
        let ty: i32 = ty.into();
        self.with_i(ty.into())
    }
}

impl BatchNormalization {
    fn to_proto(&self) -> Vec<AttributeProto> {
        vec![
            AttributeProto::default().with_name("epsilon").with_f(self.epsilon),
            AttributeProto::default().with_name("momentum").with_f(self.momentum),
        ]
    }
}

impl Cast {
    fn to_proto(&self) -> Vec<AttributeProto> {
        vec![
            AttributeProto::default().with_name("to").with_ty(self.to),
        ]
    }
}

impl Concat {
    fn to_proto(&self) -> Vec<AttributeProto> {
        vec![
            AttributeProto::default().with_name("axis").with_index(self.axis),
        ]
    }
}

impl NodeProto {
    fn write_op(&mut self, op: &str, attrs: Vec<AttributeProto>) {
        self.name = op.to_string();
        self.attribute = attrs;
    }
}

impl Operator {
    fn write(&self, dst: &mut NodeProto) {
        match &self {
            Operator::Add => dst.write_op("Add", vec![]),
            Operator::BatchNormalization(attrs) => dst.write_op("BatchNormalization", attrs.to_proto()),
            _ => todo!(),
        }
    }
}
