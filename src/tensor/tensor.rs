use crate::tensor::dimensions::{Dimension, TensorDims};
use crate::tensor::resolved_dimensions::ResolvedTensorDims;

#[derive(Debug, Clone)]
pub enum TypeError {
    BroadcastError(ResolvedTensorDims, ResolvedTensorDims),
    UnresolvedInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    F32,
    F64,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub dims: ResolvedTensorDims,
    pub data: TensorData,
}

// TODO: Complex
#[derive(Debug, Clone)]
pub enum TensorData {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct TensorType {
    pub elem_type: DataType,
    pub dims: Option<TensorDims>,
}

#[derive(Debug, Clone)]
pub struct ResolvedTensorType {
    pub elem_type: DataType,
    pub dims: ResolvedTensorDims,
}

impl TensorType {
    pub fn to_resolved(&self) -> Option<ResolvedTensorType> {
        let dims = self.dims.as_ref()?.to_resolved()?;
        Some(ResolvedTensorType {
            elem_type: self.elem_type.clone(),
            dims,
        })
    }
}

impl From<ResolvedTensorType> for TensorType {
    fn from(ty: ResolvedTensorType) -> Self {
        let dims = ty.dims.iter().map(|x| Dimension::Const(*x)).collect();
        let dims = Some(TensorDims::new(dims));
        Self {
            elem_type: ty.elem_type,
            dims,
        }
    }
}

impl Tensor {
    pub fn elem_type(&self) -> DataType {
        match self.data {
            TensorData::F32(_) => DataType::F32,
            TensorData::F64(_) => DataType::F64,
        }
    }

    pub fn tensor_type(&self) -> TensorType {
        let dims: Vec<_> = self.dims.iter().map(|x| Dimension::Const(*x)).collect();
        TensorType {
            elem_type: self.elem_type(),
            dims: Some(TensorDims::new(dims)),
        }
    }
}

impl TensorData {
    pub fn from_raw_data(ty: DataType, raw: Vec<u8>) -> Self {
        todo!("impl")
    }
}
