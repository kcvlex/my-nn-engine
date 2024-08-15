use crate::tensor::dimensions::{Dimension, TensorDims};
use crate::tensor::resolved_dimensions::ResolvedTensorDims;

#[derive(Debug, Clone)]
pub enum TypeError {
    InvalidShape(usize, ResolvedTensorDims),
    BroadcastError(ResolvedTensorDims, ResolvedTensorDims),
    ReshapeError(ResolvedTensorDims, ResolvedTensorDims),
    InferError(String),
    UnresolvedInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    I64,
    F32,
    F64,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: TensorData,
    pub ty: ResolvedTensorType,
}

#[derive(Debug, Clone)]
pub struct TensorIndex(usize);

// TODO: Complex
#[derive(Debug, Clone)]
pub enum TensorData {
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl TensorData {
    pub fn size(&self) -> usize {
        match self {
            TensorData::I64(v) => v.len(),
            TensorData::F32(v) => v.len(),
            TensorData::F64(v) => v.len(),
        }
    }

    pub fn elem_type(&self) -> DataType {
        match self {
            TensorData::I64(_) => DataType::I64,
            TensorData::F32(_) => DataType::F32,
            TensorData::F64(_) => DataType::F64,
        }
    }
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
    stride: ResolvedTensorDims,
}

impl ResolvedTensorType {
    pub fn new(elem_type: DataType, dims: ResolvedTensorDims) -> Self {
        let stride = calc_stride_reshape(&dims);
        Self {
            elem_type,
            dims,
            stride,
        }
    }

    pub fn stride(&self, i: usize) -> usize {
        self.stride[i]
    }
}

impl TensorType {
    pub fn to_resolved(&self) -> Option<ResolvedTensorType> {
        let dims = self.dims.as_ref()?.to_resolved()?;
        Some(ResolvedTensorType {
            elem_type: self.elem_type.clone(),
            stride: calc_stride_reshape(&dims),
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

fn calc_stride_broadcast(
    orig: &ResolvedTensorDims,
    target: &ResolvedTensorDims,
) -> ResolvedTensorDims {
    let mut acc = 1;
    let mut stride = vec![0; target.ndim()];
    for i in (0..orig.ndim()).rev() {
        if orig[i] == target[i] {
            stride[i] = acc;
            acc *= orig[i];
        }
    }
    ResolvedTensorDims::new(stride)
}

fn calc_stride_reshape(dims: &ResolvedTensorDims) -> ResolvedTensorDims {
    let mut acc = 1;
    let mut stride = vec![0; dims.ndim()];
    for i in (0..dims.ndim()).rev() {
        stride[i] = acc;
        acc *= dims[i];
    }
    ResolvedTensorDims::new(stride)
}

impl Tensor {
    pub fn new(dims: ResolvedTensorDims, data: TensorData) -> Result<Self, TypeError> {
        if dims.size() != data.size() {
            return Err(TypeError::InvalidShape(data.size(), dims));
        }
        let ty = ResolvedTensorType::new(data.elem_type(), dims);
        Ok(Self { data, ty })
    }

    pub fn tensor_type(&self) -> TensorType {
        self.ty.clone().into()
    }

    pub fn get_index(&self, indexes: &[usize]) -> Option<TensorIndex> {
        if indexes.len() != self.ty.dims.ndim() {
            return None;
        }
        let mut index = 0;
        for i in 0..indexes.len() {
            index += indexes[i] * self.ty.stride[i];
        }
        Some(TensorIndex(index))
    }

    pub fn raw_data(&self) -> &[u8] {
        match &self.data {
            TensorData::I64(v) => unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * std::mem::size_of::<i64>(),
                )
            },
            TensorData::F32(v) => unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * std::mem::size_of::<f32>(),
                )
            },
            TensorData::F64(v) => unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * std::mem::size_of::<f64>(),
                )
            },
        }
    }
}

impl TensorData {
    pub fn from_raw_data(ty: DataType, raw: Vec<u8>) -> Self {
        todo!("impl")
    }
}
