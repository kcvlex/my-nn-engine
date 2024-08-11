use std::ops::{Index, IndexMut};

#[derive(Debug)]
pub enum DataType {
    F32,
    F64,
}

// TODO: Complex
#[derive(Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[derive(Debug)]
pub struct Tensor {
    pub dims: TensorShape,
    pub data: TensorData,
}

#[derive(Debug, Clone)]
pub enum Dimension {
    Const(i64),
    Param(String),
}

#[derive(Debug, Clone)]
pub struct TensorShape(Vec<Dimension>);

impl Tensor {
    pub fn elem_type(&self) -> DataType {
        match self.data {
            TensorData::F32(_) => DataType::F32,
            TensorData::F64(_) => DataType::F64,
        }
    }

    pub fn tensor_type(&self) -> TensorType {
        TensorType {
            elem_type: self.elem_type(),
            dims: Some(self.dims.clone()),
        }
    }
}

impl TensorShape {
    pub fn new(v: Vec<Dimension>) -> Self {
        Self(v)
    }
}

#[derive(Debug)]
pub struct TensorType {
    pub elem_type: DataType,
    pub dims: Option<TensorShape>,
}

impl Index<usize> for TensorShape {
    type Output = Dimension;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for TensorShape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl TensorData {
    pub fn from_raw_data(ty: DataType, raw: Vec<u8>) -> Self {
        todo!("impl")
    }
}
