use crate::tensor::resolved_dimensions::ResolvedTensorDims;
use std::ops::Index;

#[derive(Debug, Clone)]
pub struct TensorDims(Vec<Dimension>);

#[derive(Debug, Clone)]
pub enum Dimension {
    Const(usize),
    Param(String),
}

impl Index<usize> for TensorDims {
    type Output = Dimension;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl TensorDims {
    pub fn new(v: Vec<Dimension>) -> Self {
        Self(v)
    }

    pub fn to_resolved(&self) -> Option<ResolvedTensorDims> {
        let mut resolved = Vec::new();
        for dim in &self.0 {
            match dim {
                Dimension::Const(x) => resolved.push(*x),
                Dimension::Param(_) => return None,
            }
        }
        Some(ResolvedTensorDims::new(resolved))
    }
}
