use crate::tensor::resolved_dimensions::ResolvedTensorDims;
use std::ops::Index;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedTensorDims(Vec<Dimension>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParamKey(String);

impl std::fmt::Display for ParamKey {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dimension {
    Const(usize),
    Param(ParamKey),
}

impl<T: Into<String>> From<T> for ParamKey {
    fn from(s: T) -> Self {
        ParamKey(s.into())
    }
}

impl Index<usize> for UnresolvedTensorDims {
    type Output = Dimension;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl UnresolvedTensorDims {
    pub fn new(v: Vec<Dimension>) -> Self {
        Self(v)
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
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

    pub fn inner(&self) -> &Vec<Dimension> {
        &self.0
    }
}
