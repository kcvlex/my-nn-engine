use std::iter::FromIterator;
use std::ops::Index;
use std::ops::IndexMut;

use crate::onnx::operator::Slice;
use crate::tensor::types;
use crate::tensor::types::TypeError;
use crate::tensor::Tensor;
use crate::tensor::TensorData;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTensorDims(Vec<usize>);

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

fn broadcast_shape_body(
    lhs: &ResolvedTensorDims,
    rhs: &ResolvedTensorDims,
) -> Result<ResolvedTensorDims, TypeError> {
    let mut res = lhs.clone();
    let rem = lhs.ndim() - rhs.ndim();
    for (v, r) in res.iter_mut().skip(rem).zip(rhs.iter()) {
        if *v == *r {
            continue;
        }
        if *v == 1 {
            *v = *r;
        } else if *r != 1 {
            return Err(TypeError::BroadcastError(lhs.clone(), rhs.clone()));
        }
    }
    Ok(res)
}

// https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
pub fn broadcast_shape(
    lhs: &ResolvedTensorDims,
    rhs: &ResolvedTensorDims,
) -> Result<ResolvedTensorDims, TypeError> {
    if lhs.ndim() < rhs.ndim() {
        broadcast_shape_body(rhs, lhs)
    } else {
        broadcast_shape_body(lhs, rhs)
    }
}

impl ResolvedTensorDims {
    pub fn new(v: Vec<usize>) -> Self {
        if v.iter().sum::<usize>() == 0 {
            Self(vec![])
        } else {
            Self(v)
        }
    }

    pub fn new_direct(v: Vec<usize>) -> Self {
        Self(v)
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn size(&self) -> usize {
        if self.0.is_empty() {
            0
        } else {
            self.0.iter().product()
        }
    }

    pub fn prefix(&self, len: usize) -> Self {
        Self::new(self.0[..len].to_vec())
    }

    pub fn suffix(&self, len: usize) -> Self {
        Self::new(self.0[self.0.len() - len..].to_vec())
    }

    pub fn push(&mut self, v: usize) {
        self.0.push(v);
    }

    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, usize> {
        self.0.iter_mut()
    }

    pub fn last(&self) -> Option<&usize> {
        self.0.last()
    }

    pub fn transpose(&self, perms: &[usize]) -> Self {
        let mut res = self.clone();
        for i in 0..perms.len() {
            res.0[i] = self.0[perms[i]];
        }
        res
    }

    pub fn inner(&self) -> &Vec<usize> {
        &self.0
    }

    pub fn slice_in_place(&mut self, slice: &Slice) {
        let Slice {
            start,
            end,
            axis,
            step,
        } = slice;
        if *step != 1 {
            unimplemented!();
        }
        self[*axis] = (end - start) as usize;
    }

    pub fn slices(&self, slices: &[Slice]) -> Self {
        let mut res = self.clone();
        for slice in slices {
            res.slice_in_place(slice);
        }
        res
    }

    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    pub fn to_tensor(&self) -> Tensor {
        let data = TensorData::SInt(
            types::SIntType::I64,
            self.0.iter().map(|x| *x as i64).collect(),
        );
        let dims = ResolvedTensorDims::new(vec![self.ndim()]);
        Tensor::new(dims, data).unwrap()
    }
}

impl<Idx> Index<Idx> for ResolvedTensorDims
where
    Idx: std::slice::SliceIndex<[usize]>,
{
    type Output = Idx::Output;
    fn index(&self, index: Idx) -> &Self::Output {
        &self.0[index]
    }
}

impl<Idx> IndexMut<Idx> for ResolvedTensorDims
where
    Idx: std::slice::SliceIndex<[usize]>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl From<Vec<usize>> for ResolvedTensorDims {
    fn from(v: Vec<usize>) -> Self {
        Self(v)
    }
}

impl FromIterator<usize> for ResolvedTensorDims {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}
