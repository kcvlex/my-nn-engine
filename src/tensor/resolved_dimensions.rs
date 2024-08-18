use crate::tensor::tensor::TypeError;
use std::ops::Index;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTensorDims(Vec<usize>);

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
        v.into()
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn size(&self) -> usize {
        self.0.iter().product()
    }

    pub fn prefix(&self, len: usize) -> Self {
        Self(self.0[..len].to_vec())
    }

    pub fn suffix(&self, len: usize) -> Self {
        Self(self.0[self.0.len() - len..].to_vec())
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

impl From<Vec<usize>> for ResolvedTensorDims {
    fn from(v: Vec<usize>) -> Self {
        Self(v)
    }
}
