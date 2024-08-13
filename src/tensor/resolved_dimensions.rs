use crate::tensor::tensor::TypeError;
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone)]
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

impl Deref for ResolvedTensorDims {
    type Target = Vec<usize>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ResolvedTensorDims {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl ResolvedTensorDims {
    pub fn new(v: Vec<usize>) -> Self {
        Self(v)
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn size(&self) -> usize {
        self.0.iter().product()
    }
}
