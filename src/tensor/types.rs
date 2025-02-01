use crate::onnx::operator::{Slice, TensorIndex};
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::dimensions::UnresolvedTensorDims;

use itertools::izip;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum DataType {
    SInt(SIntType),
    UInt(UIntType),
    Float(FloatType),
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum SIntType {
    I32,
    I64,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum UIntType {
    U64,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum FloatType {
    F32,
    F64,
}

impl Into<DataType> for SIntType {
    fn into(self) -> DataType {
        DataType::SInt(self)
    }
}

impl Into<DataType> for UIntType {
    fn into(self) -> DataType {
        DataType::UInt(self)
    }
}

impl Into<DataType> for FloatType {
    fn into(self) -> DataType {
        DataType::Float(self)
    }
}

#[derive(Debug, Clone)]
pub enum TypeError {
    InvalidShape(usize, ResolvedTensorDims),
    BroadcastError(ResolvedTensorDims, ResolvedTensorDims),
    ReshapeError(ResolvedTensorDims, ResolvedTensorDims),
    InferError(String),
    InconsistentInput,
    ElementTypeError,
    UnresolvedInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorType {
    Unresolved(UnresolvedTensorType),
    Resolved(ResolvedTensorType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedTensorType {
    pub elem_type: DataType,
    pub dims: Option<UnresolvedTensorDims>,
}

impl TensorType {
    pub fn normalize(&mut self) {
        if let TensorType::Unresolved(ref ty) = self {
            if let Some(resolved) = ty.dims.as_ref().and_then(|x| x.to_resolved()) {
                *self = TensorType::Resolved(ResolvedTensorType::new(ty.elem_type, resolved));
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTensorType {
    pub elem_type: DataType,
    pub dims: ResolvedTensorDims,
    stride: ResolvedTensorDims,
}

impl ResolvedTensorType {
    pub fn new(elem_type: DataType, dims: ResolvedTensorDims) -> Self {
        let stride = calc_stride(&dims);
        Self {
            elem_type,
            dims,
            stride,
        }
    }

    pub fn stride(&self, i: usize) -> usize {
        self.stride[i]
    }

    pub fn broadcast(&self, target: &ResolvedTensorDims) -> Self {
        let elem_type = self.elem_type;
        let dims = target.clone();
        let mut stride = Vec::with_capacity(target.ndim());
        for _ in 0..(target.ndim() - self.dims.ndim()) {
            stride.push(0);
        }
        stride.extend(self.stride.iter().copied());
        let stride = ResolvedTensorDims::new(stride);
        Self {
            elem_type,
            dims,
            stride,
        }
    }

    // TODO: Check if "broadcastable"
    pub fn is_broadcast_required(&self, target: &ResolvedTensorDims) -> bool {
        self.dims != *target
    }

    pub fn transpose(&self, perms: &[usize]) -> Self {
        let elem_type = self.elem_type;
        let dims = self.dims.transpose(perms);
        let stride = self.stride.transpose(perms);
        Self {
            elem_type,
            dims,
            stride,
        }
    }

    pub fn contiguous(&self) -> Self {
        Self::new(self.elem_type, self.dims.clone())
    }

    pub fn drop_head(&mut self) {
        self.dims = self.dims[1..].to_vec().into();
        self.stride = self.stride[1..].to_vec().into();
    }

    // pub fn reshape(&self, dims: ResolvedTensorDims) -> Self {
    //     assert!(dims.size() == self.dims.size());
    //     Self::new(self.elem_type, dims)
    // }

    pub fn pushed(&self, dim: usize) -> Self {
        let mut dims = self.dims.clone();
        dims.push(dim);
        Self::new(self.elem_type, dims)
    }

    pub fn try_reshape(&self, target: &ResolvedTensorDims) -> Option<Self> {
        if self.dims.size() != target.size() {
            return None;
        }

        let mut new_strides = vec![];
        let mut target_dims = target
            .iter()
            .copied()
            .filter(|x| *x != 1)
            .collect::<Vec<_>>();
        let (mut orig_dims, mut orig_strides): (Vec<_>, Vec<_>) =
            izip!(self.dims.iter(), self.stride.iter())
                .filter(|(x, _)| **x != 1)
                .unzip();

        while !target_dims.is_empty() {
            let mut cur = orig_dims.pop().unwrap();
            let mut target = target_dims.pop().unwrap();
            let stride = orig_strides.pop().unwrap();
            let mut last_dim = cur;
            let mut last_stride = stride;
            new_strides.push(stride);

            loop {
                if cur % target == 0 {
                    break;
                }

                if target < cur {
                    target *= target_dims.pop().unwrap();
                    new_strides.push(stride);
                } else if last_stride * last_dim == *orig_strides.last().unwrap() {
                    last_dim = orig_dims.pop().unwrap();
                    last_stride = orig_strides.pop().unwrap();
                    cur *= last_dim;
                } else {
                    return None;
                }
            }

            let div = cur / target;
            if div != 1 {
                orig_dims.push(div);
                orig_strides.push(target * stride);
            }
        }

        assert!(orig_dims.is_empty());

        new_strides.reverse();
        let new_strides = {
            let mut i = 0;
            let mut buf = Vec::with_capacity(target.ndim());
            for dim in target.iter() {
                if *dim == 1 {
                    buf.push(0);
                } else {
                    buf.push(new_strides[i]);
                    i += 1;
                }
            }
            assert_eq!(i, new_strides.len());
            buf
        };
        Some(Self {
            elem_type: self.elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(new_strides),
        })
    }

    pub fn slice_in_place(&mut self, rank: usize, start: isize, end: isize) -> usize {
        let dim = self.dims[rank];
        let start = TensorIndex::new(start).index(dim);
        let end = TensorIndex::new(end).index(dim);
        assert!(start < dim && dim <= end);
        self.dims[rank] = end - start;
        self.stride[rank] * start
    }

    pub fn slices(&self, slices: &[Slice]) -> Self {
        let mut res = self.clone();
        res.dims = res.dims.slices(slices);
        res
    }

    pub fn is_scalar(&self) -> bool {
        self.dims.is_scalar()
    }
}

impl TensorType {
    pub fn as_resolved(&self) -> Option<&ResolvedTensorType> {
        match self {
            TensorType::Resolved(ty) => Some(ty),
            TensorType::Unresolved(_) => None,
        }
    }
}

impl From<ResolvedTensorType> for TensorType {
    fn from(ty: ResolvedTensorType) -> Self {
        TensorType::Resolved(ty)
    }
}

fn calc_stride(dims: &ResolvedTensorDims) -> ResolvedTensorDims {
    let mut acc = 1;
    let mut stride = vec![0; dims.ndim()];
    for i in (0..dims.ndim()).rev() {
        stride[i] = if dims[i] == 1 { 0 } else { acc };
        acc *= dims[i];
    }
    ResolvedTensorDims::new(stride)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_reshapable0() {
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(vec![2, 3, 4]));
        let target = ResolvedTensorDims::new(vec![2, 3, 1, 2, 1, 2]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![12, 4, 0, 2, 0, 1]),
        };
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable1() {
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(vec![5, 6, 7]))
            .transpose(&[2, 1, 0]);
        let target = ResolvedTensorDims::new(vec![7, 2, 3, 5]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![1, 21, 7, 42]),
        };
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable2() {
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(vec![7, 5, 3]));
        let target = ResolvedTensorDims::new(vec![21, 5]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![5, 1]),
        };
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable3() {
        // 6, 4, 3, 5, 2
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(vec![4, 6, 3, 5, 2]))
            .transpose(&[1, 0, 2, 3, 4]);
        let target = ResolvedTensorDims::new(vec![6, 2, 2, 30]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![30, 360, 180, 1]),
        };
        assert_eq!(orig.dims, ResolvedTensorDims::new(vec![6, 4, 3, 5, 2]));
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable4() {
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(vec![6, 4, 3, 5, 2]));
        let target = ResolvedTensorDims::new(vec![3, 4, 2, 3, 5, 2]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![240, 60, 30, 10, 2, 1]),
        };
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable5() {
        // 6, 5, 3, 4
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(vec![3, 4, 5, 6]))
            .transpose(&[3, 2, 0, 1]);
        let target = ResolvedTensorDims::new(vec![6, 5, 3, 4]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![1, 6, 120, 30]),
        };
        assert_eq!(orig.dims, ResolvedTensorDims::new(vec![6, 5, 3, 4]));
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable6() {
        // 7, 5, 6, 3, 4
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(vec![3, 4, 5, 6, 7]))
            .transpose(&[4, 2, 3, 0, 1]);
        let target = ResolvedTensorDims::new(vec![7, 10, 3, 6, 2]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![1, 21, 7, 420, 210]),
        };
        assert_eq!(orig.dims, ResolvedTensorDims::new(vec![7, 5, 6, 3, 4]));
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable7() {
        // 7, 5, 6, 3, 4
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(vec![3, 4, 5, 6, 7]))
            .transpose(&[4, 2, 3, 0, 1]);
        let target = ResolvedTensorDims::new(vec![7, 30, 2, 3, 2]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![1, 7, 1260, 420, 210]),
        };
        assert_eq!(orig.dims, ResolvedTensorDims::new(vec![7, 5, 6, 3, 4]));
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_non_reshapable0() {
        // [7, 5, 3]
        let orig = ResolvedTensorType::new(
            FloatType::F32.into(),
            ResolvedTensorDims::new(vec![3, 5, 7]),
        )
        .transpose(&[2, 1, 0]);
        let target = ResolvedTensorDims::new(vec![21, 5]);
        assert_eq!(orig.dims.size(), target.size());
        assert_eq!(orig.try_reshape(&target), None);
    }

    #[test]
    fn test_non_reshapable1() {
        // 6, 4, 3, 5, 2
        let orig = ResolvedTensorType::new(
            FloatType::F32.into(),
            ResolvedTensorDims::new(vec![4, 6, 3, 5, 2]),
        )
        .transpose(&[1, 0, 2, 3, 4]);
        let target = ResolvedTensorDims::new(vec![2, 360]);
        assert_eq!(orig.dims.size(), target.size());
        assert_eq!(orig.try_reshape(&target), None);
    }

    #[test]
    fn test_non_reshapable2() {
        // 6, 4, 3, 5, 2
        let orig = ResolvedTensorType::new(
            FloatType::F32.into(),
            ResolvedTensorDims::new(vec![4, 6, 3, 5, 2]),
        )
        .transpose(&[1, 0, 2, 3, 4]);
        let target = ResolvedTensorDims::new(vec![3, 4, 2, 3, 5, 2]);
        assert_eq!(orig.dims.size(), target.size());
        assert_eq!(orig.try_reshape(&target), None);
    }
}
