use std::iter::FromIterator;
use std::ops::Index;
use std::ops::IndexMut;

use itertools::izip;
use itertools::zip_eq;
use smallvec::smallvec;
use smallvec::SmallVec;

use crate::onnx::operator::Slice;
use crate::onnx::operator::TensorIndex;
use crate::tensor::Tensor;
use crate::tensor::TensorData;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTensorDims(SmallVec<[usize; 5]>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedTensorDims(SmallVec<[Dimension; 5]>);

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
    pub fn new(v: &[Dimension]) -> Self {
        Self(v.into())
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
        Some(ResolvedTensorDims::new(&resolved))
    }

    pub fn inner(&self) -> &[Dimension] {
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
    pub fn new(v: &[usize]) -> Self {
        if v.iter().sum::<usize>() == 0 {
            Self(smallvec![])
        } else {
            Self(v.into())
        }
    }

    pub fn new_direct(v: &[usize]) -> Self {
        Self(v.into())
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
        Self::new(self.0[..len].into())
    }

    pub fn suffix(&self, len: usize) -> Self {
        Self::new(self.0[self.0.len() - len..].into())
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

    pub fn inner(&self) -> &[usize] {
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

    pub fn compatible_with_scalar(&self) -> bool {
        self.is_scalar() || self.size() == 1
    }

    pub fn to_tensor(&self) -> Tensor {
        let data = TensorData::SInt(SIntType::I64, self.0.iter().map(|x| *x as i64).collect());
        let dims = ResolvedTensorDims::new(&[self.ndim()]);
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

impl From<&[usize]> for ResolvedTensorDims {
    fn from(v: &[usize]) -> Self {
        Self(v.into())
    }
}

impl FromIterator<usize> for ResolvedTensorDims {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum DataType {
    SInt(SIntType),
    UInt(UIntType),
    Float(FloatType),
}

impl Default for DataType {
    fn default() -> Self {
        DataType::SInt(SIntType::I32)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum SIntType {
    I32,
    I64,
}

impl SIntType {
    pub fn bit_width(&self) -> usize {
        match self {
            SIntType::I32 => 32,
            SIntType::I64 => 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum UIntType {
    U64,
}

impl UIntType {
    pub fn bit_width(&self) -> usize {
        match self {
            UIntType::U64 => 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum FloatType {
    F32,
    F64,
}

impl FloatType {
    pub fn bit_width(&self) -> usize {
        match self {
            FloatType::F32 => 32,
            FloatType::F64 => 64,
        }
    }
}

impl DataType {
    pub fn float_type(&self) -> Option<FloatType> {
        match self {
            DataType::Float(t) => Some(*t),
            _ => None,
        }
    }

    pub fn is_int(&self) -> bool {
        matches!(self, DataType::SInt(_) | DataType::UInt(_))
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DataType::Float(_))
    }

    pub fn bit_width(&self) -> usize {
        match self {
            DataType::SInt(sty) => sty.bit_width(),
            DataType::UInt(uty) => uty.bit_width(),
            DataType::Float(fty) => fty.bit_width(),
        }
    }
}

impl From<SIntType> for DataType {
    fn from(val: SIntType) -> Self {
        DataType::SInt(val)
    }
}

impl From<UIntType> for DataType {
    fn from(val: UIntType) -> Self {
        DataType::UInt(val)
    }
}

impl From<FloatType> for DataType {
    fn from(val: FloatType) -> Self {
        DataType::Float(val)
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

    pub fn with_stride(
        elem_type: DataType,
        dims: ResolvedTensorDims,
        stride: ResolvedTensorDims,
    ) -> Self {
        if !dims.is_scalar() {
            assert_eq!(dims.ndim(), stride.ndim());
        }
        Self {
            elem_type,
            dims,
            stride,
        }
    }

    pub fn stride(&self, i: usize) -> usize {
        self.stride[i]
    }

    pub fn strides(&self) -> &ResolvedTensorDims {
        &self.stride
    }

    pub fn broadcast(&self, target: &ResolvedTensorDims) -> Self {
        let elem_type = self.elem_type;
        let dims = target.clone();
        let mut stride = vec![0; target.ndim()];
        if !self.stride.is_scalar() {
            for (dst, src) in zip_eq(
                stride.iter_mut().skip(target.ndim() - self.dims.ndim()),
                self.stride.iter(),
            ) {
                *dst = *src;
            }
        };
        assert_eq!(target.ndim(), stride.len());
        let stride = ResolvedTensorDims::new_direct(&stride);
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

    pub fn extend_per_channel_params(&self, target: &ResolvedTensorDims) -> Self {
        assert!(self.dims.inner().len() == 1);
        let mut strides = vec![0; target.ndim()];
        strides[1] = 1;
        Self {
            elem_type: self.elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new_direct(&strides),
        }
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

    pub fn is_contiguous(&self) -> bool {
        let mut acc = 1;
        for (i, stride) in self.stride.iter().enumerate().rev() {
            if *stride != 0 && *stride != acc {
                return false;
            }
            acc *= self.dims[i];
        }
        true
    }

    pub fn drop_head(&mut self) {
        self.dims = self.dims[1..].into();
        self.stride = self.stride[1..].into();
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
        if self.dims.compatible_with_scalar() && target.compatible_with_scalar() {
            return Some(Self::new(self.elem_type, target.clone()));
        }

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
        let stride = ResolvedTensorDims::new_direct(&new_strides);
        assert!(stride.ndim() == target.ndim());
        Some(Self {
            elem_type: self.elem_type,
            dims: target.clone(),
            stride,
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

    pub(crate) fn normalize_strides(&mut self) {
        for (dim, stride) in izip!(self.dims.iter(), self.stride.iter_mut()) {
            if *dim == 1 {
                *stride = 0;
            }
        }
    }

    pub fn slices(&self, slices: &[Slice]) -> Self {
        let mut res = self.clone();
        res.dims = res.dims.slices(slices);
        res.normalize_strides();
        res
    }

    pub fn is_scalar(&self) -> bool {
        self.dims.is_scalar() || self.dims.size() == 1
    }

    pub fn storage_num_elements(&self) -> usize {
        if self.is_scalar() {
            return 1;
        }

        izip!(self.dims.iter(), self.stride.iter())
            .map(|(dim, stride)| dim.max(&1) * stride)
            .max()
            .unwrap()
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
    ResolvedTensorDims::new_direct(&stride)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_reshapable0() {
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(&[2, 3, 4]));
        let target = ResolvedTensorDims::new(&[2, 3, 1, 2, 1, 2]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(&[12, 4, 0, 2, 0, 1]),
        };
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable1() {
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(&[5, 6, 7]))
            .transpose(&[2, 1, 0]);
        let target = ResolvedTensorDims::new(&[7, 2, 3, 5]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(&[1, 21, 7, 42]),
        };
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable2() {
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(&[7, 5, 3]));
        let target = ResolvedTensorDims::new(&[21, 5]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(&[5, 1]),
        };
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable3() {
        // 6, 4, 3, 5, 2
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(&[4, 6, 3, 5, 2]))
            .transpose(&[1, 0, 2, 3, 4]);
        let target = ResolvedTensorDims::new(&[6, 2, 2, 30]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(&[30, 360, 180, 1]),
        };
        assert_eq!(orig.dims, ResolvedTensorDims::new(&[6, 4, 3, 5, 2]));
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable4() {
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(&[6, 4, 3, 5, 2]));
        let target = ResolvedTensorDims::new(&[3, 4, 2, 3, 5, 2]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(&[240, 60, 30, 10, 2, 1]),
        };
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable5() {
        // 6, 5, 3, 4
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(&[3, 4, 5, 6]))
            .transpose(&[3, 2, 0, 1]);
        let target = ResolvedTensorDims::new(&[6, 5, 3, 4]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(&[1, 6, 120, 30]),
        };
        assert_eq!(orig.dims, ResolvedTensorDims::new(&[6, 5, 3, 4]));
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable6() {
        // 7, 5, 6, 3, 4
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(&[3, 4, 5, 6, 7]))
            .transpose(&[4, 2, 3, 0, 1]);
        let target = ResolvedTensorDims::new(&[7, 10, 3, 6, 2]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(&[1, 21, 7, 420, 210]),
        };
        assert_eq!(orig.dims, ResolvedTensorDims::new(&[7, 5, 6, 3, 4]));
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable7() {
        // 7, 5, 6, 3, 4
        let elem_type = FloatType::F32.into();
        let orig = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(&[3, 4, 5, 6, 7]))
            .transpose(&[4, 2, 3, 0, 1]);
        let target = ResolvedTensorDims::new(&[7, 30, 2, 3, 2]);
        let expected = ResolvedTensorType {
            elem_type,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(&[1, 7, 1260, 420, 210]),
        };
        assert_eq!(orig.dims, ResolvedTensorDims::new(&[7, 5, 6, 3, 4]));
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_non_reshapable0() {
        // [7, 5, 3]
        let orig =
            ResolvedTensorType::new(FloatType::F32.into(), ResolvedTensorDims::new(&[3, 5, 7]))
                .transpose(&[2, 1, 0]);
        let target = ResolvedTensorDims::new(&[21, 5]);
        assert_eq!(orig.dims.size(), target.size());
        assert_eq!(orig.try_reshape(&target), None);
    }

    #[test]
    fn test_non_reshapable1() {
        // 6, 4, 3, 5, 2
        let orig = ResolvedTensorType::new(
            FloatType::F32.into(),
            ResolvedTensorDims::new(&[4, 6, 3, 5, 2]),
        )
        .transpose(&[1, 0, 2, 3, 4]);
        let target = ResolvedTensorDims::new(&[2, 360]);
        assert_eq!(orig.dims.size(), target.size());
        assert_eq!(orig.try_reshape(&target), None);
    }

    #[test]
    fn test_non_reshapable2() {
        // 6, 4, 3, 5, 2
        let orig = ResolvedTensorType::new(
            FloatType::F32.into(),
            ResolvedTensorDims::new(&[4, 6, 3, 5, 2]),
        )
        .transpose(&[1, 0, 2, 3, 4]);
        let target = ResolvedTensorDims::new(&[3, 4, 2, 3, 5, 2]);
        assert_eq!(orig.dims.size(), target.size());
        assert_eq!(orig.try_reshape(&target), None);
    }
}
