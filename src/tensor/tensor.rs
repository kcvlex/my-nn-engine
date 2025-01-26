use crate::onnx::operator::TensorIndex;
use crate::tensor::dimensions::UnresolvedTensorDims;
use crate::tensor::resolved_dimensions::ResolvedTensorDims;

use ndarray::ArrayView;
use itertools::izip;

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

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum DataType {
    I64,
    U64,
    F32,
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub data: TensorData,
    pub ty: ResolvedTensorType,
}

impl Tensor {
    pub fn eq_with_epsilon(&self, other: &Self, epsilon: f64) -> bool {
        self.ty == other.ty && self.data.eq_with_epsillong(&other.data, epsilon)
    }
}

// TODO: Complex
#[derive(Debug, Clone)]
pub enum TensorData {
    I64(Vec<i64>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Eq for TensorData {}

impl PartialEq for TensorData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TensorData::I64(a), TensorData::I64(b)) => a == b,
            (TensorData::U64(a), TensorData::U64(b)) => a == b,
            (TensorData::F32(a), TensorData::F32(b)) => a == b,
            (TensorData::F64(a), TensorData::F64(b)) => a == b,
            _ => false,
        }
    }
}

impl TensorData {
    pub fn eq_with_epsillong(&self, other: &Self, epsilon: f64) -> bool {
        macro_rules! eq {
            ($a: expr, $b: expr) => {
                $a.iter()
                    .zip($b.iter())
                    .all(|(x, y)| ((x - y).abs() as f64) < epsilon)
            };
        }
        match (self, other) {
            (TensorData::I64(a), TensorData::I64(b)) => a == b,
            (TensorData::U64(a), TensorData::U64(b)) => a == b,
            (TensorData::F32(a), TensorData::F32(b)) => eq!(a, b),
            (TensorData::F64(a), TensorData::F64(b)) => eq!(a, b),
            _ => false,
        }
    }
}

impl TensorData {
    pub fn into_1d_tensor(self) -> Tensor {
        let dims = ResolvedTensorDims::new(vec![self.size()]);
        Tensor::new(dims, self).unwrap()
    }

    pub fn size(&self) -> usize {
        match self {
            TensorData::I64(v) => v.len(),
            TensorData::U64(v) => v.len(),
            TensorData::F32(v) => v.len(),
            TensorData::F64(v) => v.len(),
        }
    }

    pub fn elem_type(&self) -> DataType {
        match self {
            TensorData::I64(_) => DataType::I64,
            TensorData::U64(_) => DataType::U64,
            TensorData::F32(_) => DataType::F32,
            TensorData::F64(_) => DataType::F64,
        }
    }

    pub fn raw_vec(&self) -> Vec<u8> {
        macro_rules! convert {
            ($v: expr) => {{
                $v.iter().map(|x| x.to_le_bytes()).flatten().collect()
            }};
        }

        match self {
            TensorData::I64(v) => convert!(v),
            TensorData::U64(v) => convert!(v),
            TensorData::F32(v) => convert!(v),
            TensorData::F64(v) => convert!(v),
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        match self {
            TensorData::I64(v) => v.as_ptr() as *const u8,
            TensorData::U64(v) => v.as_ptr() as *const u8,
            TensorData::F32(v) => v.as_ptr() as *const u8,
            TensorData::F64(v) => v.as_ptr() as *const u8,
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            TensorData::I64(v) => v.as_mut_ptr() as *mut u8,
            TensorData::U64(v) => v.as_mut_ptr() as *mut u8,
            TensorData::F32(v) => v.as_mut_ptr() as *mut u8,
            TensorData::F64(v) => v.as_mut_ptr() as *mut u8,
        }
    }
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

    pub fn slice(&self, rank: usize, start: isize, end: isize) -> (Self, usize) {
        let mut res = self.clone();
        let offset = res.slice_in_place(rank, start, end);
        (res, offset)
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

fn ndarray_transpose<T: Clone>(data: &[T], dims: &[usize], perms: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
    ArrayView::from_shape(dims, data)
        .unwrap()
        .permuted_axes(perms)
        .into_dyn()
        .to_owned()
}

fn ndarray_slices<T: Clone>(data: &[T], dims: &[usize], starts: &[isize], ends: &[isize]) -> ndarray::Array<T, ndarray::IxDyn> {
    ArrayView::from_shape(dims, data)
        .unwrap()
        .slice_each_axis(|desc| ndarray::Slice {
            start: starts[desc.axis.index()],
            end: Some(ends[desc.axis.index()]),
            step: 1,
        })
    .into_dyn()
        .to_owned()
}

fn ndarray_concat<T: Clone>(data: &[(&[T], &[usize])], axis: usize) -> ndarray::Array<T, ndarray::IxDyn> {
    let arrays = data.iter().map(|(data, dims)| ArrayView::from_shape(*dims, data).unwrap()).collect::<Vec<_>>();
    ndarray::concatenate(ndarray::Axis(axis), &arrays[..])
        .unwrap()
        .to_owned()
}

macro_rules! apply_ndarray_ops {
    ($data: expr, $func: expr, $($args: expr),*) => {{
        match $data {
            TensorData::I64(v) => $func(&v[..], $($args,)*).try_into(),
            TensorData::U64(v) => $func(&v[..], $($args,)*).try_into(),
            TensorData::F32(v) => $func(&v[..], $($args,)*).try_into(),
            TensorData::F64(v) => $func(&v[..], $($args,)*).try_into(),
        }
    }}
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

    pub fn zeros(ty: DataType, dims: ResolvedTensorDims) -> Self {
        let data = TensorData::zeros(ty, &dims);
        Self::new(dims, data).unwrap()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.data.raw_vec()
    }

    pub fn from_bytes(ty: ResolvedTensorType, raw: &[u8]) -> Result<Self, TypeError> {
        let data = TensorData::from_bytes(ty.elem_type, raw);
        Self::new(ty.dims, data)
    }

    pub fn transpose(&self, perms: &[usize]) -> Self {
        apply_ndarray_ops!(&self.data, ndarray_transpose, &self.ty.dims[..], perms).unwrap()
    }

    pub fn slices(&self, starts: &[isize], ends: &[isize]) -> Self {
        apply_ndarray_ops!(&self.data, ndarray_slices, &self.ty.dims[..], starts, ends).unwrap()
    }

    pub fn concat(tensors: &[&Self], axis: usize) -> Result<Self, TypeError> {
        if tensors.is_empty() {
            panic!();
        }

        macro_rules! collect_slices {
            ($ty: ty) => {{
        let data = tensors.iter().map(|x| {
            let data: Result<&[$ty], _> = x.data.try_as_slice();
            let dims = x.ty.dims.as_slice();
            data.map(|x| (x, dims))
        }).collect::<Result<Vec<_>, _>>()?;
        ndarray_concat(&data, axis).try_into()
            }}
        }

        match tensors[0].data {
            TensorData::I64(_) => collect_slices!(i64),
            TensorData::U64(_) => collect_slices!(u64),
            TensorData::F32(_) => collect_slices!(f32),
            TensorData::F64(_) => collect_slices!(f64),
        }
    }
}

macro_rules! define_try_from {
    ($ty: ty, $data: ident) => {
        impl TryFrom<ndarray::Array<$ty, ndarray::IxDyn>> for Tensor {
            type Error = TypeError;
            fn try_from(array: ndarray::Array<$ty, ndarray::IxDyn>) -> Result<Self, Self::Error> {
                let dim = ResolvedTensorDims::new(array.shape().to_vec());
                let data = TensorData::$data(array.flatten().to_vec());
                Self::new(dim, data)
            }
        }
    };
}

define_try_from!(f32, F32);
define_try_from!(f64, F64);
define_try_from!(i64, I64);
define_try_from!(u64, U64);

trait TrySlice<T> {
    fn try_as_slice(&self) -> Result<&[T], TypeError>;
}

macro_rules! define_try_into_raw {
    ($ty: ty, $data: ident) => {
        impl TrySlice<$ty> for TensorData {
            fn try_as_slice(&self) -> Result<&[$ty], TypeError> {
                match self {
                    TensorData::$data(v) => Ok(&v[..]),
                    _ => Err(TypeError::ElementTypeError),
                }
            }
        }
    }
}

define_try_into_raw!(f32, F32);
define_try_into_raw!(f64, F64);
define_try_into_raw!(i64, I64);
define_try_into_raw!(u64, U64);

impl TensorData {
    pub fn from_bytes(ty: DataType, raw: &[u8]) -> Self {
        macro_rules! convert {
            ($v: expr, $ty: ty) => {{
                raw.chunks_exact(std::mem::size_of::<$ty>())
                    .map(|x| {
                        let mut bytes = [0; std::mem::size_of::<$ty>()];
                        bytes.copy_from_slice(x);
                        <$ty>::from_le_bytes(bytes)
                    })
                    .collect()
            }};
        }
        match ty {
            DataType::I64 => TensorData::I64(convert!(raw, i64)),
            DataType::U64 => TensorData::U64(convert!(raw, u64)),
            DataType::F32 => TensorData::F32(convert!(raw, f32)),
            DataType::F64 => TensorData::F64(convert!(raw, f64)),
        }
    }

    pub fn zeros(ty: DataType, dims: &ResolvedTensorDims) -> Self {
        let size = dims.size();
        match ty {
            DataType::I64 => TensorData::I64(vec![0; size]),
            DataType::U64 => TensorData::U64(vec![0; size]),
            DataType::F32 => TensorData::F32(vec![0.0; size]),
            DataType::F64 => TensorData::F64(vec![0.0; size]),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_reshapable0() {
        let orig = ResolvedTensorType::new(DataType::F32, ResolvedTensorDims::new(vec![2, 3, 4]));
        let target = ResolvedTensorDims::new(vec![2, 3, 1, 2, 1, 2]);
        let expected = ResolvedTensorType {
            elem_type: DataType::F32,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![12, 4, 0, 2, 0, 1]),
        };
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable1() {
        let orig = ResolvedTensorType::new(DataType::F32, ResolvedTensorDims::new(vec![5, 6, 7]))
            .transpose(&[2, 1, 0]);
        let target = ResolvedTensorDims::new(vec![7, 2, 3, 5]);
        let expected = ResolvedTensorType {
            elem_type: DataType::F32,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![1, 21, 7, 42]),
        };
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable2() {
        let orig = ResolvedTensorType::new(DataType::F32, ResolvedTensorDims::new(vec![7, 5, 3]));
        let target = ResolvedTensorDims::new(vec![21, 5]);
        let expected = ResolvedTensorType {
            elem_type: DataType::F32,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![5, 1]),
        };
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable3() {
        // 6, 4, 3, 5, 2
        let orig =
            ResolvedTensorType::new(DataType::F32, ResolvedTensorDims::new(vec![4, 6, 3, 5, 2]))
                .transpose(&[1, 0, 2, 3, 4]);
        let target = ResolvedTensorDims::new(vec![6, 2, 2, 30]);
        let expected = ResolvedTensorType {
            elem_type: DataType::F32,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![30, 360, 180, 1]),
        };
        assert_eq!(orig.dims, ResolvedTensorDims::new(vec![6, 4, 3, 5, 2]));
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable4() {
        let orig =
            ResolvedTensorType::new(DataType::F32, ResolvedTensorDims::new(vec![6, 4, 3, 5, 2]));
        let target = ResolvedTensorDims::new(vec![3, 4, 2, 3, 5, 2]);
        let expected = ResolvedTensorType {
            elem_type: DataType::F32,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![240, 60, 30, 10, 2, 1]),
        };
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable5() {
        // 6, 5, 3, 4
        let orig =
            ResolvedTensorType::new(DataType::F32, ResolvedTensorDims::new(vec![3, 4, 5, 6]))
                .transpose(&[3, 2, 0, 1]);
        let target = ResolvedTensorDims::new(vec![6, 5, 3, 4]);
        let expected = ResolvedTensorType {
            elem_type: DataType::F32,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![1, 6, 120, 30]),
        };
        assert_eq!(orig.dims, ResolvedTensorDims::new(vec![6, 5, 3, 4]));
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable6() {
        // 7, 5, 6, 3, 4
        let orig =
            ResolvedTensorType::new(DataType::F32, ResolvedTensorDims::new(vec![3, 4, 5, 6, 7]))
                .transpose(&[4, 2, 3, 0, 1]);
        let target = ResolvedTensorDims::new(vec![7, 10, 3, 6, 2]);
        let expected = ResolvedTensorType {
            elem_type: DataType::F32,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![1, 21, 7, 420, 210]),
        };
        assert_eq!(orig.dims, ResolvedTensorDims::new(vec![7, 5, 6, 3, 4]));
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_reshapable7() {
        // 7, 5, 6, 3, 4
        let orig =
            ResolvedTensorType::new(DataType::F32, ResolvedTensorDims::new(vec![3, 4, 5, 6, 7]))
                .transpose(&[4, 2, 3, 0, 1]);
        let target = ResolvedTensorDims::new(vec![7, 30, 2, 3, 2]);
        let expected = ResolvedTensorType {
            elem_type: DataType::F32,
            dims: target.clone(),
            stride: ResolvedTensorDims::new(vec![1, 7, 1260, 420, 210]),
        };
        assert_eq!(orig.dims, ResolvedTensorDims::new(vec![7, 5, 6, 3, 4]));
        assert_eq!(orig.try_reshape(&target), Some(expected));
    }

    #[test]
    fn test_non_reshapable0() {
        // [7, 5, 3]
        let orig = ResolvedTensorType::new(DataType::F32, ResolvedTensorDims::new(vec![3, 5, 7]))
            .transpose(&[2, 1, 0]);
        let target = ResolvedTensorDims::new(vec![21, 5]);
        assert_eq!(orig.dims.size(), target.size());
        assert_eq!(orig.try_reshape(&target), None);
    }

    #[test]
    fn test_non_reshapable1() {
        // 6, 4, 3, 5, 2
        let orig =
            ResolvedTensorType::new(DataType::F32, ResolvedTensorDims::new(vec![4, 6, 3, 5, 2]))
                .transpose(&[1, 0, 2, 3, 4]);
        let target = ResolvedTensorDims::new(vec![2, 360]);
        assert_eq!(orig.dims.size(), target.size());
        assert_eq!(orig.try_reshape(&target), None);
    }

    #[test]
    fn test_non_reshapable2() {
        // 6, 4, 3, 5, 2
        let orig =
            ResolvedTensorType::new(DataType::F32, ResolvedTensorDims::new(vec![4, 6, 3, 5, 2]))
                .transpose(&[1, 0, 2, 3, 4]);
        let target = ResolvedTensorDims::new(vec![3, 4, 2, 3, 5, 2]);
        assert_eq!(orig.dims.size(), target.size());
        assert_eq!(orig.try_reshape(&target), None);
    }

    macro_rules! make_range_tensor {
        ($ty: ty, $($dim: expr),*) => {{
            let len = [$($dim),*].iter().product();
            let orig = ndarray::Array::from_iter((0..len).map(|x| x as $ty))
                .into_shape_with_order(($($dim),*))
                .unwrap();
            let res: Result<(Tensor, _), _> = orig
                .clone()
                .into_dyn()
                .try_into()
                .map(|t| (t, orig));
            res.unwrap()
        }};
    }
    
    macro_rules! tensor_assert_eq {
        ($left: expr, $right: expr) => {{
            let right = Tensor::try_from($right).unwrap();
            assert_eq!($left, right);
        }};
    }

    #[test]
    fn test_slice0() {
        let (t, orig) = make_range_tensor!(i64, 3, 4, 5);
        let s = t.slices(&[1, 2, 0], &[2, 4, 5]);
        let expected = orig.slice(ndarray::s![1..2, 2..4, 0..5]).into_dyn().to_owned();
        tensor_assert_eq!(s, expected);
    }
    
    #[test]
    fn test_slice1() {
        let (t, orig) = make_range_tensor!(i64, 3, 4, 5);
        let s = t.slices(&[-2, 0, 1], &[3, 4, -1]);
        let expected = orig.slice(ndarray::s![-2..3, 0..4, 1..-1]).into_dyn().to_owned();
        println!("{:?}", s);
        tensor_assert_eq!(s, expected);
    }
}
