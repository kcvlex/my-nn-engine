pub mod data;
pub mod dimensions;
mod ops;
pub mod types;

use crate::onnx::operator::TensorIndex;
use data::TensorData;
use dimensions::ResolvedTensorDims;
use ops::*;
use types::{DataType, ResolvedTensorType, TensorType, TypeError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub data: TensorData,
    pub ty: ResolvedTensorType,
}

impl Tensor {
    pub fn eq_with_epsilon(&self, other: &Self, epsilon: f64) -> bool {
        self.ty == other.ty && self.data.eq_with_epsillong(&other.data, epsilon)
    }

    pub fn to_indices(&self) -> Option<Vec<TensorIndex>> {
        macro_rules! to_indices {
            ($data: expr, $ty: ty) => {{
                $data
                    .iter()
                    .map(|x| TensorIndex::new(*x as isize))
                    .collect()
            }};
        }

        if self.ty.dims.ndim() != 1 {
            return None;
        }

        match &self.data {
            TensorData::I64(ref v) => Some(to_indices!(v, i64)),
            TensorData::U64(ref v) => Some(to_indices!(v, u64)),
            _ => None,
        }
    }
}

impl TensorData {
    pub fn into_1d_tensor(self) -> Tensor {
        let dims = ResolvedTensorDims::new(vec![self.size()]);
        Tensor::new(dims, self).unwrap()
    }
}

macro_rules! into_raw_tensor {
    ($tensor: expr, $ty: ty) => {{
        let data: Result<&[$ty], _> = $tensor.data.try_as_slice();
        data.map(|data| RawTensor {
            data,
            dims: &$tensor.ty.dims[..],
        })
    }};

    ($data: expr, $dims: expr) => {{
        RawTensor {
            data: &$data[..],
            dims: &$dims[..],
        }
    }};
}

pub(crate) use into_raw_tensor;

pub trait TrySlice<T> {
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
    };
}

define_try_into_raw!(f32, F32);
define_try_into_raw!(f64, F64);
define_try_into_raw!(i64, I64);
define_try_into_raw!(u64, U64);

macro_rules! apply_ndarray_ops {
    ($self: expr, $func: expr, $($args: expr),*) => {{
        match &$self.data {
            TensorData::I64(v) => $func(into_raw_tensor!(v, $self.ty.dims), $($args,)*).try_into(),
            TensorData::U64(v) => $func(into_raw_tensor!(v, $self.ty.dims), $($args,)*).try_into(),
            TensorData::F32(v) => $func(into_raw_tensor!(v, $self.ty.dims), $($args,)*).try_into(),
            TensorData::F64(v) => $func(into_raw_tensor!(v, $self.ty.dims), $($args,)*).try_into(),
        }
    }};
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
        apply_ndarray_ops!(self, ndarray_transpose, perms).unwrap()
    }

    pub fn slices(&self, starts: &[isize], ends: &[isize]) -> Self {
        apply_ndarray_ops!(self, ndarray_slices, starts, ends).unwrap()
    }

    pub fn concat(tensors: &[&Self], axis: usize) -> Result<Self, TypeError> {
        if tensors.is_empty() {
            panic!();
        }

        macro_rules! collect_slices {
            ($ty: ty) => {{
                let data = tensors
                    .iter()
                    .map(|x| into_raw_tensor!(x, $ty))
                    .collect::<Result<Vec<_>, _>>()?;
                ndarray_concat(&data, axis).try_into()
            }};
        }

        match tensors[0].data {
            TensorData::I64(_) => collect_slices!(i64),
            TensorData::U64(_) => collect_slices!(u64),
            TensorData::F32(_) => collect_slices!(f32),
            TensorData::F64(_) => collect_slices!(f64),
        }
    }

    pub fn gather(&self, indices: &Self, axis: usize) -> Self {
        match &indices.data {
            TensorData::I64(_) => {
                let indices = into_raw_tensor!(indices, i64).unwrap();
                apply_ndarray_ops!(self, ndarray_gather, axis, indices).unwrap()
            }
            TensorData::U64(_) => {
                let indices = into_raw_tensor!(indices, u64).unwrap();
                apply_ndarray_ops!(self, ndarray_gather, axis, indices).unwrap()
            }
            _ => panic!(),
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

#[cfg(test)]
mod test {
    use super::*;

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
        let expected = orig
            .slice(ndarray::s![1..2, 2..4, 0..5])
            .into_dyn()
            .to_owned();
        tensor_assert_eq!(s, expected);
    }

    #[test]
    fn test_slice1() {
        let (t, orig) = make_range_tensor!(i64, 3, 4, 5);
        let s = t.slices(&[-2, 0, 1], &[3, 4, -1]);
        let expected = orig
            .slice(ndarray::s![-2..3, 0..4, 1..-1])
            .into_dyn()
            .to_owned();
        tensor_assert_eq!(s, expected);
    }
}
