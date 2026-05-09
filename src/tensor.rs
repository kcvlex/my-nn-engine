pub mod data;
mod ops;
pub mod types;

use data::CompPolicy;
use data::TensorData;
use ops::*;
use types::DataType;
use types::FloatType;
use types::ResolvedTensorDims;
use types::ResolvedTensorType;
use types::SIntType;
use types::TypeError;
use types::UIntType;

use crate::graph::operator::TensorIndex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub data: TensorData,
    pub dims: ResolvedTensorDims,
}

impl Tensor {
    pub fn eq_with_epsilon(&self, other: &Self, epsilon: f64, policy: CompPolicy) -> bool {
        self.dims == other.dims && self.data.eq_with_epsilon(&other.data, epsilon, policy)
    }

    pub fn to_indices(&self) -> Option<Vec<TensorIndex>> {
        macro_rules! to_indices {
            ($data: expr) => {{
                $data
                    .iter()
                    .map(|x| TensorIndex::new(*x as isize))
                    .collect()
            }};
        }

        if self.dims.ndim() != 1 {
            return None;
        }

        match &self.data {
            TensorData::SInt(_, ref v) => Some(to_indices!(v)),
            TensorData::UInt(_, ref v) => Some(to_indices!(v)),
            _ => None,
        }
    }

    pub fn to_1d_floats(&self) -> Option<Vec<f64>> {
        if self.dims.ndim() != 1 {
            return None;
        }

        match &self.data {
            TensorData::Float(_, ref v) => Some(v.clone()),
            _ => None,
        }
    }

    pub fn to_1d_sints(&self) -> Option<Vec<i64>> {
        if self.dims.ndim() != 1 {
            return None;
        }

        match &self.data {
            TensorData::SInt(_, ref v) => Some(v.clone()),
            _ => None,
        }
    }

    pub fn tensor_type(&self) -> ResolvedTensorType {
        ResolvedTensorType::new(self.data.elem_type(), self.dims.clone())
    }
}

impl TensorData {
    pub fn into_1d_tensor(self) -> Tensor {
        let dims = ResolvedTensorDims::new(&[self.size()]);
        Tensor::new(dims, self).unwrap()
    }
}

macro_rules! cast_vec {
    ($data: expr, $ty: ty) => {{
        $data.iter().map(|x| *x as $ty).collect::<Vec<_>>()
    }};
}

macro_rules! into_raw_tensor {
    ($tensor: expr, $ty: ty) => {{
        match &$tensor.data {
            TensorData::Bool(v) => RawTensor {
                data: v.iter().map(|&b| b as $ty).collect(),
                dims: &$tensor.dims[..],
            },
            TensorData::SInt(_, v) => RawTensor {
                data: cast_vec!(v, $ty),
                dims: &$tensor.dims[..],
            },
            TensorData::UInt(_, v) => RawTensor {
                data: cast_vec!(v, $ty),
                dims: &$tensor.dims[..],
            },
            TensorData::Float(_, v) => RawTensor {
                data: cast_vec!(v, $ty),
                dims: &$tensor.dims[..],
            },
        }
    }};
}

macro_rules! apply_ndarray_ops {
    ($self: expr, $func: expr, $($args: expr),*) => {{
        match &$self.data {
            TensorData::Bool(_) | TensorData::UInt(UIntType::U8, _) => $func(into_raw_tensor!($self, u8), $($args,)*).try_into(),
            TensorData::SInt(SIntType::I8, _) => $func(into_raw_tensor!($self, i8), $($args,)*).try_into(),
            TensorData::SInt(SIntType::I32, _) => $func(into_raw_tensor!($self, i32), $($args,)*).try_into(),
            TensorData::SInt(SIntType::I64, _) => $func(into_raw_tensor!($self, i64), $($args,)*).try_into(),
            TensorData::UInt(UIntType::U64, _) => $func(into_raw_tensor!($self, u64), $($args,)*).try_into(),
            TensorData::Float(FloatType::F32, _) => $func(into_raw_tensor!($self, f32), $($args,)*).try_into(),
            TensorData::Float(FloatType::F64, _) => $func(into_raw_tensor!($self, f64), $($args,)*).try_into(),
            TensorData::Float(FloatType::BF16, _) => unimplemented!("BF16 inline ndarray ops"),
        }
    }};
}

impl Tensor {
    pub fn new(dims: ResolvedTensorDims, data: TensorData) -> Result<Self, TypeError> {
        if dims.size().max(1) != data.size().max(1) {
            return Err(TypeError::InvalidShape(data.size(), dims));
        }
        Ok(Self { data, dims })
    }

    // pub fn tensor_type(&self) -> TensorType {
    //     self.ty.clone().into()
    // }

    pub fn zeros(ty: DataType, dims: ResolvedTensorDims) -> Self {
        let data = TensorData::zeros(ty, &dims);
        Self::new(dims, data).unwrap()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.data.raw_vec()
    }

    pub fn to_ndarray(&self) -> ndarray::ArrayD<f64> {
        let dims: Vec<usize> = self.dims.iter().copied().collect();
        let v: Vec<f64> = match &self.data {
            TensorData::Bool(v) => v.iter().map(|&b| b as u8 as f64).collect(),
            TensorData::Float(_, v) => v.clone(),
            TensorData::SInt(_, v) => v.iter().map(|&x| x as f64).collect(),
            TensorData::UInt(_, v) => v.iter().map(|&x| x as f64).collect(),
        };
        ndarray::ArrayD::from_shape_vec(dims, v).unwrap()
    }

    // pub fn from_bytes(ty: ResolvedTensorType, raw: &[u8]) -> Result<Self, TypeError> {
    //     let data = TensorData::from_bytes(ty.elem_type, raw);
    //     Self::new(ty.dims, data)
    // }

    pub fn reshape(&self, dims: &ResolvedTensorDims) -> Self {
        assert!(self.dims.size().max(1) == dims.size().max(1));
        Self {
            data: self.data.clone(),
            dims: dims.clone(),
        }
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
                    .map(|x| into_raw_tensor!(*x, $ty))
                    .collect::<Vec<_>>();
                ndarray_concat(&data, axis).try_into()
            }};
        }

        match tensors[0].data {
            TensorData::Bool(_) | TensorData::UInt(UIntType::U8, _) => collect_slices!(u8),
            TensorData::SInt(SIntType::I8, _) => collect_slices!(i8),
            TensorData::SInt(SIntType::I32, _) => collect_slices!(i32),
            TensorData::SInt(SIntType::I64, _) => collect_slices!(i64),
            TensorData::UInt(UIntType::U64, _) => collect_slices!(u64),
            TensorData::Float(FloatType::F32, _) => collect_slices!(f32),
            TensorData::Float(FloatType::F64, _) => collect_slices!(f64),
            TensorData::Float(FloatType::BF16, _) => unimplemented!("BF16 inline concat"),
        }
    }

    pub fn gather(&self, indices: &Self, axis: usize) -> Self {
        match &indices.data {
            TensorData::SInt(_, data) => {
                let indices = RawTensor {
                    data: data.clone(),
                    dims: &indices.dims[..],
                };
                apply_ndarray_ops!(self, ndarray_gather, axis, indices).unwrap()
            }
            TensorData::UInt(_, data) => {
                let indices = RawTensor {
                    data: data.clone(),
                    dims: &indices.dims[..],
                };
                apply_ndarray_ops!(self, ndarray_gather, axis, indices).unwrap()
            }
            _ => panic!(),
        }
    }

    pub fn broadcast(&self, dims: &ResolvedTensorDims) -> Self {
        apply_ndarray_ops!(self, ndarray_broadcast, &dims[..]).unwrap()
    }
}

macro_rules! define_try_from {
    ($ty: ty) => {
        impl TryFrom<ndarray::Array<$ty, ndarray::IxDyn>> for Tensor {
            type Error = TypeError;
            fn try_from(array: ndarray::Array<$ty, ndarray::IxDyn>) -> Result<Self, Self::Error> {
                let dim = ResolvedTensorDims::new(&array.shape()[..]);
                let data: TensorData = array.flatten().to_vec().into();
                Self::new(dim, data)
            }
        }
    };
}

define_try_from!(f32);
define_try_from!(f64);
define_try_from!(i8);
define_try_from!(i32);
define_try_from!(i64);
define_try_from!(u64);
define_try_from!(u8);

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
