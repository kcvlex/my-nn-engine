use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::*;

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
