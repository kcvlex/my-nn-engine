use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::*;
use itertools::zip_eq;

// TODO: Complex
#[derive(Debug, Clone)]
pub enum TensorData {
    SInt(SIntType, Vec<i64>),
    UInt(UIntType, Vec<u64>),
    Float(FloatType, Vec<f64>),
}

impl Eq for TensorData {}

impl PartialEq for TensorData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TensorData::SInt(a0, a1), TensorData::SInt(b0, b1)) => a0 == b0 && a1 == b1,
            (TensorData::UInt(a0, a1), TensorData::UInt(b0, b1)) => a0 == b0 && a1 == b1,
            (TensorData::Float(a0, a1), TensorData::Float(b0, b1)) => a0 == b0 && a1 == b1,
            _ => false,
        }
    }
}

impl TensorData {
    pub fn eq_with_epsillong(&self, other: &Self, epsilon: f64) -> bool {
        match (self, other) {
            (TensorData::SInt(a0, a1), TensorData::SInt(b0, b1)) => a0 == b0 && a1 == b1,
            (TensorData::UInt(a0, a1), TensorData::UInt(b0, b1)) => a0 == b0 && a1 == b1,
            (TensorData::Float(a0, a1), TensorData::Float(b0, b1)) => {
                a0 == b0 &&
                    zip_eq(a1.iter(), b1.iter()).all(|(x, y)| ((x - y).abs() as f64) < epsilon)
            }
            _ => false,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            TensorData::SInt(_, v) => v.len(),
            TensorData::UInt(_, v) => v.len(),
            TensorData::Float(_, v) => v.len(),
        }
    }

    pub fn elem_type(&self) -> DataType {
        match self {
            TensorData::SInt(t, _) => DataType::SInt(*t),
            TensorData::UInt(t, _) => DataType::UInt(*t),
            TensorData::Float(t, _) => DataType::Float(*t),
        }
    }

    pub fn raw_vec(&self) -> Vec<u8> {
        macro_rules! convert {
            ($v: expr) => {{
                $v.iter().map(|x| x.to_le_bytes()).flatten().collect()
            }};
        }

        match self {
            TensorData::SInt(_, v) => convert!(v),
            TensorData::UInt(_, v) => convert!(v),
            TensorData::Float(_, v) => convert!(v),
        }
    }

    pub fn from_bytes(ty: DataType, raw: &[u8]) -> Self {
        macro_rules! convert {
            ($v: expr, $from: ty, $to: ty) => {{
                raw.chunks_exact(std::mem::size_of::<$from>())
                    .map(|x| {
                        let mut bytes = [0; std::mem::size_of::<$from>()];
                        bytes.copy_from_slice(x);
                        <$from>::from_le_bytes(bytes) as $to
                    })
                    .collect()
            }};
        }
        match ty {
            DataType::SInt(ty @ SIntType::I32) => TensorData::SInt(ty, convert!(raw, i32, i64)),
            DataType::SInt(ty @ SIntType::I64) => TensorData::SInt(ty, convert!(raw, i64, i64)),
            DataType::UInt(ty @ UIntType::U64) => TensorData::UInt(ty, convert!(raw, u64, u64)),
            DataType::Float(ty @ FloatType::F32) => TensorData::Float(ty, convert!(raw, f32, f64)),
            DataType::Float(ty @ FloatType::F64) => TensorData::Float(ty, convert!(raw, f64, f64)),
        }
    }

    pub fn zeros(ty: DataType, dims: &ResolvedTensorDims) -> Self {
        let size = dims.size();
        match ty {
            DataType::SInt(t) => TensorData::SInt(t, vec![0; size]),
            DataType::UInt(t) => TensorData::UInt(t, vec![0; size]),
            DataType::Float(t) => TensorData::Float(t, vec![0.0; size]),
        }
    }
}

impl From<Vec<i32>> for TensorData {
    fn from(v: Vec<i32>) -> Self {
        TensorData::SInt(SIntType::I32, v.into_iter().map(|x| x as i64).collect())
    }
}

impl From<Vec<i64>> for TensorData {
    fn from(v: Vec<i64>) -> Self {
        TensorData::SInt(SIntType::I64, v)
    }
}

impl From<Vec<u64>> for TensorData {
    fn from(v: Vec<u64>) -> Self {
        TensorData::UInt(UIntType::U64, v)
    }
}

impl From<Vec<f32>> for TensorData {
    fn from(v: Vec<f32>) -> Self {
        TensorData::Float(FloatType::F32, v.into_iter().map(|x| x as f64).collect())
    }
}

impl From<Vec<f64>> for TensorData {
    fn from(v: Vec<f64>) -> Self {
        TensorData::Float(FloatType::F64, v)
    }
}
