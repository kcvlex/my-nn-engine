use itertools::zip_eq;

use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::*;

// TODO: Complex
#[derive(Debug, Clone)]
pub enum TensorData {
    SInt(SIntType, Vec<i64>),
    UInt(UIntType, Vec<u64>),
    Float(FloatType, Vec<f64>),
}

#[derive(Debug, Clone, Copy)]
pub enum ScalarData {
    SInt(SIntType, i64),
    UInt(UIntType, u64),
    Float(FloatType, f64),
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

impl Eq for ScalarData {}

impl PartialEq for ScalarData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ScalarData::SInt(a0, a1), ScalarData::SInt(b0, b1)) => a0 == b0 && a1 == b1,
            (ScalarData::UInt(a0, a1), ScalarData::UInt(b0, b1)) => a0 == b0 && a1 == b1,
            (ScalarData::Float(a0, a1), ScalarData::Float(b0, b1)) => a0 == b0 && a1 == b1,
            _ => false,
        }
    }
}

impl ScalarData {
    pub fn to_tensor_data(&self, num: usize) -> TensorData {
        match self {
            ScalarData::SInt(ty, v) => TensorData::SInt(*ty, vec![*v; num]),
            ScalarData::UInt(ty, v) => TensorData::UInt(*ty, vec![*v; num]),
            ScalarData::Float(ty, v) => TensorData::Float(*ty, vec![*v; num]),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CompPolicy {
    Abs,
    Rel,
    Either,
}

impl TensorData {
    pub fn eq_with_epsillong(&self, other: &Self, epsilon: f64, comp: CompPolicy) -> bool {
        match (self, other) {
            (TensorData::SInt(a0, a1), TensorData::SInt(b0, b1)) => a0 == b0 && a1 == b1,
            (TensorData::UInt(a0, a1), TensorData::UInt(b0, b1)) => a0 == b0 && a1 == b1,
            (TensorData::Float(a0, a1), TensorData::Float(b0, b1)) => {
                a0 == b0 &&
                    zip_eq(a1.iter(), b1.iter())
                        .map(|(x, y)| {
                            let res = if x.is_infinite() && y.is_infinite() {
                                x.is_sign_positive() == y.is_sign_positive()
                            } else {
                                let diff = (x - y).abs();
                                let abs = diff < epsilon;
                                let x = x.abs();
                                let rel = if x == 0.0 {
                                    diff < epsilon
                                } else {
                                    diff / x < epsilon
                                };
                                match comp {
                                    CompPolicy::Abs => abs,
                                    CompPolicy::Rel => rel,
                                    CompPolicy::Either => abs || rel,
                                }
                            };
                            (x, y, res)
                        })
                        .enumerate()
                        .inspect(|(i, (x, y, res))| {
                            if !res {
                                println!("{}: {} != {}", i, x, y);
                            }
                        })
                        .all(|(_, (_, _, res))| res)
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

    pub fn to_scalar_data(&self) -> Option<ScalarData> {
        match self {
            TensorData::SInt(t, v) if v.len() == 1 => Some(ScalarData::SInt(*t, v[0])),
            TensorData::UInt(t, v) if v.len() == 1 => Some(ScalarData::UInt(*t, v[0])),
            TensorData::Float(t, v) if v.len() == 1 => Some(ScalarData::Float(*t, v[0])),
            _ => None,
        }
    }

    pub fn to_scalars(&self) -> Vec<ScalarData> {
        match self {
            TensorData::SInt(t, v) => v.iter().map(|x| ScalarData::SInt(*t, *x)).collect(),
            TensorData::UInt(t, v) => v.iter().map(|x| ScalarData::UInt(*t, *x)).collect(),
            TensorData::Float(t, v) => v.iter().map(|x| ScalarData::Float(*t, *x)).collect(),
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

impl ScalarData {
    pub fn elem_type(&self) -> DataType {
        match self {
            ScalarData::SInt(t, _) => DataType::SInt(*t),
            ScalarData::UInt(t, _) => DataType::UInt(*t),
            ScalarData::Float(t, _) => DataType::Float(*t),
        }
    }
}

impl TryInto<ScalarData> for TensorData {
    type Error = TypeError;

    fn try_into(self) -> Result<ScalarData, Self::Error> {
        match self {
            TensorData::SInt(ty, v) if v.len() == 1 => Ok(ScalarData::SInt(ty, v[0])),
            TensorData::UInt(ty, v) if v.len() == 1 => Ok(ScalarData::UInt(ty, v[0])),
            TensorData::Float(ty, v) if v.len() == 1 => Ok(ScalarData::Float(ty, v[0])),
            _ => Err(TypeError::InvalidShape(
                self.size(),
                ResolvedTensorDims::new(&[1]),
            )),
        }
    }
}

impl std::fmt::Display for ScalarData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarData::SInt(_, v) => write!(f, "{}", v),
            ScalarData::UInt(_, v) => write!(f, "{}", v),
            ScalarData::Float(_, v) => write!(f, "{}", v),
        }
    }
}
