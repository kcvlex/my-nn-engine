use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FType {
    F32 = 0,
    Bf16 = 1,
}

impl FType {
    pub fn from_data_type(dt: DataType) -> Self {
        match dt {
            DataType::Float(FloatType::F32) => FType::F32,
            DataType::Float(FloatType::BF16) => FType::Bf16,
            other => panic!("unsupported float type for W8A8 kernel: {other:?}"),
        }
    }
}

/// # Safety
/// `ptr` must be valid for `i + 1` elements of the type named by `dtype`.
#[inline(always)]
pub unsafe fn load(ptr: *const u8, i: usize, dtype: FType) -> f32 {
    match dtype {
        // bf16 is the top 16 bits of an f32; widen by shifting back.
        FType::Bf16 => {
            let bits = std::ptr::read_unaligned((ptr as *const u16).add(i)) as u32;
            f32::from_bits(bits << 16)
        }
        FType::F32 => std::ptr::read_unaligned((ptr as *const f32).add(i)),
        FType::F64 => std::ptr::read_unaligned((ptr as *const f64).add(i)) as f32,
    }
}

/// Store `val` into element `i` of a float buffer, encoding bf16 (round to
/// nearest, ties to even) if needed.
///
/// # Safety
/// `ptr` must be valid for writing `i + 1` elements of the type named by `dtype`.
#[inline(always)]
pub unsafe fn store(ptr: *mut u8, i: usize, val: f32, dtype: FType) {
    match dtype {
        FType::Bf16 => {
            let bits = val.to_bits();
            let rounded = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
            std::ptr::write_unaligned((ptr as *mut u16).add(i), rounded as u16);
        }
        FType::F32 => std::ptr::write_unaligned((ptr as *mut f32).add(i), val),
        FType::F64 => std::ptr::write_unaligned((ptr as *mut f64).add(i), val as f64),
    }
}
