//! Shared float element-type handling for the W8A8 kernels. The activation,
//! the per-row / per-channel scales, and the matmul output all carry the
//! model's float type (f32 or bf16). It crosses the C ABI as a fieldless
//! `#[repr(u32)]` enum (FFI-safe, unlike the graph's `DataType`).

use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FType {
    F32 = 0,
    Bf16 = 1,
}

impl FType {
    /// Map a graph float `DataType` to the kernel tag. Panics on unsupported
    /// types -- the W8A8 path only produces f32 / bf16.
    pub fn from_data_type(dt: DataType) -> Self {
        match dt {
            DataType::Float(FloatType::F32) => FType::F32,
            DataType::Float(FloatType::BF16) => FType::Bf16,
            other => panic!("unsupported float type for W8A8 kernel: {other:?}"),
        }
    }
}

/// Load element `i` of a float buffer as f32, decoding bf16 if needed.
///
/// # Safety
/// `ptr` must be valid for `i + 1` elements of the type named by `dtype`.
#[inline(always)]
pub unsafe fn load(ptr: *const u8, i: usize, dtype: FType) -> f32 {
    unsafe {
        match dtype {
            // bf16 is the top 16 bits of an f32; widen by shifting back.
            FType::Bf16 => f32::from_bits((*(ptr as *const u16).add(i) as u32) << 16),
            FType::F32 => *(ptr as *const f32).add(i),
        }
    }
}

/// Store `val` into element `i` of a float buffer, encoding bf16 (round to
/// nearest, ties to even) if needed.
///
/// # Safety
/// `ptr` must be valid for writing `i + 1` elements of the type named by `dtype`.
#[inline(always)]
pub unsafe fn store(ptr: *mut u8, i: usize, val: f32, dtype: FType) {
    unsafe {
        match dtype {
            FType::Bf16 => {
                let bits = val.to_bits();
                let rounded = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
                *(ptr as *mut u16).add(i) = rounded as u16;
            }
            FType::F32 => *(ptr as *mut f32).add(i) = val,
        }
    }
}
