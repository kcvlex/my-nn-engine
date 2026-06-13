use super::ftype::FType;
use super::ftype::{self};

/// Quantize `x` `[M, K]` (f32 or bf16, per `dtype`) to int8 `y` with per-row
/// symmetric `scale` `[M]` (same `dtype` as `x`):
///   `scale[m] = max_k |x[m,k]| / 127`,  `y[m,k] = round(x[m,k] / scale[m])`.
/// A zero row yields `scale = 1` (avoids div-by-zero; all outputs are 0).
///
/// # Safety
/// `y` points to `m*k` i8; `scale` to `m` and `x` to `m*k` elements of the
/// float type named by `dtype`. All must be valid.
#[no_mangle]
pub unsafe extern "C" fn mynn_dynquant_i8(
    y: *mut i8,
    scale: *mut u8,
    x: *const u8,
    m: usize,
    k: usize,
    dtype: FType,
) {
    let y = unsafe { std::slice::from_raw_parts_mut(y, m * k) };

    for mi in 0..m {
        let base = mi * k;
        let mut amax = 0f32;
        for j in 0..k {
            amax = amax.max(unsafe { ftype::load(x, base + j, dtype) }.abs());
        }
        let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let inv = sc.recip();
        for j in 0..k {
            let v = unsafe { ftype::load(x, base + j, dtype) };
            y[base + j] = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        unsafe { ftype::store(scale, mi, sc, dtype) };
    }
}
