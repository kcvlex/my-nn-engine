//! Symmetric int8 x int8 matmul (W8A8): the fast CPU decode path. Memory-bound
//! -- streams the int8 weight from DDR once, so a plain widening dot under AVX2
//! already saturates DDR bandwidth (no tiling / VNNI needed).

use rayon::prelude::*;

use super::ftype::FType;
use super::ftype::{self};

macro_rules! dot_body {
    ($w:expr, $a:expr) => {{
        let mut acc: i32 = 0;
        for i in 0..$w.len() {
            acc += $w[i] as i32 * $a[i] as i32;
        }
        acc
    }};
}

fn dot_scalar(w: &[i8], a: &[i8]) -> i32 {
    dot_body!(w, a)
}

#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(w: &[i8], a: &[i8]) -> i32 {
    dot_body!(w, a)
}

/// `out[m, n] = (sum_k lhs[m, k] * rhs[n, k]) * lhs_scale[m] * rhs_scale[n]`.
///
/// `rhs` is the row-major `[N, K]` int8 weight; `lhs` the `[M, K]` int8
/// activation. The scales and `out` carry the model float type named by
/// `dtype` (f32 / bf16); `lhs`/`rhs` are int8. Decode is `M == 1` (GEMV); larger
/// `M` loops over rows. Symmetric quantization -- no zero points. N output rows
/// are split across the rayon pool.
///
/// # Safety
/// `out` is `m*n` and the scales are `m` / `n` elements of `dtype`; `lhs` is
/// `m*k` i8 and `rhs` is `n*k` i8. All valid for those lengths, `out`
/// non-aliasing.
#[no_mangle]
pub unsafe extern "C" fn mynn_qgemv_i8i8(
    out: *mut u8,
    lhs: *const i8,
    lhs_scale: *const u8,
    rhs: *const i8,
    rhs_scale: *const u8,
    m: usize,
    n: usize,
    k: usize,
    dtype: FType,
) {
    let lhs = unsafe { std::slice::from_raw_parts(lhs, m * k) };
    let rhs = unsafe { std::slice::from_raw_parts(rhs, n * k) };

    let avx2 = is_x86_feature_detected!("avx2");
    // Raw pointers aren't Send; pass addresses into the rayon closure. Each `ni`
    // writes a distinct `out` element and only reads shared data, so the
    // concurrent raw writes have no data race.
    let out_addr = out as usize;
    let rhs_scale_addr = rhs_scale as usize;

    for mi in 0..m {
        let a = &lhs[mi * k..mi * k + k];
        let asc = unsafe { ftype::load(lhs_scale, mi, dtype) };
        (0..n).into_par_iter().for_each(|ni| {
            let w = &rhs[ni * k..ni * k + k];
            let acc = if avx2 {
                unsafe { dot_avx2(w, a) }
            } else {
                dot_scalar(w, a)
            };
            let wsc = unsafe { ftype::load(rhs_scale_addr as *const u8, ni, dtype) };
            unsafe {
                ftype::store(
                    out_addr as *mut u8,
                    mi * n + ni,
                    acc as f32 * asc * wsc,
                    dtype,
                )
            };
        });
    }
}
