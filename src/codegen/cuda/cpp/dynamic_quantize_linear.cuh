#ifndef INCLUDE_DYNAMIC_QUANTIZE_LINEAR_CUH_
#define INCLUDE_DYNAMIC_QUANTIZE_LINEAR_CUH_

// DynamicQuantizeLinear: float [...] -> y (int8/uint8) + scale + zero_point.
//
// Layout: x is treated as [axis_dim, inner]; one block handles one axis
// slice and runs a block-wide reduction. axis_dim == 1 collapses to the
// per-tensor case automatically.
//
// Multi-dim outer (e.g. axis in the middle of a 3D input) is not supported.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename T>
static __device__ __forceinline__ float to_float(T v) {
    return (float)v;
}
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

static __device__ __forceinline__ void chmax(float &a, float b) {
    a = fmaxf(a, b);
}

static __device__ __forceinline__ void chmin(float &a, float b) {
    a = fminf(a, b);
}

static __device__ __forceinline__ int clamp(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

template <typename FLOAT_T, typename Q_T, int BLOCK_SIZE, bool SYMMETRIC>
__global__ void dynamic_quantize_linear_kernel(
    Q_T *y,
    FLOAT_T *y_scale,
    Q_T *y_zero_point,
    const FLOAT_T *x,
    int axis_dim,
    int inner_size
) {
    constexpr int WARP_SIZE = 32;
    int axis = blockIdx.x;
    int tid = threadIdx.x;
    int slice_len = inner_size;

    const FLOAT_T *x_slice = x + (size_t)axis * slice_len;
    Q_T *y_slice = y + (size_t)axis * slice_len;

    float local_max_abs = 0.0f;
    float local_min = INFINITY;
    float local_max = -INFINITY;
    for (int i = tid; i < slice_len; i += BLOCK_SIZE) {
        float v = to_float<FLOAT_T>(x_slice[i]);
        if (SYMMETRIC) {
            local_max_abs = fmaxf(local_max_abs, fabsf(v));
        } else {
            local_min = fminf(local_min, v);
            local_max = fmaxf(local_max, v);
        }
    }

    __shared__ float max_buf[BLOCK_SIZE / WARP_SIZE];
    __shared__ float min_buf[BLOCK_SIZE / WARP_SIZE];
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> tile32 = cg::tiled_partition<WARP_SIZE>(cta);
    assert(BLOCK_SIZE / WARP_SIZE <= WARP_SIZE);
    if (SYMMETRIC) {
        for (int s = tile32.size() / 2; 0 < s; s >>= 1) {
            chmax(local_max_abs, tile32.shfl_down(local_max_abs, s));
        }
        if (lane == 0) {
            max_buf[warp_id] = local_max_abs;
        }
        cg::sync(cta);

        if (warp_id == 0) {
            float v = (lane < BLOCK_SIZE / WARP_SIZE) ? max_buf[lane] : 0;
            for (int s = (BLOCK_SIZE / WARP_SIZE) / 2; 0 < s; s >>= 1) {
                chmax(v, tile32.shfl_down(v, s));
            }
            if (lane == 0) {
                max_buf[0] = v;
            }
        }
    } else {
        for (int s = tile32.size() / 2; 0 < s; s >>= 1) {
            local_min = fminf(local_min, tile32.shfl_down(local_min, s));
            local_max = fmaxf(local_max, tile32.shfl_down(local_max, s));
        }
        if (lane == 0) {
            max_buf[warp_id] = local_max;
            min_buf[warp_id] = local_min;
        }
        cg::sync(cta);
        if (warp_id == 0) {
            float vmin = (lane < BLOCK_SIZE / WARP_SIZE) ? min_buf[lane] : INFINITY;
            float vmax = (lane < BLOCK_SIZE / WARP_SIZE) ? max_buf[lane] : -INFINITY;
            for (int s = (BLOCK_SIZE / WARP_SIZE) / 2; 0 < s; s >>= 1) {
                chmin(vmin, tile32.shfl_down(vmin, s));
                chmax(vmax, tile32.shfl_down(vmax, s));
            }
            if (lane == 0) {
                max_buf[0] = vmax;
                min_buf[0] = vmin;
            }
        }
    }
    cg::sync(cta);

    __shared__ float s_scale;
    __shared__ int s_zp;
    if (tid == 0) {
        if (SYMMETRIC) {
            s_scale = max_buf[0] / 127.0f;
            s_zp = 0;
        } else {
            float mn = min_buf[0];
            float mx = max_buf[0];
            // Standard ONNX clamps min <= 0 and 0 <= max (zero is in the range).
            mn = fminf(mn, 0.0f);
            mx = fmaxf(mx, 0.0f);
            float scale = (mx - mn) / 255.0f;
            if (scale == 0.0f) scale = 1.0f;
            int zp = (int)nearbyintf(-mn / scale);
            s_scale = scale;
            s_zp = clamp(zp, 0, 255);
        }
        y_scale[axis] = (FLOAT_T)s_scale;
        y_zero_point[axis] = (Q_T)s_zp;
    }
    cg::sync(cta);

    float scale = s_scale;
    int zp = s_zp;
    int qmin = SYMMETRIC ? -127 : 0;
    int qmax = SYMMETRIC ? 127 : 255;

    for (int i = tid; i < slice_len; i += BLOCK_SIZE) {
        float v = to_float<FLOAT_T>(x_slice[i]);
        int q = (int)nearbyintf(v / scale) + zp;
        y_slice[i] = (Q_T)clamp(q, qmin, qmax);
    }
}

#endif
