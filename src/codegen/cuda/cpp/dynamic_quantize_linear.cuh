#ifndef INCLUDE_DYNAMIC_QUANTIZE_LINEAR_CUH_
#define INCLUDE_DYNAMIC_QUANTIZE_LINEAR_CUH_

// DynamicQuantizeLinear: float [...] -> y (int8/uint8) + scale + zero_point.
//
//   Standard ONNX (symmetric=false, axis=None):
//     scale       = (max(x) - min(x)) / 255
//     zero_point  = round(-min(x) / scale)  clipped to [0, 255]
//     y[i]        = clip(round(x[i] / scale) + zero_point, 0, 255)
//
//   Symmetric variant (symmetric=true):
//     scale       = max(|x|) / 127
//     zero_point  = 0
//     y[i]        = clip(round(x[i] / scale), -127, 127)
//
//   Per-axis (axis_dim > 1): the above is computed independently for each
//   slice along the axis. With axis_dim = 1 (per-tensor), there is one
//   global scale/zero_point.
//
// Layout: x is treated as [axis_dim, inner]; one block handles one axis
// slice and runs a block-wide reduction. axis_dim == 1 collapses to the
// per-tensor case automatically.
//
// Multi-dim outer (e.g. axis in the middle of a 3D input) is not handled
// in this kernel and is rejected by the dispatch.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

template <typename T>
static __device__ __forceinline__ float to_float(T v) {
    return (float)v;
}
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
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
    int axis = blockIdx.x;  // [0, axis_dim)
    int tid = threadIdx.x;
    int slice_len = inner_size;

    const FLOAT_T *x_slice = x + (size_t)axis * slice_len;
    Q_T *y_slice = y + (size_t)axis * slice_len;

    // Phase 1: reduce across the inner_size slice.
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

    // Block reduce
    __shared__ float warp_buf[BLOCK_SIZE / 32 * 3];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (SYMMETRIC) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            local_max_abs = fmaxf(local_max_abs, __shfl_xor_sync(0xFFFFFFFF, local_max_abs, o));
        }
        if (lane == 0) warp_buf[warp_id] = local_max_abs;
        __syncthreads();
        if (warp_id == 0) {
            float v = (lane < BLOCK_SIZE / 32) ? warp_buf[lane] : 0.0f;
            #pragma unroll
            for (int o = (BLOCK_SIZE / 32) / 2; o > 0; o >>= 1) {
                v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, o));
            }
            if (lane == 0) warp_buf[0] = v;
        }
    } else {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            local_min = fminf(local_min, __shfl_xor_sync(0xFFFFFFFF, local_min, o));
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, o));
        }
        if (lane == 0) {
            warp_buf[warp_id] = local_min;
            warp_buf[warp_id + BLOCK_SIZE / 32] = local_max;
        }
        __syncthreads();
        if (warp_id == 0) {
            float vmn = (lane < BLOCK_SIZE / 32) ? warp_buf[lane] : 0.0f;
            float vmx = (lane < BLOCK_SIZE / 32)
                ? warp_buf[lane + BLOCK_SIZE / 32]
                : 0.0f;
            #pragma unroll
            for (int o = (BLOCK_SIZE / 32) / 2; o > 0; o >>= 1) {
                vmn = fminf(vmn, __shfl_xor_sync(0xFFFFFFFF, vmn, o));
                vmx = fmaxf(vmx, __shfl_xor_sync(0xFFFFFFFF, vmx, o));
            }
            if (lane == 0) {
                warp_buf[0] = vmn;
                warp_buf[1] = vmx;
            }
        }
    }
    __syncthreads();

    // Compute scale + zp (one thread broadcasts via smem).
    __shared__ float s_scale;
    __shared__ int s_zp;
    if (tid == 0) {
        if (SYMMETRIC) {
            float max_abs = warp_buf[0];
            float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            s_scale = scale;
            s_zp = 0;
        } else {
            float mn = warp_buf[0];
            float mx = warp_buf[1];
            // Standard ONNX clamps min <= 0 and max >= 0 (zero is in the range).
            mn = fminf(mn, 0.0f);
            mx = fmaxf(mx, 0.0f);
            float scale = (mx - mn) / 255.0f;
            if (scale == 0.0f) scale = 1.0f;
            int zp = (int)nearbyintf(-mn / scale);
            if (zp < 0) zp = 0;
            if (zp > 255) zp = 255;
            s_scale = scale;
            s_zp = zp;
        }
        y_scale[axis] = (FLOAT_T)s_scale;
        y_zero_point[axis] = (Q_T)s_zp;
    }
    __syncthreads();

    float scale = s_scale;
    int zp = s_zp;
    float inv_scale = 1.0f / scale;
    int qmin = SYMMETRIC ? -127 : 0;
    int qmax = SYMMETRIC ? 127 : 255;

    // Phase 2: quantize
    for (int i = tid; i < slice_len; i += BLOCK_SIZE) {
        float v = to_float<FLOAT_T>(x_slice[i]);
        int q = (int)nearbyintf(v * inv_scale) + zp;
        if (q < qmin) q = qmin;
        if (q > qmax) q = qmax;
        y_slice[i] = (Q_T)q;
    }
}

#endif
