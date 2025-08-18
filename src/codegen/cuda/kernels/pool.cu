#include "common.h"
#include <cuda.h>

#pragma once

__device__ void divmod(i64 x, i64 y, i64 *q, i64 *r) {
    *q = x / y;
    *r = x % y;
}

template <typename T>
__global__ void max_pool_kernel(
    T *out,
    T *in,
    T init,
    i64 nbatch,
    i64 channels,
    i64 height,
    i64 width,
    i64 o_height,
    i64 o_width,
    i64 kernel_h,
    i64 kernel_w,
    i64 stride_h,
    i64 stride_w,
    i64 pad_h,
    i64 pad_w
) {
    i64 id = blockIdx.x * blockDim.x + threadIdx.x;
    i64 output_sz = nbatch * channels * o_height * o_width;
    if (output_sz <= id) return;

    i64 tmp = id;
    i64 o_w_idx, o_h_idx, o_c_idx, o_b_idx;
    divmod(tmp, o_width, &tmp, &o_w_idx);
    divmod(tmp, o_height, &tmp, &o_h_idx);
    divmod(tmp, channels, &o_b_idx, &o_c_idx);

    i64 i_w_begin = o_w_idx * stride_w - pad_w;
    i64 i_w_end = i_w_begin + kernel_w;
    i64 i_h_begin = o_h_idx * stride_h - pad_h;
    i64 i_h_end = i_h_begin + kernel_h;

    T max_val = init;
    for (i64 h = i_h_begin; h < i_h_end; ++h) {
        for (i64 w = i_w_begin; w < i_w_end; ++w) {
            if (0 <= h && h < height && 0 <= w && w < width) {
                i64 in_idx = ((o_b_idx * channels + o_c_idx) * height + h) * width + w;
                max_val = max(max_val, in[in_idx]);
            }
        }
    }

    out[id] = max_val;
}
