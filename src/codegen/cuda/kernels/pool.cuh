#ifndef INCLUDE_POOL_CUH_
#define INCLUDE_POOL_CUH_

#include "common.cuh"
#include <cuda.h>

template <typename T>
__global__ void max_pool_kernel(
    T *out,
    T *in,
    T init,
    int nbatch,
    int channels,
    int height,
    int width,
    int o_height,
    int o_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int output_sz = nbatch * channels * o_height * o_width;
    if (output_sz <= id) return;

    int tmp = id;
    int o_w_idx, o_h_idx, o_c_idx, o_b_idx;
    divmod(tmp, o_width, &tmp, &o_w_idx);
    divmod(tmp, o_height, &tmp, &o_h_idx);
    divmod(tmp, channels, &o_b_idx, &o_c_idx);

    int i_w_begin = o_w_idx * stride_w - pad_w;
    int i_w_end = i_w_begin + kernel_w;
    int i_h_begin = o_h_idx * stride_h - pad_h;
    int i_h_end = i_h_begin + kernel_h;

    T max_val = init;
    for (int h = i_h_begin; h < i_h_end; ++h) {
        for (int w = i_w_begin; w < i_w_end; ++w) {
            if (0 <= h && h < height && 0 <= w && w < width) {
                int in_idx = ((o_b_idx * channels + o_c_idx) * height + h) * width + w;
                max_val = max(max_val, in[in_idx]);
            }
        }
    }

    out[id] = max_val;
}

#endif
