#ifndef INCLUDE_RMS_NORM_CUH_
#define INCLUDE_RMS_NORM_CUH_

#include <cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// y = x / sqrt(mean(x^2) + epsilon) * scale
template <typename T, int BLOCK_SIZE, int DIM>
__global__ void rms_norm(
    T *out,
    T *in,
    T *scale,
    T epsilon,
    int input_size
) {
    __shared__ T buf[BLOCK_SIZE];
    __shared__ T mean_sq;
    constexpr int REPEAT = (DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int tid = threadIdx.x;
    if (input_size <= blockIdx.x * DIM) return;

    T inputs[REPEAT] = {};
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    for (int i = 0, offset = tid; i < REPEAT && offset < DIM; i++, offset += BLOCK_SIZE) {
        inputs[i] = in[blockIdx.x * DIM + offset];
    }

    {
        T sum_acc = 0;
        for (int i = 0; i < REPEAT; i++) {
            sum_acc += inputs[i] * inputs[i];
        }
        for (int s = tile32.size() / 2; 0 < s; s /= 2) {
            sum_acc += tile32.shfl_down(sum_acc, s);
        }
        buf[tid] = sum_acc;
        cg::sync(cta);
        for (int s = BLOCK_SIZE / 2; tile32.size() <= s; s >>= 1) {
            if (tid < s) {
                buf[tid] += buf[tid + s];
            }
            cg::sync(cta);
        }
        if (tid == 0) mean_sq = buf[0] / DIM;
        cg::sync(cta);
    }

    T inv_std = T(1) / sqrt(mean_sq + epsilon);

    for (int i = 0, offset = tid; i < REPEAT && offset < DIM; i++, offset += BLOCK_SIZE) {
        T y = inputs[i] * inv_std;
        y *= scale[offset];
        out[blockIdx.x * DIM + offset] = y;
    }
}

#endif
