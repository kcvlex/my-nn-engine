#ifndef INCLUDE_LAYER_NORM_CUH_
#define INCLUDE_LAYER_NORM_CUH_

#include <cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename T, int BLOCK_SIZE, int DIM>
__global__ void layer_norm(
    T *out,
    T *in,
    T *scale,
    T *bias,
    float epsilon,
    int input_size
) {
    __shared__ float buf[BLOCK_SIZE];
    __shared__ float mean;
    constexpr int REPEAT = (DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int tid = threadIdx.x;
    if (input_size <= blockIdx.x * DIM) return;

    float inputs[REPEAT] = {};
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    for (int i = 0, offset = tid; i < REPEAT && offset < DIM; i++, offset += BLOCK_SIZE) {
        inputs[i] = (float)in[blockIdx.x * DIM + offset];
    }

    {
        float sum_acc = 0.0f;
        for (int i = 0; i < REPEAT; i++) {
            sum_acc += inputs[i];
        }
        for (int s = tile32.size() / 2; 0 < s; s /= 2) {
            float other = tile32.shfl_down(sum_acc, s);
            sum_acc += other;
        }
        buf[tid] = sum_acc;
        cg::sync(cta);
        for (int s = BLOCK_SIZE / 2; tile32.size() <= s; s >>= 1) {
            if (tid < s) {
                buf[tid] += buf[tid + s];
            }
            cg::sync(cta);
        }
        if (tid == 0) mean = buf[0] / DIM;
        cg::sync(cta);
    }

    float vars[REPEAT] = {};
    for (int i = 0; i < REPEAT && i * BLOCK_SIZE + tid < DIM; i++) {
        inputs[i] -= mean;
        vars[i] = inputs[i] * inputs[i];
    }

    {
        float sum_acc = 0.0f;
        for (int i = 0; i < REPEAT; i++) {
            sum_acc += vars[i];
        }
        for (int s = tile32.size() / 2; 0 < s; s /= 2) {
            float other = tile32.shfl_down(sum_acc, s);
            sum_acc += other;
        }
        buf[tid] = sum_acc;
        cg::sync(cta);
        for (int s = BLOCK_SIZE / 2; tile32.size() <= s; s >>= 1) {
            if (tid < s) {
                buf[tid] += buf[tid + s];
            }
            cg::sync(cta);
        }
        if (tid == 0) mean = buf[0] / DIM;
        cg::sync(cta);
    }

    float std_dev = sqrtf(mean + epsilon);

    for (int i = 0, offset = tid; i < REPEAT && offset < DIM; i++, offset += BLOCK_SIZE) {
        float y = inputs[i] / std_dev;
        y *= (float)scale[offset];
        y += (float)bias[offset];
        out[blockIdx.x * DIM + offset] = (T)y;
    }
}

#endif
