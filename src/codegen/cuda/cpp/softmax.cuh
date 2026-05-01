#ifndef INCLUDE_SOFTMAX_CUH_
#define INCLUDE_SOFTMAX_CUH_
#include <cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename T, int BLOCK_SIZE, int DIM, int STRIDE>
__global__ void softmax(
    T *out,
    T *in,
    int input_size
) {
    __shared__ float buf[BLOCK_SIZE];
    __shared__ float axis_max;
    __shared__ float axis_sum;

    int tid = threadIdx.x;
    int base = (blockIdx.x / STRIDE) * DIM * STRIDE + (blockIdx.x % STRIDE);

#define AXIS_POS(I) ((I) * BLOCK_SIZE + tid)
#define ABS_POS(I) (base + AXIS_POS(I) * STRIDE)

    constexpr int REPEAT = (DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float inputs[REPEAT] = {};
    for (int i = 0; i < REPEAT; i++) {
        if (AXIS_POS(i) < DIM) {
            inputs[i] = (float)in[ABS_POS(i)];
        }
    }

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    float max_acc = -INFINITY;
    for (int i = 0; i < REPEAT; i++) {
        int axis_pos = i * BLOCK_SIZE + tid;
        if (axis_pos < DIM) {
            max_acc = max(max_acc, inputs[i]);
        }
    }
    for (int s = tile32.size() / 2; 0 < s; s /= 2) {
        float other = tile32.shfl_down(max_acc, s);
        max_acc = max(max_acc, other);
    }
    buf[tid] = max_acc;
    cg::sync(cta);
    for (int s = BLOCK_SIZE / 2; tile32.size() <= s; s >>= 1) {
        if (tid < s) {
            buf[tid] = max(buf[tid], buf[tid + s]);
        }
        cg::sync(cta);
    }

    if (tid == 0) axis_max = buf[0];
    cg::sync(cta);

    float sum_acc = 0.0f;
    for (int i = 0; i < REPEAT; i++) {
        int axis_pos = i * BLOCK_SIZE + tid;
        if (axis_pos < DIM) {
            inputs[i] = expf(inputs[i] - axis_max);
            sum_acc += inputs[i];
        }
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
    if (tid == 0) axis_sum = buf[0];
    cg::sync(cta);

    for (int i = 0; i < REPEAT; i++) {
        if (AXIS_POS(i) < DIM) {
            out[ABS_POS(i)] = (T)(inputs[i] / axis_sum);
        }
    }

#undef ABS_POS
#undef AXIS_POS
}

#endif
