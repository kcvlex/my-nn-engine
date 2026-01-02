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
    __shared__ T buf[BLOCK_SIZE];
    __shared__ T axis_max;
    __shared__ T axis_sum;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (input_size <= idx) return;

    constexpr int REPEAT = (DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    T inputs[REPEAT] = {};
    for (int i = 0, offset = idx; i < REPEAT; i++, offset += BLOCK_SIZE * STRIDE) {
        if (offset < input_size) {
            inputs[i] = in[offset];
        }
    }

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    T max_acc = std::numeric_limits<T>::min();
    for (int i = 0; i < REPEAT; i++) {
        max_acc = max(max_acc, inputs[i]);
    }
    for (int s = tile32.size() / 2; 0 < s; s /= 2) {
        T other = tile32.shfl_down(max_acc, s);
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

    T sum_acc = 0;
    for (int i = 0; i < REPEAT; i++) {
        inputs[i] = exp(inputs[i] - axis_max);
        sum_acc += inputs[i];
    }
    for (int s = tile32.size() / 2; 0 < s; s /= 2) {
        T other = tile32.shfl_down(sum_acc, s);
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

    for (int i = 0, offset = idx; i < REPEAT; i++, offset += BLOCK_SIZE * STRIDE) {
        if (offset < input_size) {
            out[offset] = inputs[i] / axis_sum;
        }
    }
}

#endif
