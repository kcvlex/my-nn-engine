#ifndef INCLUDE_REDUCE_CUH_
#define INCLUDE_REDUCE_CUH_
#include <cuda.h>
#include <cooperative_groups.h>
#include <limits>

namespace cg = cooperative_groups;

template <typename T>
struct ReduceOpSum {
    __device__ static T init() { return T(0); }
    __device__ static T op(T a, T b) { return a + b; }
    __device__ static T finalize(T x, int col) { return x; }
};

template <typename T>
struct ReduceOpMean {
    __device__ static T init() { return T(0); }
    __device__ static T op(T a, T b) { return a + b; }
    __device__ static T finalize(T x, int col) { return x / T(col); }
};

template <typename T>
struct ReduceOpMax {
    __device__ static T init() { return std::numeric_limits<T>::min(); }
    __device__ static T op(T a, T b) { return max(a, b); }
    __device__ static T finalize(T x, int col) { return x; }
};

template <typename T, typename Op, int BLOCK_SIZE>
__global__ void reduce_matrix_kernel(T *out, const T *in, int col) {
    __shared__ T shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int row_id = blockIdx.x;
    int ceil_col = ((col + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    T acc = Op::init();
    for (int i = tid; i < ceil_col; i += BLOCK_SIZE) {
        if (i < col) {
            acc = Op::op(acc, in[row_id * col + i]);
        }
    }

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    for (int s = tile32.size() / 2; s > 0; s >>= 1) {
        acc = Op::op(acc, tile32.shfl_down(acc, s));
    }

    shared[tid] = acc;
    cg::sync(cta);

    for (int s = BLOCK_SIZE / 2; tile32.size() <= s; s >>= 1) {
        if (tid < s) {
            shared[tid] = Op::op(shared[tid], shared[tid + s]);
        }
        cg::sync(cta);
    }

    if (tid == 0) {
        out[row_id] = Op::finalize(shared[0], col);
    }
}

#endif
