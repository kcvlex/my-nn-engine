#ifndef INCLUDE_REDUCE_CUH_
#define INCLUDE_REDUCE_CUH_
#include <cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename T>
struct ReduceOpSum {
    using AccT = float;
    __device__ static AccT init() { return 0.0f; }
    __device__ static AccT op(AccT a, T b) { return a + (float)b; }
    __device__ static AccT combine(AccT a, AccT b) { return a + b; }
    __device__ static T finalize(AccT x, int col) { return (T)x; }
};

template <typename T>
struct ReduceOpMean {
    using AccT = float;
    __device__ static AccT init() { return 0.0f; }
    __device__ static AccT op(AccT a, T b) { return a + (float)b; }
    __device__ static AccT combine(AccT a, AccT b) { return a + b; }
    __device__ static T finalize(AccT x, int col) { return (T)(x / (float)col); }
};

template <typename T>
struct ReduceOpMax {
    using AccT = float;
    __device__ static AccT init() { return -INFINITY; }
    __device__ static AccT op(AccT a, T b) { return max(a, (float)b); }
    __device__ static AccT combine(AccT a, AccT b) { return max(a, b); }
    __device__ static T finalize(AccT x, int col) { return (T)x; }
};

template <typename T, typename Op, int BLOCK_SIZE>
__global__ void reduce_matrix_kernel(T *out, const T *in, int col) {
    using AccT = typename Op::AccT;
    __shared__ AccT shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int row_id = blockIdx.x;
    int ceil_col = ((col + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    AccT acc = Op::init();
    for (int i = tid; i < ceil_col; i += BLOCK_SIZE) {
        if (i < col) {
            acc = Op::op(acc, in[row_id * col + i]);
        }
    }

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    for (int s = tile32.size() / 2; s > 0; s >>= 1) {
        acc = Op::combine(acc, tile32.shfl_down(acc, s));
    }

    shared[tid] = acc;
    cg::sync(cta);

    for (int s = BLOCK_SIZE / 2; tile32.size() <= s; s >>= 1) {
        if (tid < s) {
            shared[tid] = Op::combine(shared[tid], shared[tid + s]);
        }
        cg::sync(cta);
    }

    if (tid == 0) {
        out[row_id] = Op::finalize(shared[0], col);
    }
}

#endif
