#ifndef INCLUDE_REDUCE_CUH_
#define INCLUDE_REDUCE_CUH_

#include "common.cuh"
#include <cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

enum class ReduceType {
    Max,
    Mean,
};

template <typename T, ReduceType RT>
__device__ T reduce_init() {
    if constexpr (RT == ReduceType::Max) {
        return std::numeric_limits<T>::min();
    } else if constexpr (RT == ReduceType::Mean) {
        return static_cast<T>(0);
    } else {
        // Dummy.
        return static_cast<T>(0);
    }
}

template <typename T, ReduceType RT>
__device__ T reduce_op(T a, T b) {
    if constexpr (RT == ReduceType::Max) {
        return std::max(a, b);
    } else if constexpr (RT == ReduceType::Mean) {
        return a + b;
    } else {
        // Dummy.
        return a;
    }
}

template <typename T, ReduceType RT, size_t BLOCK_SIZE>
__global__ void reduce2d(
    T *out,
    T *in,
    i64 row,
    i64 col
) {
    extern __shared__ T shared_data[];
    i64 tid = threadIdx.x;
    i64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    i64 ceil_col = (col + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    i64 row_id = idx / ceil_col;
    if (row <= row_id) return;

    T acc_block = reduce_init<T, RT>();
    for (i64 i = tid; i < ceil_col; i += blockDim.x) {
        if (i < col) {
            acc_block = reduce_op<T, RT>(acc_block, in[row_id * col + i]);
        }
    }
    
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    for (int s = 16; 0 < s; s >>= 1) {
        acc_block = reduce_op<T, RT>(acc_block, tile32.shfl_down(acc_block, s));
    }

    shared_data[tid] = acc_block;
    cg::sync(cta);

    for (i64 s = BLOCK_SIZE / 2; 32 <= s; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = reduce_op<T, RT>(shared_data[tid], shared_data[tid + s]);
        }
        cg::sync(cta);
    }

    if (tid == 0) {
        T result = shared_data[0];
        if constexpr (RT == ReduceType::Mean) {
            result = result / static_cast<T>(col);
        }
        out[row_id] = result;
    }
}

#endif
