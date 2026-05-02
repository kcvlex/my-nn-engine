#ifndef INCLUDE_DEQUANT_MATMUL_CUH_
#define INCLUDE_DEQUANT_MATMUL_CUH_
#include <cuda.h>

// Block: BLOCK_SIZE threads (1D), reduces along K.
// Grid:  (N, M, 1). Each block computes out[m, n] for one (m, n) pair.
//
//   out[m, n] = scale[n] * sum_{k=0..K} (float) act[m, k] * (float) wq[n, k]
//
// Memory access pattern: each block reads K elements of act[m, :] and wq[n, :]
// in BLOCK_SIZE strides. Reduction in shared memory.
template <typename TAct, typename TOut, int BLOCK_SIZE>
__global__ void dequant_matmul(
    TOut *out,
    const TAct *act,
    const signed char *wq,
    const TOut *scale,
    int M,
    int N,
    int K
) {
    int n = blockIdx.x;
    int m = blockIdx.y;
    int tid = threadIdx.x;

    if (n >= N || m >= M) return;

    __shared__ float sdata[BLOCK_SIZE];

    float acc = 0.0f;
    const TAct *a_row = act + m * K;
    const signed char *w_row = wq + n * K;
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        float a = (float) a_row[k];
        float w = (float) w_row[k];
        acc += a * w;
    }

    sdata[tid] = acc;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float sc = (float) scale[n];
        out[m * N + n] = (TOut)(sdata[0] * sc);
    }
}

#endif
