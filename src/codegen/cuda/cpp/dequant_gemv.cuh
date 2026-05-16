#ifndef INCLUDE_DEQUANT_GEMV_CUH_
#define INCLUDE_DEQUANT_GEMV_CUH_

// Specialized for M=1 (decode)

#include <cuda.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32

template <typename TAct, typename TOut>
__global__ void dequant_gemv(
    TOut *out,
    const TAct *act,
    const int8_t *wq,
    const TOut *scale,
    int N,
    int K
) {
    assert(K % 16 == 0);
    int n = blockIdx.x;
    if (N <= n) {
        return;
    }
    int tid = threadIdx.x;

    constexpr int VEC = 16;
    const int8_t *w_row = wq + n * K;
    float s = (float)scale[n];

    float acc = 0.0f;
    for (int k_base = 0; k_base < K; k_base += WARP_SIZE * VEC) {
        int k = k_base + tid * VEC;
        if (k + VEC <= K) {
            int4 a_lo = *(const int4 *)&act[k];
            int4 a_hi = *(const int4 *)&act[k + 8];
            int4 w_packed = *(const int4 *)&w_row[k];
            const TAct *a0 = reinterpret_cast<const TAct *>(&a_lo);
            const TAct *a1 = reinterpret_cast<const TAct *>(&a_hi);
            const int8_t *w = reinterpret_cast<const int8_t *>(&w_packed);

            for (int i = 0; i < 8; i++) {
                acc += (float)a0[i] * (float)w[i];
                acc += (float)a1[i] * (float)w[i + 8];
            }
        }
    }
    acc *= s;

    acc += __shfl_down_sync(0xffffffff, acc, 16);
    acc += __shfl_down_sync(0xffffffff, acc, 8);
    acc += __shfl_down_sync(0xffffffff, acc, 4);
    acc += __shfl_down_sync(0xffffffff, acc, 2);
    acc += __shfl_down_sync(0xffffffff, acc, 1);

    if (tid == 0) {
        out[n] = (TOut)acc;
    }
}

#endif
