#ifndef INCLUDE_QUANTIZED_GEMV_INT8_CUH_
#define INCLUDE_QUANTIZED_GEMV_INT8_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>

namespace cg = cooperative_groups;

// One block = one warp = one output column n.
__global__ void quantized_gemv_int8(
    __nv_bfloat16 *out,
    const int8_t *lhs,
    const __nv_bfloat16 *lhs_scale,
    const int8_t *rhs,
    const __nv_bfloat16 *rhs_scale,
    int N,
    int K
) {
    assert(K % 16 == 0);
    int n = blockIdx.x;
    if (N <= n) {
        return;
    }
    constexpr int WARP_SIZE = 32;
    constexpr int VEC = 16;

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(cta);

    const int8_t *w_row = rhs + (size_t)n * K;
    int32_t acc = 0;

    for (int k = warp.thread_rank() * VEC; k + VEC <= K; k += WARP_SIZE * VEC) {
        int4 a = *reinterpret_cast<const int4 *>(&lhs[k]);
        int4 w = *reinterpret_cast<const int4 *>(&w_row[k]);
        acc = __dp4a(a.x, w.x, acc);
        acc = __dp4a(a.y, w.y, acc);
        acc = __dp4a(a.z, w.z, acc);
        acc = __dp4a(a.w, w.w, acc);
    }

    acc = cg::reduce(warp, acc, cg::plus<int32_t>());

    if (warp.thread_rank() == 0) {
        float result = (float)acc * (float)lhs_scale[0] * (float)rhs_scale[n];
        out[n] = __float2bfloat16(result);
    }
}

#endif
