#ifndef INCLUDE_QUANTIZED_MATMUL_INT8_CUH_
#define INCLUDE_QUANTIZED_MATMUL_INT8_CUH_

// Symmetric INT8 matmul on Tensor Cores: mma.m16n8k32.s32.s8.s8.s32 with a
// combined-scale epilogue. A is per-row scaled, B is per-channel scaled.
//
// Lane layout (PTX m16n8k32 s8.s8.s32, A=.row, B=.col):
//   A frag (4 uint32 / lane):
//     a0 = A[r=t/4,     c=4(t%4)..+3]
//     a1 = A[r=t/4 + 8, c=4(t%4)..+3]
//     a2 = A[r=t/4,     c=4(t%4)+16..+19]
//     a3 = A[r=t/4 + 8, c=4(t%4)+16..+19]
//   B frag (2 uint32 / lane):
//     b0 = B[k=4(t%4)..+3,     n=t/4]
//     b1 = B[k=4(t%4)+16..+19, n=t/4]
//   C accum (4 s32 / lane):
//     c0,c1 at (r=t/4,   c=2(t%4)..+1)
//     c2,c3 at (r=t/4+8, c=2(t%4)..+1)

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdint>

#define QMM_WARP_SIZE 32
#define QMM_BK 64

static __device__ __forceinline__ void qmm_ldmatrix_x4(
    uint32_t out[4], uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(smem_addr)
    );
}

static __device__ __forceinline__ void qmm_ldmatrix_x2(
    uint32_t out[2], uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(out[0]), "=r"(out[1])
        : "r"(smem_addr)
    );
}

static __device__ __forceinline__ void qmm_mma_m16n8k32(
    int32_t c[4], const uint32_t a[4], const uint32_t b[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}

template <int BM>
static __device__ __forceinline__ void qmm_load_a(
    int8_t (*a_smem)[QMM_BK],
    const int8_t *a,
    int M, int K, int block_row, int k
) {
    constexpr int VEC = 16;
    for (int i = threadIdx.x * VEC; i < BM * QMM_BK; i += blockDim.x * VEC) {
        int r = i / QMM_BK;
        int c = i % QMM_BK;
        int gr = block_row + r;
        int gc = k + c;
        size_t valid = (gr < M && gc < K) ? min(K - gc, VEC) : 0;
        __pipeline_memcpy_async(
            &a_smem[r][c],
            &a[gr * K + gc],
            16,
            16 - valid
        );
    }
}

template <int BN>
static __device__ __forceinline__ void qmm_load_b(
    int8_t (*b_smem)[QMM_BK],
    const int8_t *b,
    int N, int K, int block_col, int k
) {
    constexpr int VEC = 16;
    for (int i = threadIdx.x * VEC; i < BN * QMM_BK; i += blockDim.x * VEC) {
        int r = i / QMM_BK;
        int c = i % QMM_BK;
        int gr = block_col + r;
        int gc = k + c;
        size_t valid = (gr < N && gc < K) ? min(K - gc, VEC) : 0;
        __pipeline_memcpy_async(
            &b_smem[r][c],
            &b[gr * K + gc],
            16,
            16 - valid
        );
    }
}

template <int BM, int BN, int STAGES>
struct alignas(16) QmmSmem {
    int8_t a[STAGES][BM][QMM_BK];
    int8_t b[STAGES][BN][QMM_BK];
    __nv_bfloat16 lhs_scale[BM];
    __nv_bfloat16 rhs_scale[BN];
};

template <int BM, int BN, int WARP_M, int WARP_N, int STAGES>
__global__ void quantized_matmul_int8(
    __nv_bfloat16 *out,
    const int8_t *lhs,
    const __nv_bfloat16 *lhs_scale,    // M bf16
    const int8_t *rhs,
    const __nv_bfloat16 *rhs_scale,    // N bf16
    int M, int N, int K
) {
    static_assert(WARP_M % 16 == 0);
    static_assert(WARP_N % 8 == 0);
    static_assert(BM % WARP_M == 0);
    static_assert(BN % WARP_N == 0);
    static_assert(QMM_BK % 32 == 0);
    static_assert(STAGES >= 1);

    constexpr int WM_ITER = WARP_M / 16;
    constexpr int WN_ITER = WARP_N / 8;
    constexpr int N_WARPS_N = BN / WARP_N;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int warp_id = threadIdx.x / QMM_WARP_SIZE;
    int lane = threadIdx.x % QMM_WARP_SIZE;
    int warp_row = warp_id / N_WARPS_N;
    int warp_col = warp_id % N_WARPS_N;

    extern __shared__ __align__(16) char dyn_smem[];
    auto &smem = *reinterpret_cast<QmmSmem<BM, BN, STAGES> *>(dyn_smem);

    // Preload scales
    for (int i = threadIdx.x; i < BM; i += blockDim.x) {
        int gm = block_row + i;
        smem.lhs_scale[i] = (gm < M) ? lhs_scale[gm] : __float2bfloat16(0.0f);
    }
    for (int i = threadIdx.x; i < BN; i += blockDim.x) {
        int gn = block_col + i;
        smem.rhs_scale[i] = (gn < N) ? rhs_scale[gn] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    int32_t c[WM_ITER][WN_ITER][4];
    #pragma unroll
    for (int wm = 0; wm < WM_ITER; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WN_ITER; wn++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) c[wm][wn][i] = 0;
        }
    }

    int n_iters = (K + QMM_BK - 1) / QMM_BK;

    // Prologue
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < n_iters) {
            int kk = s * QMM_BK;
            qmm_load_a<BM>(smem.a[s], lhs, M, K, block_row, kk);
            qmm_load_b<BN>(smem.b[s], rhs, N, K, block_col, kk);
        }
        __pipeline_commit();
    }

    int write_stage = (STAGES - 1) % STAGES;
    int read_stage = 0;

    for (int it = 0; it < n_iters; it++) {
        int next_it = it + STAGES - 1;
        if (next_it < n_iters) {
            int kk = next_it * QMM_BK;
            qmm_load_a<BM>(smem.a[write_stage], lhs, M, K, block_row, kk);
            qmm_load_b<BN>(smem.b[write_stage], rhs, N, K, block_col, kk);
        }
        __pipeline_commit();
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        int8_t (*a_cur)[QMM_BK] = smem.a[read_stage];
        int8_t (*b_cur)[QMM_BK] = smem.b[read_stage];

        #pragma unroll
        for (int wk = 0; wk < QMM_BK; wk += 32) {
            uint32_t a_frag[WM_ITER][4];
            uint32_t b_frag[WN_ITER][2];

            #pragma unroll
            for (int wm = 0; wm < WM_ITER; wm++) {
                int row_base = warp_row * WARP_M + wm * 16;
                int frag_id = lane / 8;
                int frag_row = lane % 8;
                int r_off = (frag_id & 1) * 8;
                int c_off = (frag_id >> 1) * 16;
                int smem_r = row_base + r_off + frag_row;
                int smem_c = wk + c_off;
                uint32_t addr = __cvta_generic_to_shared(
                    &a_cur[smem_r][smem_c]
                );
                qmm_ldmatrix_x4(a_frag[wm], addr);
            }

            #pragma unroll
            for (int wn = 0; wn < WN_ITER; wn++) {
                int col_base = warp_col * WARP_N + wn * 8;
                int frag_id = (lane / 8) & 1;
                int frag_row = lane % 8;
                int smem_n = col_base + frag_row;
                int smem_k = wk + frag_id * 16;
                uint32_t addr = __cvta_generic_to_shared(
                    &b_cur[smem_n][smem_k]
                );
                qmm_ldmatrix_x2(b_frag[wn], addr);
            }

            #pragma unroll
            for (int wm = 0; wm < WM_ITER; wm++) {
                #pragma unroll
                for (int wn = 0; wn < WN_ITER; wn++) {
                    qmm_mma_m16n8k32(c[wm][wn], a_frag[wm], b_frag[wn]);
                }
            }
        }
        __syncthreads();

        write_stage = (write_stage + 1) % STAGES;
        read_stage = (read_stage + 1) % STAGES;
    }

    // Epilogue: c[..][..] is int32; multiply by lhs_scale[row] * rhs_scale[col]
    // to recover float, then narrow to bf16.
    int t_row = lane / 4;
    int t_col = (lane % 4) * 2;
    #pragma unroll
    for (int wm = 0; wm < WM_ITER; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WN_ITER; wn++) {
            int base_r = block_row + warp_row * WARP_M + wm * 16;
            int base_c = block_col + warp_col * WARP_N + wn * 8;
            int gr0 = base_r + t_row;
            int gr1 = base_r + t_row + 8;
            int gc0 = base_c + t_col;
            int gc1 = base_c + t_col + 1;

            int lr0 = warp_row * WARP_M + wm * 16 + t_row;
            int lr1 = lr0 + 8;
            int lc0 = warp_col * WARP_N + wn * 8 + t_col;
            int lc1 = lc0 + 1;

            float as0 = (float)smem.lhs_scale[lr0];
            float as1 = (float)smem.lhs_scale[lr1];
            float ws0 = (float)smem.rhs_scale[lc0];
            float ws1 = (float)smem.rhs_scale[lc1];

            float v00 = (float)c[wm][wn][0] * as0 * ws0;
            float v01 = (float)c[wm][wn][1] * as0 * ws1;
            float v10 = (float)c[wm][wn][2] * as1 * ws0;
            float v11 = (float)c[wm][wn][3] * as1 * ws1;

            if (gr0 < M) {
                if (gc0 < N) out[gr0 * N + gc0] = __float2bfloat16(v00);
                if (gc1 < N) out[gr0 * N + gc1] = __float2bfloat16(v01);
            }
            if (gr1 < M) {
                if (gc0 < N) out[gr1 * N + gc0] = __float2bfloat16(v10);
                if (gc1 < N) out[gr1 * N + gc1] = __float2bfloat16(v11);
            }
        }
    }
}

#endif
