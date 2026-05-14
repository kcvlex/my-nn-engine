#ifndef INCLUDE_DEQUANT_MATMUL_WMMA_CUH_
#define INCLUDE_DEQUANT_MATMUL_WMMA_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda::wmma;

#define WARP_SIZE 32
#define BK 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <int BM>
static __device__ __forceinline__ void load_a(
        __nv_bfloat16 a_smem[][BK + 8],
        const __nv_bfloat16 *a,
        int M, 
        int N,
        int K,
        int block_row,  
        int k
) {
    constexpr int VEC = 8;
    for (int i = threadIdx.x * VEC; i < BM * BK; i += blockDim.x * VEC) {
        int r = i / BK;
        int c = i % BK;
        int gr = block_row + r;
        int gc = k + c;
        size_t valid = (gr < M && gc < K) ? min(K - gc, VEC) * sizeof(__nv_bfloat16) : 0;
        __pipeline_memcpy_async(
                &a_smem[r][c],
                &a[gr * K + gc],
                16,
                16 - valid
        );

    }
}

template <int BN>
static __device__ __forceinline__ void load_b(
    __nv_bfloat16 b_smem[][BK + 8],
    const int8_t *bq,
    const __nv_bfloat16 *scale,
    int M,
    int N,
    int K,
    int block_col,
    int k
) {
    for (int i = threadIdx.x; i < BN * BK; i += blockDim.x) {
        int r = i / BK;
        int c = i % BK;
        int gr = block_col + r;
        int gc = k + c;
        if (gr < N && gc < K) {
            int8_t b = bq[gr * K + gc];
            b_smem[r][c] = __float2bfloat16((float)b * (float)scale[gr]);
        } else {
            b_smem[r][c] = __float2bfloat16(0.0f);
        }
    }
}

template <int BM, int BN, int WARP_TILE_M, int WARP_TILE_N>
__global__ void dequant_matmul_wmma(
        __nv_bfloat16 *out,
        const __nv_bfloat16 *act,
        const int8_t *wq,
        const __nv_bfloat16 *scale,
        int M,
        int N,
        int K
) {
    assert(BM % WARP_TILE_M == 0);
    assert(BN % WARP_TILE_N == 0);
    assert(WARP_TILE_M % WMMA_M == 0);
    assert(WARP_TILE_N % WMMA_N == 0);

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int warp_id = threadIdx.x / WARP_SIZE;
    int block_row_idx = warp_id / (BN / WARP_TILE_N);
    int block_col_idx = warp_id % (BN / WARP_TILE_N);
    int lane_id = threadIdx.x % WARP_SIZE;

    constexpr int N_WARPS = (BM / WARP_TILE_M) * (BN / WARP_TILE_N);
    constexpr int WM_ITER = WARP_TILE_M / WMMA_M;
    constexpr int WN_ITER = WARP_TILE_N / WMMA_N;
    __shared__ union {
        struct {
            __nv_bfloat16 a[2][BM][BK + 8];
            __nv_bfloat16 b[2][BN][BK + 8];
        } loads;
        float store[N_WARPS][WMMA_M * WMMA_N];
    } smem;
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag[WM_ITER];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag[WN_ITER];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WM_ITER][WN_ITER];
    for (int i = 0; i < WM_ITER; i++) {
        for (int j = 0; j < WN_ITER; j++) {
            fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int buf = 0;
    load_a<BM>(smem.loads.a[buf], act, M, N, K, block_row, 0);
    load_b<BN>(smem.loads.b[buf], wq, scale, M, N, K, block_col, 0);
    __pipeline_commit();

    for (int k = 0; k < K; k += BK) {
        int next  = 1 - buf;
        if (k + BK < K) {
            load_a<BM>(smem.loads.a[next], act, M, N, K, block_row, k + BK);
            load_b<BN>(smem.loads.b[next], wq, scale, M, N, K, block_col, k + BK);
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }

        __syncthreads();

        for (int wk = 0; wk < BK; wk += WMMA_K) {
            for (int wm = 0; wm < WM_ITER; wm++) {
                load_matrix_sync(a_frag[wm], &smem.loads.a[buf][block_row_idx * WARP_TILE_M + wm * WMMA_M][wk], BK + 8);
            }
            for (int wn = 0; wn < WN_ITER; wn++) {
                load_matrix_sync(b_frag[wn], &smem.loads.b[buf][block_col_idx * WARP_TILE_N + wn * WMMA_N][wk], BK + 8);
            }

            for (int wm = 0; wm < WM_ITER; wm++) {
                for (int wn = 0; wn < WN_ITER; wn++) {
                    mma_sync(c_frag[wm][wn], a_frag[wm], b_frag[wn], c_frag[wm][wn]);
                }
            }
        }
        __syncthreads();
        buf = next;
    }

    __syncthreads();

    float* my_scratch = smem.store[warp_id];

    for (int wm = 0; wm < WM_ITER; wm++) {
        for (int wn = 0; wn < WN_ITER; wn++) {
            store_matrix_sync(my_scratch, c_frag[wm][wn], WMMA_N, mem_row_major);
            int row_base = block_row + block_row_idx * WARP_TILE_M + wm * WMMA_M;
            int col_base = block_col + block_col_idx * WARP_TILE_N + wn * WMMA_N;
            for (int i = lane_id; i < WMMA_M * WMMA_N; i += WARP_SIZE) {
                int r = i / WMMA_N;
                int c = i % WMMA_N;
                int gr = row_base + r;
                int gc = col_base + c;
                if (gr < M && gc < N) {
                    out[gr * N + gc] = __float2bfloat16(my_scratch[r * WMMA_N + c]);
                }
            }
        }
    }
}

#endif
