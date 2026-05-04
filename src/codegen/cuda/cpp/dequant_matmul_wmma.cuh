#ifndef INCLUDE_DEQUANT_MATMUL_WMMA_CUH_
#define INCLUDE_DEQUANT_MATMUL_WMMA_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <int BM, int BN>
__global__ void dequant_matmul_wmma(
        __nv_bfloat16 *out,
        const __nv_bfloat16 *act,
        const int8_t *wq,
        const __nv_bfloat16 *scale,
        int M,
        int N,
        int K
) {
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int warp_id = threadIdx.x / WARP_SIZE;
    int block_row_idx = warp_id / (BN / WMMA_N);
    int block_col_idx = warp_id % (BN / WMMA_N);
    int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ __nv_bfloat16 a_shared[BM][WMMA_K + 8];
    __shared__ __nv_bfloat16 b_shared[BN][WMMA_K + 8];
    __shared__ float c_shared[BM][BN + 8];
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        for (int i = threadIdx.x; i < BM * WMMA_K; i += blockDim.x) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int gr = block_row + r;
            int gc = k + c;
            if (gr < M && gc < K) {
                a_shared[r][c] = act[gr * K + gc];
            } else {
                a_shared[r][c] = __float2bfloat16(0.0f);
            }
        }
        for (int i = threadIdx.x; i < BN * WMMA_K; i += blockDim.x) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int gr = block_col + r;
            int gc = k + c;
            if (gr < N && gc < K) {
                int8_t b = wq[gr * K + gc];
                b_shared[r][c] = __float2bfloat16((float)b * (float)scale[gr]);
            } else {
                b_shared[r][c] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        load_matrix_sync(a_frag, &a_shared[block_row_idx * WMMA_M][0], WMMA_K + 8);
        load_matrix_sync(b_frag, &b_shared[block_col_idx * WMMA_N][0], WMMA_K + 8);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    store_matrix_sync(&c_shared[block_row_idx * WMMA_M][block_col_idx * WMMA_N], c_frag, BN + 8, mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int r = i / BN;
        int c = i % BN;
        int gr = block_row + r;
        int gc = block_col + c;
        if (gr < M && gc < N) {
            out[gr * N + gc] = __float2bfloat16(c_shared[r][c]);
        }
    }
}

#endif
