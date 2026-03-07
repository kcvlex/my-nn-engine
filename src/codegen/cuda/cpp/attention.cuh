#ifndef INCLUDE_ATTENTION_CUH_
#define INCLUDE_ATTENTION_CUH_

#include <cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename T, int Br, int Bc, int THREADS_PER_ROW, int HEAD_DIM>
__global__ void attention(
    T *out,
    T *Q,
    T *K,
    T *V,
    T scale,
    bool is_causal,
    T penalty,
    int N
) {
    constexpr int ELEMENTS_PER_THREAD = (HEAD_DIM + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    __shared__ T s_Q[Br][HEAD_DIM + 1];
    __shared__ T s_K[Bc][HEAD_DIM + 1];
    __shared__ T s_V[Bc][HEAD_DIM + 1];

    static_assert(THREADS_PER_ROW <= 32);
    static_assert((THREADS_PER_ROW & (THREADS_PER_ROW - 1)) == 0, "THREADS_PER_ROW must be a power of 2");
    assert(Br * THREADS_PER_ROW <= blockDim.x);
    assert(Bc * THREADS_PER_ROW <= blockDim.x);

    K += blockIdx.y * N * HEAD_DIM;
    V += blockIdx.y * N * HEAD_DIM;
    out += blockIdx.y * N * HEAD_DIM + blockIdx.x * Br * HEAD_DIM;
    Q += blockIdx.y * N * HEAD_DIM + blockIdx.x * Br * HEAD_DIM;
    int num_q_row = min(Br, N - blockIdx.x * Br);

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<THREADS_PER_ROW> tile = cg::tiled_partition<THREADS_PER_ROW>(cta);

    for (int i = threadIdx.x; i < num_q_row * HEAD_DIM; i += blockDim.x) {
        int q_row = i / HEAD_DIM;
        int q_col = i % HEAD_DIM;
        s_Q[q_row][q_col] = Q[i];
    }

    cg::sync(cta);

    T row_max = -INFINITY;
    T row_sum = 0;
    T block_O[ELEMENTS_PER_THREAD] = {};
    int local_row = threadIdx.x / THREADS_PER_ROW;

    for (int i = 0; i < (N + Bc - 1) / Bc; i++) {
        int num_kv_row = min(Bc, N - i * Bc);
        for (int j = threadIdx.x; j < num_kv_row * HEAD_DIM; j += blockDim.x) {
            int kv_row = j / HEAD_DIM;
            int kv_col = j % HEAD_DIM;
            s_K[kv_row][kv_col] = K[i * Bc * HEAD_DIM + j];
            s_V[kv_row][kv_col] = V[i * Bc * HEAD_DIM + j];
        }
        cg::sync(cta);

        T P[Bc];
        T old_max = row_max;
        for (int j = 0; j < num_kv_row; j++) {
            T sum = 0;
            for (int k = tile.thread_rank(); k < HEAD_DIM; k += THREADS_PER_ROW) {
                sum += s_Q[local_row][k] * s_K[j][k];
            }
            int g_row = blockIdx.x * Br + local_row;
            int g_col = i * Bc + j;
            bool is_masked = is_causal && g_row < g_col;
            T val = -penalty;
            if (!is_masked) {
                for (int s = tile.size() / 2; 0 < s; s /= 2) {
                    T other = tile.shfl_down(sum, s);
                    sum += other;
                }
                sum = tile.shfl(sum, 0);
                val = sum * scale;
            }
            P[j] = val;
            row_max = max(row_max, val);
        }

        T p_sum = 0;
        for (int j = 0; j < num_kv_row; j++) {
            P[j] = exp(P[j] - row_max);
            p_sum += P[j];
        }
        T coeff = exp(old_max - row_max);
        row_sum = coeff * row_sum + p_sum;

        for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
            block_O[j] *= coeff;
        }
        for (int j = 0; j < num_kv_row; j++) {
            for (int k = 0, idx = tile.thread_rank(); idx < HEAD_DIM; k++, idx += THREADS_PER_ROW) {
                block_O[k] += P[j] * s_V[j][idx];
            }
        }
        cg::sync(cta);
    }

    for (int i = 0, idx = tile.thread_rank(); idx < HEAD_DIM; i++, idx += THREADS_PER_ROW) {
        s_Q[local_row][idx] = block_O[i] / row_sum;
    }
    cg::sync(cta);

    for (int i = threadIdx.x; i < num_q_row * HEAD_DIM; i += blockDim.x) {
        int q_row = i / HEAD_DIM;
        int q_col = i % HEAD_DIM;
        out[i] = s_Q[q_row][q_col];
    }
}

#endif
