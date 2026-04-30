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
    T *mask,
    int mask_outer_stride,
    int mask_row_stride,
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
            bool masked_out = is_causal && g_row < g_col;
            T val;
            if (masked_out) {
                val = -INFINITY;
            } else {
                for (int s = tile.size() / 2; 0 < s; s /= 2) {
                    T other = tile.shfl_down(sum, s);
                    sum += other;
                }
                sum = tile.shfl(sum, 0);
                val = sum * scale;
                if (mask) {
                    val += mask[blockIdx.y * mask_outer_stride + g_row * mask_row_stride + g_col];
                }
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

// cache_seq_len: K/V buffer's per-(batch,head) seq stride (static dim of K).
// active_seq_kv: number of K/V rows to actually attend over (runtime, 0 <= active <= cache_seq_len).
// For static-shape attention these are equal; for KV-cache decode active_seq_kv = past_len + 1.
//
// Grouped-Query Attention (GQA): Q has num_q_heads heads, K/V have num_kv_heads heads.
// Each group of (num_q_heads / num_kv_heads) Q-heads shares the same K/V head.
// MHA is the special case num_kv_heads == num_q_heads.
//
// Grid layout: blockIdx.x = batch_idx * num_q_heads + q_head_idx, one block per (batch, q_head).
template <typename T, int HEAD_DIM, int BLOCK_SIZE>
__global__ void attention_decode(
    T *out,
    T *Q,
    T *K,
    T *V,
    T scale,
    int cache_seq_len,
    int active_seq_kv,
    int num_q_heads,
    int num_kv_heads
) {
    __shared__ T s_Q[HEAD_DIM];
    __shared__ T s_O[HEAD_DIM];
    __shared__ T dot_buf[BLOCK_SIZE];

    int b = blockIdx.x / num_q_heads;
    int q_head = blockIdx.x % num_q_heads;
    int group_size = num_q_heads / num_kv_heads;
    int kv_bh = b * num_kv_heads + (q_head / group_size);

    out += blockIdx.x * HEAD_DIM;
    Q += blockIdx.x * HEAD_DIM;
    K += kv_bh * cache_seq_len * HEAD_DIM;
    V += kv_bh * cache_seq_len * HEAD_DIM;

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
        s_Q[i] = Q[i];
        s_O[i] = 0;
    }


    T row_max = -INFINITY;
    T row_sum = 0;
    for (int row_K = 0; row_K < active_seq_kv; row_K++) {
        T old_max = row_max;
        T dot = 0;
        for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
            dot += s_Q[i] * K[row_K * HEAD_DIM + i];
        }

        for (int s = tile.size() / 2; 0 < s; s /= 2) {
            T other_dot = tile.shfl_down(dot, s);
            dot += other_dot;
        }
        dot_buf[threadIdx.x] = dot * scale;
        cg::sync(cta);
        for (int s = BLOCK_SIZE / 2; tile.size() <= s; s /= 2) {
            if (threadIdx.x < s) {
                dot_buf[threadIdx.x] += dot_buf[threadIdx.x + s];
            }
            cg::sync(cta);
        }

        row_max = max(row_max, dot_buf[0]);
        T score = exp(dot_buf[0] - row_max);
        T coeff = exp(old_max - row_max);
        row_sum = row_sum * coeff  + score;

        for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
            s_O[i] *= coeff;
            s_O[i] += score * V[row_K * HEAD_DIM + i];
        }
    }

    for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
        out[i] = s_O[i] / row_sum;
    }
}

#endif
