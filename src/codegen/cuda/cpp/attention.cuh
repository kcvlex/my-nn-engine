#ifndef INCLUDE_ATTENTION_CUH_
#define INCLUDE_ATTENTION_CUH_

#include <cuda.h>
#include <cooperative_groups.h>
#include <type_traits>
#include "common.cuh"

namespace cg = cooperative_groups;

// Tiled flash-attention. When `TKV == T` the K/V cache is in the same dtype as Q
// and scale pointers are unused. When `TKV == signed char` the cache is INT8 and
// `k_scale` / `v_scale` provide one BF16 scale per (kv_head, token).
template <typename T, typename TKV, int Br, int Bc, int THREADS_PER_ROW, int HEAD_DIM>
__global__ void attention(
    T *out,
    T *Q,
    TKV *K,
    TKV *V,
    T *k_scale,
    T *v_scale,
    float scale,
    bool is_causal,
    T *mask,
    int mask_outer_stride,
    int mask_row_stride,
    int q_seq_len,
    int kv_active_seq,
    int kv_cache_stride,
    int q_pos_offset,
    int num_q_heads,
    int num_kv_heads,
    int ring_sink,
    int ring_window,
    int ring_start
) {
    constexpr bool QUANT = !std::is_same<T, TKV>::value;
    constexpr int ELEMENTS_PER_THREAD = (HEAD_DIM + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    __shared__ T s_Q[Br][HEAD_DIM + 1];
    __shared__ TKV s_K[Bc][HEAD_DIM + 1];
    __shared__ TKV s_V[Bc][HEAD_DIM + 1];
    __shared__ float s_K_scale[QUANT ? Bc : 1];
    __shared__ float s_V_scale[QUANT ? Bc : 1];

    static_assert(THREADS_PER_ROW <= 32);
    static_assert((THREADS_PER_ROW & (THREADS_PER_ROW - 1)) == 0, "THREADS_PER_ROW must be a power of 2");
    assert(Br * THREADS_PER_ROW <= blockDim.x);
    assert(Bc * THREADS_PER_ROW <= blockDim.x);

    int b = blockIdx.y / num_q_heads;
    int q_head = blockIdx.y % num_q_heads;
    int group_size = num_q_heads / num_kv_heads;
    int kv_bh = b * num_kv_heads + (q_head / group_size);

    K += kv_bh * kv_cache_stride * HEAD_DIM;
    V += kv_bh * kv_cache_stride * HEAD_DIM;
    if constexpr (QUANT) {
        k_scale += kv_bh * kv_cache_stride;
        v_scale += kv_bh * kv_cache_stride;
    }
    out += blockIdx.y * q_seq_len * HEAD_DIM + blockIdx.x * Br * HEAD_DIM;
    Q += blockIdx.y * q_seq_len * HEAD_DIM + blockIdx.x * Br * HEAD_DIM;
    int num_q_row = min(Br, q_seq_len - blockIdx.x * Br);

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<THREADS_PER_ROW> tile = cg::tiled_partition<THREADS_PER_ROW>(cta);

    for (int i = threadIdx.x; i < num_q_row * HEAD_DIM; i += blockDim.x) {
        int q_row = i / HEAD_DIM;
        int q_col = i % HEAD_DIM;
        s_Q[q_row][q_col] = Q[i];
    }

    cg::sync(cta);

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float block_O[ELEMENTS_PER_THREAD] = {};
    int local_row = threadIdx.x / THREADS_PER_ROW;

    for (int i = 0; i < (kv_active_seq + Bc - 1) / Bc; i++) {
        int num_kv_row = min(Bc, kv_active_seq - i * Bc);
        for (int j = threadIdx.x; j < num_kv_row * HEAD_DIM; j += blockDim.x) {
            int kv_row = j / HEAD_DIM;
            int kv_col = j % HEAD_DIM;
            int phys = ring_phys_index(i * Bc + kv_row, ring_sink, ring_window, ring_start);
            s_K[kv_row][kv_col] = K[phys * HEAD_DIM + kv_col];
            s_V[kv_row][kv_col] = V[phys * HEAD_DIM + kv_col];
        }
        if constexpr (QUANT) {
            for (int j = threadIdx.x; j < num_kv_row; j += blockDim.x) {
                int phys = ring_phys_index(i * Bc + j, ring_sink, ring_window, ring_start);
                s_K_scale[j] = (float)k_scale[phys];
                s_V_scale[j] = (float)v_scale[phys];
            }
        }
        cg::sync(cta);

        float P[Bc];
        float old_max = row_max;
        for (int j = 0; j < num_kv_row; j++) {
            float sum = 0.0f;
            for (int k = tile.thread_rank(); k < HEAD_DIM; k += THREADS_PER_ROW) {
                sum += (float)s_Q[local_row][k] * (float)s_K[j][k];
            }
            if constexpr (QUANT) {
                sum *= s_K_scale[j];
            }
            int g_row = blockIdx.x * Br + local_row;
            int g_col = i * Bc + j;
            bool masked_out = is_causal && (q_pos_offset + g_row) < g_col;
            float val;
            if (masked_out) {
                val = -INFINITY;
            } else {
                for (int s = tile.size() / 2; 0 < s; s /= 2) {
                    float other = tile.shfl_down(sum, s);
                    sum += other;
                }
                sum = tile.shfl(sum, 0);
                val = sum * scale;
                if (mask) {
                    val += (float)mask[blockIdx.y * mask_outer_stride + g_row * mask_row_stride + g_col];
                }
            }
            P[j] = val;
            row_max = max(row_max, val);
        }

        float p_sum = 0.0f;
        for (int j = 0; j < num_kv_row; j++) {
            P[j] = expf(P[j] - row_max);
            p_sum += P[j];
        }
        float coeff = expf(old_max - row_max);
        row_sum = coeff * row_sum + p_sum;

        for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
            block_O[j] *= coeff;
        }
        for (int j = 0; j < num_kv_row; j++) {
            float pj = P[j];
            if constexpr (QUANT) {
                pj *= s_V_scale[j];
            }
            for (int k = 0, idx = tile.thread_rank(); idx < HEAD_DIM; k++, idx += THREADS_PER_ROW) {
                block_O[k] += pj * (float)s_V[j][idx];
            }
        }
        cg::sync(cta);
    }

    for (int i = 0, idx = tile.thread_rank(); idx < HEAD_DIM; i++, idx += THREADS_PER_ROW) {
        s_Q[local_row][idx] = (T)(block_O[i] / row_sum);
    }
    cg::sync(cta);

    for (int i = threadIdx.x; i < num_q_row * HEAD_DIM; i += blockDim.x) {
        int q_row = i / HEAD_DIM;
        int q_col = i % HEAD_DIM;
        out[i] = s_Q[q_row][q_col];
    }
}

template <typename T, typename TKV, int HEAD_DIM, int BLOCK_SIZE>
__global__ void attention_decode(
    T *out,
    T *Q,
    TKV *K,
    TKV *V,
    T *k_scale,
    T *v_scale,
    float scale,
    int cache_seq_len,
    int active_seq_kv,
    int num_q_heads,
    int num_kv_heads,
    int ring_sink,
    int ring_window,
    int ring_start
) {
    constexpr bool QUANT = !std::is_same<T, TKV>::value;
    __shared__ T s_Q[HEAD_DIM];
    __shared__ float s_O[HEAD_DIM];
    __shared__ float dot_buf[BLOCK_SIZE];

    int b = blockIdx.x / num_q_heads;
    int q_head = blockIdx.x % num_q_heads;
    int group_size = num_q_heads / num_kv_heads;
    int kv_bh = b * num_kv_heads + (q_head / group_size);

    out += blockIdx.x * HEAD_DIM;
    Q += blockIdx.x * HEAD_DIM;
    K += kv_bh * cache_seq_len * HEAD_DIM;
    V += kv_bh * cache_seq_len * HEAD_DIM;
    if constexpr (QUANT) {
        k_scale += kv_bh * cache_seq_len;
        v_scale += kv_bh * cache_seq_len;
    }

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
        s_Q[i] = Q[i];
        s_O[i] = 0.0f;
    }


    float row_max = -INFINITY;
    float row_sum = 0.0f;
    for (int row_K = 0; row_K < active_seq_kv; row_K++) {
        int phys_K = ring_phys_index(row_K, ring_sink, ring_window, ring_start);
        float old_max = row_max;
        float dot = 0.0f;
        for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
            dot += (float)s_Q[i] * (float)K[phys_K * HEAD_DIM + i];
        }
        if constexpr (QUANT) {
            dot *= (float)k_scale[phys_K];
        }

        for (int s = tile.size() / 2; 0 < s; s /= 2) {
            float other_dot = tile.shfl_down(dot, s);
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
        float score = expf(dot_buf[0] - row_max);
        float coeff = expf(old_max - row_max);
        row_sum = row_sum * coeff + score;
        if constexpr (QUANT) {
            score *= (float)v_scale[phys_K];
        }

        for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
            s_O[i] *= coeff;
            s_O[i] += score * (float)V[phys_K * HEAD_DIM + i];
        }
    }

    for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
        out[i] = (T)(s_O[i] / row_sum);
    }
}

#endif
