#ifndef INCLUDE_ATTENTION_CUH_
#define INCLUDE_ATTENTION_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <type_traits>
#include <cstdint>
#include "common.cuh"

namespace cg = cooperative_groups;

static __device__ __forceinline__ void attn_ldmatrix_x4(
    uint32_t out[4], uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(smem_addr)
    );
}

static __device__ __forceinline__ void attn_ldmatrix_x2(
    uint32_t out[2], uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(out[0]), "=r"(out[1])
        : "r"(smem_addr)
    );
}

static __device__ __forceinline__ void attn_mma_m16n8k16(
    float c[4], const uint32_t a[4], const uint32_t b[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}

// bf16 Tensor Core FlashAttention-2 body (single warp per block, Br==Bc==32).
// S = Q*K^T and O += P*V both run on mma.m16n8k16. S is written to fp32 smem,
// the online softmax row reduction runs over those contiguous rows, and P is
// written back as bf16 for ldmatrix into the P*V mma (the smem round-trip
// handles the C-fragment -> A-fragment layout change in hardware).
template <typename T, typename TKV, int Br, int Bc, int THREADS_PER_ROW, int HEAD_DIM>
__global__ void attention_tc(
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
    int ring_start,
    const T *cos_table,
    const T *sin_table,
    const long long *kv_position
) {
    constexpr bool QUANT = !std::is_same<T, TKV>::value;
    constexpr int HALF = HEAD_DIM / 2;
    constexpr int NUM_WARPS = Br / 16;  // split-Q: one warp owns 16 query rows
    constexpr int N_ITER = Bc / 8;
    constexpr int K_ITER = HEAD_DIM / 16;
    constexpr int D_ITER = HEAD_DIM / 8;
    static_assert(Br % 16 == 0 && Bc % 16 == 0 && HEAD_DIM % 16 == 0);

    bool rope_on = (kv_position != nullptr);
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int warp_row0 = warp * 16;  // first query row this warp owns

    __shared__ __nv_bfloat16 s_Q[Br][HEAD_DIM];
    __shared__ __nv_bfloat16 s_K[Bc][HEAD_DIM];
    __shared__ __nv_bfloat16 s_V[HEAD_DIM][Bc];
    __shared__ __nv_bfloat16 s_P[Br][Bc];
    __shared__ float s_S[Br][Bc];
    __shared__ float m_smem[Br];
    __shared__ float l_smem[Br];
    __shared__ float corr_smem[Br];

    cg::thread_block cta = cg::this_thread_block();
    int batch = blockIdx.y / num_q_heads;
    int q_head = blockIdx.y % num_q_heads;
    int group_size = num_q_heads / num_kv_heads;
    int kv_bh = batch * num_kv_heads + (q_head / group_size);
    K += kv_bh * kv_cache_stride * HEAD_DIM;
    V += kv_bh * kv_cache_stride * HEAD_DIM;
    if constexpr (QUANT) {
        k_scale += kv_bh * kv_cache_stride;
        v_scale += kv_bh * kv_cache_stride;
    }
    out += blockIdx.y * q_seq_len * HEAD_DIM + blockIdx.x * Br * HEAD_DIM;
    Q += blockIdx.y * q_seq_len * HEAD_DIM + blockIdx.x * Br * HEAD_DIM;
    int num_q_row = min(Br, q_seq_len - blockIdx.x * Br);

    for (int i = tid; i < Br * HEAD_DIM; i += blockDim.x) {
        int r = i / HEAD_DIM;
        int c = i % HEAD_DIM;
        s_Q[r][c] = (r < num_q_row) ? Q[r * HEAD_DIM + c] : (T)0;
    }
    if (tid < Br) {
        m_smem[tid] = -INFINITY;
        l_smem[tid] = 0.0f;
    }
    cg::sync(cta);

    // load this warp's 16 Q rows into fragments once (reused across KV blocks)
    uint32_t q_frag[K_ITER][4];
    #pragma unroll
    for (int ki = 0; ki < K_ITER; ki++) {
        int row = warp_row0 + (lane & 15);
        int col = ki * 16 + (lane >> 4) * 8;
        uint32_t addr = __cvta_generic_to_shared(&s_Q[row][col]);
        attn_ldmatrix_x4(q_frag[ki], addr);
    }

    float o_acc[D_ITER][4] = {};

    int n_kv_blocks = (kv_active_seq + Bc - 1) / Bc;
    // Causal: KV blocks are ascending, so once a block's first column exceeds
    // the largest query position in this Q-block, all remaining blocks are
    // fully masked -> stop. The diagonal block is still processed (partial mask
    // handled post-mma below).
    if (is_causal) {
        int q_max_pos = q_pos_offset + (int)blockIdx.x * Br + (num_q_row - 1);
        n_kv_blocks = min(n_kv_blocks, q_max_pos / Bc + 1);
    }
    for (int blk = 0; blk < n_kv_blocks; blk++) {
        int num_kv_row = min(Bc, kv_active_seq - blk * Bc);

        // Stage K (dequant + rope) and V (dequant) into bf16 smem (cooperative)
        for (int i = tid; i < Bc * HEAD_DIM; i += blockDim.x) {
            int r = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            if (num_kv_row <= r) {
                s_K[r][c] = (__nv_bfloat16)0;
                s_V[c][r] = (__nv_bfloat16)0;
                continue;
            }
            int phys = ring_phys_index(blk * Bc + r, ring_sink, ring_window, ring_start);
            float ks = QUANT ? (float)k_scale[phys] : 1.0f;
            float vs = QUANT ? (float)v_scale[phys] : 1.0f;
            float kv = (float)K[phys * HEAD_DIM + c] * ks;
            if (rope_on) {
                int pos = (int)kv_position[blk * Bc + r];
                int c_pair = (c < HALF) ? c + HALF : c - HALF;
                float sign = (c < HALF) ? -1.0f : 1.0f;
                float kp = (float)K[phys * HEAD_DIM + c_pair] * ks;
                float cc = (float)cos_table[pos * HEAD_DIM + c];
                float sn = (float)sin_table[pos * HEAD_DIM + c];
                kv = kv * cc + sign * kp * sn;
            }
            s_K[r][c] = (__nv_bfloat16)kv;
            s_V[c][r] = (__nv_bfloat16)((float)V[phys * HEAD_DIM + c] * vs);
        }
        cg::sync(cta);

        // S = Q * K^T for this warp's 16 rows
        float s_acc[N_ITER][4] = {};
        #pragma unroll
        for (int ki = 0; ki < K_ITER; ki++) {
            uint32_t k_frag[N_ITER][2];
            #pragma unroll
            for (int ni = 0; ni < N_ITER; ni++) {
                int n = ni * 8 + (lane & 7);
                int k = ki * 16 + (lane >> 3) * 8;
                uint32_t addr = __cvta_generic_to_shared(&s_K[n][k]);
                attn_ldmatrix_x2(k_frag[ni], addr);
            }
            #pragma unroll
            for (int ni = 0; ni < N_ITER; ni++) {
                attn_mma_m16n8k16(s_acc[ni], q_frag[ki], k_frag[ni]);
            }
        }

        // Write scores (scale + causal + additive mask) to fp32 smem
        int q_blk_base = blockIdx.x * Br;
        #pragma unroll
        for (int ni = 0; ni < N_ITER; ni++) {
            int r0 = warp_row0 + lane / 4;
            int r1 = r0 + 8;
            int c0 = ni * 8 + (lane % 4) * 2;
            int c1 = c0 + 1;
            int g_col0 = blk * Bc + c0;
            int g_col1 = blk * Bc + c1;
            int g_row0 = q_blk_base + r0;
            int g_row1 = q_blk_base + r1;
            float v00 = s_acc[ni][0] * scale;
            float v01 = s_acc[ni][1] * scale;
            float v10 = s_acc[ni][2] * scale;
            float v11 = s_acc[ni][3] * scale;
            if (mask) {
                long base = blockIdx.y * mask_outer_stride;
                bool r0_ok = g_row0 < q_seq_len;
                bool r1_ok = g_row1 < q_seq_len;
                if (r0_ok && c0 < num_kv_row) v00 += (float)mask[base + g_row0 * mask_row_stride + g_col0];
                if (r0_ok && c1 < num_kv_row) v01 += (float)mask[base + g_row0 * mask_row_stride + g_col1];
                if (r1_ok && c0 < num_kv_row) v10 += (float)mask[base + g_row1 * mask_row_stride + g_col0];
                if (r1_ok && c1 < num_kv_row) v11 += (float)mask[base + g_row1 * mask_row_stride + g_col1];
            }
            if (is_causal) {
                if ((q_pos_offset + g_row0) < g_col0) v00 = -INFINITY;
                if ((q_pos_offset + g_row0) < g_col1) v01 = -INFINITY;
                if ((q_pos_offset + g_row1) < g_col0) v10 = -INFINITY;
                if ((q_pos_offset + g_row1) < g_col1) v11 = -INFINITY;
            }
            if (num_kv_row <= c0) v00 = -INFINITY;
            if (num_kv_row <= c1) v01 = -INFINITY;
            if (num_kv_row <= c0) v10 = -INFINITY;
            if (num_kv_row <= c1) v11 = -INFINITY;
            s_S[r0][c0] = v00;
            s_S[r0][c1] = v01;
            s_S[r1][c0] = v10;
            s_S[r1][c1] = v11;
        }
        __syncwarp();

        // Online softmax row reduction; lane l (<16) owns this warp's row l
        if (lane < 16) {
            int r = warp_row0 + lane;
            float old_m = m_smem[r];
            float new_m = old_m;
            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                new_m = max(new_m, s_S[r][c]);
            }
            float corr = (new_m == -INFINITY) ? 1.0f : expf(old_m - new_m);
            float psum = 0.0f;
            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                float p = (s_S[r][c] == -INFINITY) ? 0.0f : expf(s_S[r][c] - new_m);
                s_P[r][c] = (__nv_bfloat16)p;
                psum += p;
            }
            m_smem[r] = new_m;
            l_smem[r] = l_smem[r] * corr + psum;
            corr_smem[r] = corr;
        }
        __syncwarp();

        // Rescale running output by corr for the rows this lane owns
        float corr0 = corr_smem[warp_row0 + lane / 4];
        float corr1 = corr_smem[warp_row0 + lane / 4 + 8];
        #pragma unroll
        for (int di = 0; di < D_ITER; di++) {
            o_acc[di][0] *= corr0;
            o_acc[di][1] *= corr0;
            o_acc[di][2] *= corr1;
            o_acc[di][3] *= corr1;
        }

        // O += P * V
        #pragma unroll
        for (int ki = 0; ki < (Bc / 16); ki++) {
            uint32_t p_frag[4];
            {
                int row = warp_row0 + (lane & 15);
                int col = ki * 16 + (lane >> 4) * 8;
                uint32_t addr = __cvta_generic_to_shared(&s_P[row][col]);
                attn_ldmatrix_x4(p_frag, addr);
            }
            uint32_t v_frag[D_ITER][2];
            #pragma unroll
            for (int di = 0; di < D_ITER; di++) {
                int n = di * 8 + (lane & 7);
                int k = ki * 16 + (lane >> 3) * 8;
                uint32_t addr = __cvta_generic_to_shared(&s_V[n][k]);
                attn_ldmatrix_x2(v_frag[di], addr);
            }
            #pragma unroll
            for (int di = 0; di < D_ITER; di++) {
                attn_mma_m16n8k16(o_acc[di], p_frag, v_frag[di]);
            }
        }
        cg::sync(cta);
    }

    // Write O = o_acc / row_sum for this warp's 16 rows
    int r0 = warp_row0 + lane / 4;
    int r1 = r0 + 8;
    float inv0 = (0.0f < l_smem[r0]) ? 1.0f / l_smem[r0] : 0.0f;
    float inv1 = (0.0f < l_smem[r1]) ? 1.0f / l_smem[r1] : 0.0f;
    #pragma unroll
    for (int di = 0; di < D_ITER; di++) {
        int c0 = di * 8 + (lane % 4) * 2;
        int c1 = c0 + 1;
        if (r0 < num_q_row) {
            out[r0 * HEAD_DIM + c0] = (T)(o_acc[di][0] * inv0);
            out[r0 * HEAD_DIM + c1] = (T)(o_acc[di][1] * inv0);
        }
        if (r1 < num_q_row) {
            out[r1 * HEAD_DIM + c0] = (T)(o_acc[di][2] * inv1);
            out[r1 * HEAD_DIM + c1] = (T)(o_acc[di][3] * inv1);
        }
    }
}

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
    int ring_start,
    const T *cos_table,
    const T *sin_table,
    const long long *kv_position
) {
    constexpr bool QUANT = !std::is_same<T, TKV>::value;
    constexpr int HALF = HEAD_DIM / 2;
    constexpr int ELEMENTS_PER_THREAD = (HEAD_DIM + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    bool rope_on = (kv_position != nullptr);
    __shared__ T s_Q[Br][HEAD_DIM + 1];
    __shared__ TKV s_K[Bc][HEAD_DIM + 1];
    __shared__ TKV s_V[Bc][HEAD_DIM + 1];
    __shared__ float s_K_scale[QUANT ? Bc : 1];
    __shared__ float s_V_scale[QUANT ? Bc : 1];

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

    static_assert(THREADS_PER_ROW <= 32);
    static_assert((THREADS_PER_ROW & (THREADS_PER_ROW - 1)) == 0, "THREADS_PER_ROW must be a power of 2");
    assert(Br * THREADS_PER_ROW <= blockDim.x);
    assert(Bc * THREADS_PER_ROW <= blockDim.x);

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
            int rank = i * Bc + j;
            float sum = 0.0f;
            if (rope_on) {
                int pos = (int)kv_position[rank];
                float k_scale_val = QUANT ? s_K_scale[j] : 1.0f;
                for (int k = tile.thread_rank(); k < HEAD_DIM; k += THREADS_PER_ROW) {
                    int k_pair = (k < HALF) ? k + HALF : k - HALF;
                    float sign = (k < HALF) ? -1.0f : 1.0f;
                    float kk = (float)s_K[j][k] * k_scale_val;
                    float kp = (float)s_K[j][k_pair] * k_scale_val;
                    float c = (float)cos_table[pos * HEAD_DIM + k];
                    float ss = (float)sin_table[pos * HEAD_DIM + k];
                    sum += (float)s_Q[local_row][k] * (kk * c + sign * kp * ss);
                }
            } else {
                for (int k = tile.thread_rank(); k < HEAD_DIM; k += THREADS_PER_ROW) {
                    sum += (float)s_Q[local_row][k] * (float)s_K[j][k];
                }
                if constexpr (QUANT) {
                    sum *= s_K_scale[j];
                }
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

// `cos_table`, `sin_table`, `kv_position` are optional (pass nullptr to skip
// the on-the-fly RoPE). When non-null, K is dequant + half-split-rotated using
// the row index `kv_position[row_K]` of the cos/sin tables before the dot
// product. V is never rotated.
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
    int ring_start,
    const T *cos_table,
    const T *sin_table,
    const long long *kv_position
) {
    constexpr bool QUANT = !std::is_same<T, TKV>::value;
    constexpr int HALF = HEAD_DIM / 2;
    bool rope_on = (kv_position != nullptr);
    __shared__ T s_Q[HEAD_DIM];
    __shared__ TKV s_K[HEAD_DIM];
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
        float k_scale_val = QUANT ? (float)k_scale[phys_K] : 1.0f;

        for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
            s_K[i] = K[phys_K * HEAD_DIM + i];
        }
        cg::sync(cta);

        float old_max = row_max;
        float dot = 0.0f;
        if (rope_on) {
            int pos = (int)kv_position[row_K];
            for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
                int i_pair = (i < HALF) ? i + HALF : i - HALF;
                float sign = (i < HALF) ? -1.0f : 1.0f;
                float k = (float)s_K[i] * k_scale_val;
                float kp = (float)s_K[i_pair] * k_scale_val;
                float c = (float)cos_table[pos * HEAD_DIM + i];
                float s = (float)sin_table[pos * HEAD_DIM + i];
                dot += (float)s_Q[i] * (k * c + sign * kp * s);
            }
        } else {
            for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
                dot += (float)s_Q[i] * (float)s_K[i] * k_scale_val;
            }
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
