#ifndef INCLUDE_QUANTIZING_KVCACHE_UPDATE_CUH_
#define INCLUDE_QUANTIZING_KVCACHE_UPDATE_CUH_
#include <cuda.h>

// Block: BLOCK_SIZE threads (1D), one block per (batch * head * new_seq_token).
// For each (b, h, t):
//   s = max_{d} |new[b,h,t,d]| / 127
//   cache[b,h,offset+t,d] = round(new[b,h,t,d] / s).clamp(-128, 127)  for all d
//   scale[b,h,offset+t]   = s
//
// Layout: cache and new are [B, H, ..., HEAD_DIM] row-major
//   cache: row-major [B, H, MAX_SEQ, HEAD_DIM]
//   scale: row-major [B, H, MAX_SEQ]
//   new:   row-major [B, H, NEW_SEQ, HEAD_DIM]
template <typename TNew, typename TScale, int HEAD_DIM, int BLOCK_SIZE>
__global__ void quantizing_kvcache_update(
    signed char *cache,
    TScale *scale,
    const TNew *new_kv,
    int max_seq_len,
    int new_seq_len,
    int offset
) {
    int gid = blockIdx.x;
    int t = gid % new_seq_len;
    int bh = gid / new_seq_len;
    int tid = threadIdx.x;

    const TNew *src = new_kv + bh * new_seq_len * HEAD_DIM + t * HEAD_DIM;
    signed char *dst = cache + bh * max_seq_len * HEAD_DIM + (offset + t) * HEAD_DIM;
    TScale *scale_dst = scale + bh * max_seq_len + (offset + t);

    __shared__ float reduction[BLOCK_SIZE];

    float local_max = 0.0f;
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        float v = (float) src[d];
        float a = fabsf(v);
        if (a > local_max) local_max = a;
    }
    reduction[tid] = local_max;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = reduction[tid + s];
            if (other > reduction[tid]) reduction[tid] = other;
        }
        __syncthreads();
    }
    float max_abs = reduction[0];
    float s = max_abs / 127.0f;
    float inv_s = (max_abs == 0.0f) ? 0.0f : (1.0f / s);

    if (tid == 0) {
        *scale_dst = (TScale) s;
    }

    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        float v = (float) src[d];
        float q = roundf(v * inv_s);
        if (q < -128.0f) q = -128.0f;
        if (q > 127.0f) q = 127.0f;
        dst[d] = (signed char) q;
    }
}

#endif
