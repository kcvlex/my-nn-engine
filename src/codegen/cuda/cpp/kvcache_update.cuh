#ifndef INCLUDE_KVCACHE_UPDATE_CUH_
#define INCLUDE_KVCACHE_UPDATE_CUH_
#include <cuda.h>
#include "common.cuh"

// `offset` is the logical recency rank of the first new token (i.e. past_len).
// With `ring_window == 0` the kernel behaves as a contiguous write to
// `cache[..., offset:offset+new_seq, :]`. With `ring_window > 0` each new
// token's destination slot is remapped through the streaming sink+ring
// layout (see `ring_phys_index`).
template <typename T, int HEAD_DIM>
__global__ void kvcache_update(
    T *cache,
    T *new_kv,
    int cache_seq_len,
    int new_seq_len,
    int offset,
    int ring_sink,
    int ring_window,
    int ring_start
) {
    int bh = blockIdx.x;
    cache += bh * cache_seq_len * HEAD_DIM;
    new_kv += bh * new_seq_len * HEAD_DIM;
    for (int t = 0; t < new_seq_len; t++) {
        int phys = ring_phys_index(offset + t, ring_sink, ring_window, ring_start);
        for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
            cache[phys * HEAD_DIM + i] = new_kv[t * HEAD_DIM + i];
        }
    }
}

#endif
