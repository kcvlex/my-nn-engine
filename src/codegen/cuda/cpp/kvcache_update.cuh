#ifndef INCLUDE_KVCACHE_UPDATE_CUH_
#define INCLUDE_KVCACHE_UPDATE_CUH_
#include <cuda.h>

template <typename T, int HEAD_DIM>
__global__ void kvcache_update(
    T *cache,
    T *new_kv,
    int cache_seq_len,
    int new_seq_len,
    int offset
) {
    int bh = blockIdx.x;
    int total = new_seq_len * HEAD_DIM;
    cache += bh * cache_seq_len * HEAD_DIM + offset * HEAD_DIM;
    new_kv += bh * new_seq_len * HEAD_DIM;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        cache[i] = new_kv[i];
    }
}

#endif
