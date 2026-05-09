#ifndef INCLUDE_ROPE_CUH_
#define INCLUDE_ROPE_CUH_

#include <cuda.h>

// Rotary Position Embedding with per-token gather.
//
// Layout: x is treated as [outer, S, D] (D = HEAD_DIM, S = seq_len).
// Each row's rotation row index is read from `position[s]` (i64 host -> i32
// would lose precision; we use long long to match graph i64 inputs).
//
//   low  = x[..., :D/2]   high = x[..., D/2:]
//   out_low  = x_low  * cos[pos, j]       - x_high * sin[pos, j]
//   out_high = x_high * cos[pos, j+D/2]   + x_low  * sin[pos, j+D/2]
//
// Grid: outer * S blocks of HEAD_DIM/2 threads. Each thread handles one
// (low, high) pair to write both halves.
template <typename T, int HEAD_DIM>
__global__ void rope(
    T *out,
    const T *x,
    const T *cos_table,
    const T *sin_table,
    const long long *position,
    int seq_len
) {
    constexpr int HALF = HEAD_DIM / 2;
    int row = blockIdx.x;            // outer * S + s
    int s = row % seq_len;
    int j = threadIdx.x;
    if (HALF <= j) return;

    long long pos = position[s];
    int row_off = row * HEAD_DIM;
    int tab_off = (int)pos * HEAD_DIM;

    float x_lo = (float)x[row_off + j];
    float x_hi = (float)x[row_off + j + HALF];
    float cos_lo = (float)cos_table[tab_off + j];
    float cos_hi = (float)cos_table[tab_off + j + HALF];
    float sin_lo = (float)sin_table[tab_off + j];
    float sin_hi = (float)sin_table[tab_off + j + HALF];

    out[row_off + j]        = (T)(x_lo * cos_lo - x_hi * sin_lo);
    out[row_off + j + HALF] = (T)(x_hi * cos_hi + x_lo * sin_hi);
}

#endif
