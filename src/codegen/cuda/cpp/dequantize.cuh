#ifndef INCLUDE_DEQUANTIZE_CUH_
#define INCLUDE_DEQUANTIZE_CUH_
#include <cuda.h>

template <typename Tin, typename Tout>
__global__ void dequantize_linear(
    Tout *out,
    const Tin *x,
    const Tout *scale,
    int axis_dim,
    int inner_size,
    int total
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (total <= gid) return;
    int axis_idx = (gid / inner_size) % axis_dim;
    float s = (float)scale[axis_idx];
    out[gid] = (Tout)((float)x[gid] * s);
}

#endif
