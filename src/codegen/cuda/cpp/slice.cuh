#ifndef INCLUDE_SLICE_CUH_
#define INCLUDE_SLICE_CUH_
#include <cuda.h>

template <int NDIM>
struct SliceConfig {
    int output_dims[NDIM];
    int input_strides[NDIM];
    int base_offset;
};

template <typename T, int NDIM>
__global__ void slice_kernel(
    T *out,
    const T *in,
    SliceConfig<NDIM> cfg,
    int size
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (size <= gid) return;
    int src_idx = cfg.base_offset;
    int rem = gid;
#pragma unroll
    for (int i = NDIM - 1; i >= 0; i--) {
        int idx = rem % cfg.output_dims[i];
        rem /= cfg.output_dims[i];
        src_idx += idx * cfg.input_strides[i];
    }
    out[gid] = in[src_idx];
}

#endif
