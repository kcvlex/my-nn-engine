#ifndef INCLUDE_CONCAT_CUH_
#define INCLUDE_CONCAT_CUH_
#include <cuda.h>

template <typename T, int NDIM, int N_INPUTS>
struct ConcatInputs {
    const T *ins[N_INPUTS];
    int in_sizes_acc[N_INPUTS];
    int in_axis_sizes_acc[N_INPUTS];
    int input_dims[N_INPUTS][NDIM];
    int input_strides[N_INPUTS][NDIM];
    int output_dims[NDIM];
};

template <typename T, int NDIM, int N_INPUTS>
__global__ void concat_kernel(
    T *out,
    ConcatInputs<T, NDIM, N_INPUTS> cfg,
    int axis,
    int total_size
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (total_size <= gid) return;
    int select = 0;
#pragma unroll
    for (int i = 1; i < N_INPUTS; i++) {
        if (cfg.in_sizes_acc[i] <= gid) select = i;
    }
    int in_offset = gid - cfg.in_sizes_acc[select];
    T value = cfg.ins[select][in_offset];
    int indexes[NDIM];
#pragma unroll
    for (int i = 0; i < NDIM; i++) {
        int stride = cfg.input_strides[select][i];
        int dim = cfg.input_dims[select][i];
        indexes[i] = (dim <= 1 || stride == 0) ? 0 : (in_offset / stride % dim);
    }
    indexes[axis] += cfg.in_axis_sizes_acc[select];
    int out_offset = 0;
#pragma unroll
    for (int i = 0; i < NDIM; i++) {
        out_offset *= cfg.output_dims[i];
        out_offset += indexes[i];
    }
    out[out_offset] = value;
}

#endif
