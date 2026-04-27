#ifndef INCLUDE_COPY_CUH_
#define INCLUDE_COPY_CUH_
#include <cuda.h>

template <typename T>
__global__ void copy_kernel(T *out, const T *in, int size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (size <= gid) return;
    out[gid] = in[gid];
}

#endif
