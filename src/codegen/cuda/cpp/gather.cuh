#ifndef INCLUDE_GATHER_CUH_
#define INCLUDE_GATHER_CUH_
#include <cuda.h>

template <typename T, typename IDX>
__global__ void gather_axis0_kernel(
    T *out,
    const T *in,
    const IDX *indices,
    int axis_dim,
    int repeat,
    int size
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (size <= gid) return;
    int index = static_cast<int>(indices[gid]);
    if (index < 0) index += axis_dim;
    for (int i = 0; i < repeat; i++) {
        out[gid * repeat + i] = in[index * repeat + i];
    }
}

#endif
