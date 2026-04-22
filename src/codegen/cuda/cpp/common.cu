#include "common.cuh"
#include <cuda_runtime.h>

extern "C" void *alloc_pinned(size_t bytes) {
    void *p = nullptr;
    cudaError_t err = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) return nullptr;
    return p;
}

extern "C" void free_pinned(void *p) {
    if (p) cudaFreeHost(p);
}

__device__ int to_tensor_idx2d(int flat_idx, int dim1, int stride0, int stride1) {
    int res = 0, i1;
    divmod(flat_idx, dim1, &flat_idx, &i1);
    res += i1 * stride1;
    res += flat_idx * stride0;
    return res;
}

__device__ int to_tensor_idx3d(int flat_idx, int dim1, int dim2, int stride0, int stride1, int stride2) {
    int res = 0, i1, i2;
    divmod(flat_idx, dim2, &flat_idx, &i2);
    divmod(flat_idx, dim1, &flat_idx, &i1);
    res += i2 * stride2;
    res += i1 * stride1;
    res += flat_idx * stride0;
    return res;
}

__device__ int to_tensor_idx4d(int flat_idx, int dim1, int dim2, int dim3, int stride0, int stride1, int stride2, int stride3) {
    int res = 0, i1, i2, i3;
    divmod(flat_idx, dim3, &flat_idx, &i3);
    divmod(flat_idx, dim2, &flat_idx, &i2);
    divmod(flat_idx, dim1, &flat_idx, &i1);
    res += i3 * stride3;
    res += i2 * stride2;
    res += i1 * stride1;
    res += flat_idx * stride0;
    return res;
}

__device__ int to_tensor_idx5d(int flat_idx, int dim1, int dim2, int dim3, int dim4, int stride0, int stride1, int stride2, int stride3, int stride4) {
    int res = 0, i1, i2, i3, i4;
    divmod(flat_idx, dim4, &flat_idx, &i4);
    divmod(flat_idx, dim3, &flat_idx, &i3);
    divmod(flat_idx, dim2, &flat_idx, &i2);
    divmod(flat_idx, dim1, &flat_idx, &i1);
    res += i4 * stride4;
    res += i3 * stride3;
    res += i2 * stride2;
    res += i1 * stride1;
    res += flat_idx * stride0;
    return res;
}
