#include "common.cuh"

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
