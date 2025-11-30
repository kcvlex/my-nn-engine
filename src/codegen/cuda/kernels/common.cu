#include "common.cuh"

__device__ i64 to_tensor_idx2d(i64 flat_idx, i64 dim1, i64 stride0, i64 stride1) {
    i64 res = 0, i1;
    divmod(flat_idx, dim1, &flat_idx, &i1);
    res += i1 * stride1;
    res += flat_idx * stride0;
    return res;
}

__device__ i64 to_tensor_idx3d(i64 flat_idx, i64 dim1, i64 dim2, i64 stride0, i64 stride1, i64 stride2) {
    i64 res = 0, i1, i2;
    divmod(flat_idx, dim2, &flat_idx, &i2);
    divmod(flat_idx, dim1, &flat_idx, &i1);
    res += i2 * stride2;
    res += i1 * stride1;
    res += flat_idx * stride0;
    return res;
}

__device__ i64 to_tensor_idx4d(i64 flat_idx, i64 dim1, i64 dim2, i64 dim3, i64 stride0, i64 stride1, i64 stride2, i64 stride3) {
    i64 res = 0, i1, i2, i3;
    divmod(flat_idx, dim3, &flat_idx, &i3);
    divmod(flat_idx, dim2, &flat_idx, &i2);
    divmod(flat_idx, dim1, &flat_idx, &i1);
    res += i3 * stride3;
    res += i2 * stride2;
    res += i1 * stride1;
    res += flat_idx * stride0;
    return res;
}
