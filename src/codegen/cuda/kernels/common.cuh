#ifndef INCLUDE_COMMON_CUH_
#define INCLUDE_COMMON_CUH_

#include <cstdio>
#include <cstdint>

#define cudaCheckErr(STATUS) \
    do { \
        if (STATUS != cudaSuccess) { \
            fprintf(stderr, \
                    "CUDA eror: %s %s %d\n", \
                    cudaGetErrorString(STATUS), \
                    __FILE__, \
                    __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define cudnnCheckErr(expression) \
  do { \
    cudnnStatus_t status = (expression); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, \
                "cuDNN eror: %s %s %d \n", \
                cudnnGetErrorString(status), \
                __FILE__, \
                __LINE__); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

#define cublasCheckErr(expression) \
  do { \
    cublasStatus_t status = (expression); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, \
                "cuBLAS eror: %d %s %d \n", \
                status, \
                __FILE__, \
                __LINE__); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

using i64 = std::int64_t;
using u64 = std::uint64_t;

template <typename T>
__device__ void divmod(T x, T y, T *q, T *r) {
    *q = x / y;
    *r = x % y;
}

__device__ i64 to_tensor_idx2d(i64 flat_idx, i64 dim1, i64 stride0, i64 stride1);
__device__ i64 to_tensor_idx3d(i64 flat_idx, i64 dim1, i64 dim2, i64 stride0, i64 stride1, i64 stride2);
__device__ i64 to_tensor_idx4d(i64 flat_idx, i64 dim1, i64 dim2, i64 dim3, i64 stride0, i64 stride1, i64 stride2, i64 stride3);

#endif
