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

using i32 = std::int32_t;
using i64 = std::int64_t;
using u64 = std::uint64_t;

template <typename T>
__device__ void divmod(T x, T y, T *q, T *r) {
    *q = x / y;
    *r = x % y;
}

__device__ int to_tensor_idx2d(int flat_idx, int dim1, int stride0, int stride1);
__device__ int to_tensor_idx3d(int flat_idx, int dim1, int dim2, int stride0, int stride1, int stride2);
__device__ int to_tensor_idx4d(int flat_idx, int dim1, int dim2, int dim3, int stride0, int stride1, int stride2, int stride3);
__device__ int to_tensor_idx5d(int flat_idx, int dim1, int dim2, int dim3, int dim4, int stride0, int stride1, int stride2, int stride3, int stride4);

#endif
