#include <cblas.h>

// C = A x B
#define DEFINE_NN_N(P, TYPE) \
    void my_##P##gemm_nn_n( \
        const TYPE *A, \
        const TYPE *B, \
        TYPE *C, \
        int M, \
        int N, \
        int K, \
        TYPE alpha, \
        TYPE beta \
    ) { \
        cblas_##P##gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N); \
    }

// C = A x trans(B)
#define DEFINE_NT_N(P, TYPE) \
    void my_##P##gemm_nt_n( \
        const TYPE *A, \
        const TYPE *B, \
        TYPE *C, \
        int M, \
        int N, \
        int K, \
        TYPE alpha, \
        TYPE beta \
    ) { \
        cblas_##P##gemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, N); \
    }

DEFINE_NN_N(s, float);
DEFINE_NT_N(s, float);

DEFINE_NN_N(d, double);
DEFINE_NT_N(d, double);
