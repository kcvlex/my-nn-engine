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

using i64 = std::int64_t;
using u64 = std::uint64_t;
