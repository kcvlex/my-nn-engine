#ifndef INCLUDE_LAYER_NORM_CUH_
#define INCLUDE_LAYER_NORM_CUH_

#include <cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// From https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
//
//   Mean = ReduceMean<axes=normalized_axes>(X)
//   D = Sub(X, Mean)
//   DD = Mul(D, D)
//   Var = ReduceMean<axes=normalized_axes>(DD)
//   VarEps = Add(Var, epsilon)
//   StdDev = Sqrt(VarEps)
//   InvStdDev = Reciprocal(StdDev)
//   Normalized = Mul(D, InvStdDev)
//   NormalizedScaled = Mul(Normalized, Scale)
//   Y = Add(NormalizedScaled, B)
template <typename T, int BLOCK_SIZE, int DIM>
__global__ void layer_norm(
    T *out,
    T *in,
    T *scale,
    T *bias,
    T epsilon,
    int input_size
) {
    __shared__ T buf[BLOCK_SIZE];
    __shared__ T mean;
    constexpr int REPEAT = (DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int tid = threadIdx.x;
    if (input_size <= blockIdx.x * DIM) return;

    T inputs[REPEAT] = {};
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    for (int i = 0, offset = tid; i < REPEAT && offset < DIM; i++, offset += BLOCK_SIZE) {
        inputs[i] = in[blockIdx.x * DIM + offset];
    }

    // Mean = ReduceMean<axes=normalized_axes>(X)
    {
        T sum_acc = 0;
        for (int i = 0; i < REPEAT; i++) {
            sum_acc += inputs[i];
        }
        for (int s = tile32.size() / 2; 0 < s; s /= 2) {
            T other = tile32.shfl_down(sum_acc, s);
            sum_acc += other;
        }
        buf[tid] = sum_acc;
        cg::sync(cta);
        for (int s = BLOCK_SIZE / 2; tile32.size() <= s; s >>= 1) {
            if (tid < s) {
                buf[tid] += buf[tid + s];
            }
            cg::sync(cta);
        }
        if (tid == 0) mean = buf[0] / DIM;
        cg::sync(cta);
    }

    // D = Sub(X, Mean)
    // DD = Mul(D, D)
    T vars[REPEAT] = {};
    for (int i = 0; i < REPEAT && i * BLOCK_SIZE + tid < DIM; i++) {
        inputs[i] -= mean;
        vars[i] = inputs[i] * inputs[i];
    }

    // Var = ReduceMean<axes=normalized_axes>(DD)
    {
        T sum_acc = 0;
        for (int i = 0; i < REPEAT; i++) {
            sum_acc += vars[i];
        }
        for (int s = tile32.size() / 2; 0 < s; s /= 2) {
            T other = tile32.shfl_down(sum_acc, s);
            sum_acc += other;
        }
        buf[tid] = sum_acc;
        cg::sync(cta);
        for (int s = BLOCK_SIZE / 2; tile32.size() <= s; s >>= 1) {
            if (tid < s) {
                buf[tid] += buf[tid + s];
            }
            cg::sync(cta);
        }
        if (tid == 0) mean = buf[0] / DIM;
        cg::sync(cta);
    }

    // VarEps = Add(Var, epsilon)
    // StdDev = Sqrt(VarEps)
    T std_dev = sqrt(mean + epsilon);

    // InvStdDev = Reciprocal(StdDev)
    // Normalized = Mul(D, InvStdDev)
    // NormalizedScaled = Mul(Normalized, Scale)
    // Y = Add(NormalizedScaled, B)
    for (int i = 0, offset = tid; i < REPEAT && offset < DIM; i++, offset += BLOCK_SIZE) {
        T y = inputs[i] / std_dev;
        y *= scale[offset];
        y += bias[offset];
        out[blockIdx.x * DIM + offset] = y;
    }
}

#endif
