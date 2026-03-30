#ifndef INCLUDE_CUDNN_SETTING_H_
#define INCLUDE_CUDNN_SETTING_H_

#include <cudnn.h>
#include <cstddef>

struct CudnnHandlerContext {
    cudnnHandle_t handle;
    void *workspace = nullptr;
    size_t workspace_max_size_in_bytes = 1;
};

struct CudnnConvSetting {
    double alpha_ = 1.0;
    cudnnTensorDescriptor_t x_desc = nullptr;
    void *x = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    void *w = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    double beta_ = 0.0;
    size_t workspace_size_in_bytes = 1;
    cudnnTensorDescriptor_t y_desc = nullptr;
    void *y = nullptr;

    // For bias activation forward conv
    double alpha2_ = 0.0;
    // const cudnnTensorDescriptor_t z_desc = nullptr;
    // const void *z = nullptr;
    cudnnTensorDescriptor_t bias_desc = nullptr;
    void *bias = nullptr;
    cudnnActivationDescriptor_t activation_desc = nullptr;

    void find_best_algo(CudnnHandlerContext *ctx) {
        cudnnConvolutionFwdAlgoPerf_t perf[8];
        int count = 0;
        cudnnFindConvolutionForwardAlgorithmEx(
            ctx->handle,
            x_desc, x,
            w_desc, w,
            conv_desc,
            y_desc, y,
            8, &count, perf,
            ctx->workspace, ctx->workspace_max_size_in_bytes
        );
        if (count > 0 && perf[0].status == CUDNN_STATUS_SUCCESS) {
            algo = perf[0].algo;
            workspace_size_in_bytes = perf[0].memory;
        }
    }

    template <typename Float>
    cudnnStatus_t call_conv_forward(CudnnHandlerContext *ctx) {
        find_best_algo(ctx);
        Float alpha = static_cast<Float>(alpha_);
        Float beta = static_cast<Float>(beta_);
        return cudnnConvolutionForward(
            ctx->handle,
            &alpha,
            x_desc,
            x,
            w_desc,
            w,
            conv_desc,
            algo,
            ctx->workspace,
            workspace_size_in_bytes,
            &beta,
            y_desc,
            y
        );
    }

    template <typename Float>
    cudnnStatus_t call_conv_bias_activation_forward(CudnnHandlerContext *ctx) {
        find_best_algo(ctx);
        Float alpha1 = static_cast<Float>(alpha_);
        Float alpha2 = static_cast<Float>(alpha2_);

        // y = act (alpha1 * conv(x) + alpha2 * z + bias)
        return cudnnConvolutionBiasActivationForward(
            ctx->handle,
            &alpha1,
            x_desc,
            x,
            w_desc,
            w,
            conv_desc,
            algo,
            ctx->workspace,
            workspace_size_in_bytes,
            &alpha2,
            // z_desc,
            // z,
            y_desc,
            y,
            bias_desc,
            bias,
            activation_desc,
            y_desc,
            y
        );
    }
};

#endif
