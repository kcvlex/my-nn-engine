#include <cudnn.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      fprintf(stderr, "Error on line %d: %s\n", __LINE__, cudnnGetErrorString(status)); \
      exit(EXIT_FAILURE);                               \
    }                                                        \
  }


extern "C" void my_conv(void **output_ptrs, void **input_ptrs, void **) {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    float *input = static_cast<float *>(input_ptrs[0]);
    float *output = static_cast<float *>(output_ptrs[0]);
    float *kernel = static_cast<float *>(input_ptrs[1]);

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            1, 1, 5, 5));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW,
                                            1, 1, 3, 3));
    
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                                 1, 1, // padding
                                                 1, 1, // stride
                                                 1, 1, // dilation
                                                 CUDNN_CROSS_CORRELATION,
                                                 CUDNN_DATA_FLOAT));
    
    int batch_size{0}, channels{0}, height{0}, width{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     &batch_size,
                                                     &channels,
                                                     &height,
                                                     &width));
    
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            batch_size, channels, height, width));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                         input_descriptor,
                                                         kernel_descriptor,
                                                         convolution_descriptor,
                                                         output_descriptor,
                                                         algo,
                                                         &workspace_bytes));

    float *d_input, *d_output, *d_workspace, *d_kernel;
    size_t input_bytes = 1 * 1 * 5 * 5 * sizeof(float);
    size_t kernel_bytes = 1 * 1 * 3 * 3 * sizeof(float);
    size_t output_bytes = batch_size * channels * height * width * sizeof(float);
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMalloc(&d_workspace, workspace_bytes);
    cudaMalloc(&d_kernel, kernel_bytes);

    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);

  const float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     d_input,
                                     kernel_descriptor,
                                     d_kernel,
                                     convolution_descriptor,
                                     algo,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     d_output));

  cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_kernel);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
  cudnnDestroy(cudnn);
}
