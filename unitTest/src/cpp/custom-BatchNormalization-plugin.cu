#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(err); \
        } \
    } while (0)

// CUDA 核函数和 Launcher
__global__ void batchNormalizationKernel(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, float epsilon, int n, int c, int h, int w);

void batchNormalizationKernelLauncher(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, float epsilon, int n, int c, int h, int w, cudaStream_t stream) {
    int size = n * c * h * w;
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    batchNormalizationKernel<<<gridSize, blockSize, 0, stream>>>(input, output, scale, bias, mean, variance, epsilon, n, c, h, w);
    CUDA_CHECK(cudaGetLastError());
}

// 核函数实现
__global__ void batchNormalizationKernel(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, float epsilon, int n, int c, int h, int w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int size = n * c * h * w;
    if (index < size) {
        int nhw = h * w;
        int chw = c * nhw;
        int n_idx = index / chw;
        int c_idx = (index % chw) / nhw;
        int hw_idx = index % nhw;

        float x = input[index];
        float m = mean[c_idx];
        float v = variance[c_idx];
        float gamma = scale[c_idx];
        float beta = bias[c_idx];

        output[index] = gamma * ((x - m) / sqrtf(v + epsilon)) + beta;
    }
}
