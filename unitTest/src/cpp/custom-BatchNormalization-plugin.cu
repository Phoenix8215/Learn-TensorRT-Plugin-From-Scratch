#include <cuda_runtime.h>
#include <math.h>

__global__ void batchNormalizationKernel(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, float epsilon, int n, int c, int h, int w)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int size = n * c * h * w;
    if (index < size)
    {
        int n_stride = c * h * w;
        int c_stride = h * w;
        int h_stride = w;
        
        int n_idx = index / n_stride;
        int c_idx = (index % n_stride) / c_stride;
        int h_idx = (index % c_stride) / h_stride;
        int w_idx = index % w;

        float mean_val = mean[c_idx];
        float var_val = variance[c_idx];
        float scale_val = scale[c_idx];
        float bias_val = bias[c_idx];

        float norm_val = (input[index] - mean_val) / sqrtf(var_val + epsilon);
        output[index] = norm_val * scale_val + bias_val;
    }
}

void batchNormalizationKernelLauncher(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, float epsilon, int n, int c, int h, int w, cudaStream_t stream)
{
    int size = n * c * h * w;
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    batchNormalizationKernel<<<numBlocks, blockSize, 0, stream>>>(input, output, scale, bias, mean, variance, epsilon, n, c, h, w);
}
