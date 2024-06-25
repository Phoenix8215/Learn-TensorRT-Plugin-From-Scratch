#include <cuda_runtime.h>
#include <math.h>

__device__ float mish(float x)
{
    return x * tanh(log1p(exp(x)));
}

__global__ void mishKernel(const float* inputs, float* outputs, int nElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nElements)
    {
        outputs[idx] = mish(inputs[idx]);
    }
}

void mishKernelLauncher(const float* inputs, float* outputs, int nElements, cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (nElements + blockSize - 1) / blockSize;
    mishKernel<<<gridSize, blockSize, 0, stream>>>(inputs, outputs, nElements);
}
