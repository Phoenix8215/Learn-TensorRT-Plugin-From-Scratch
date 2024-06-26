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
    dim3 blockSize(256, 1, 1);
    dim3 gridSize(ceil(float(nElements) / 256), 1, 1);
    mishKernel<<<gridSize, blockSize, 0, stream>>>(inputs, outputs, nElements);
}
