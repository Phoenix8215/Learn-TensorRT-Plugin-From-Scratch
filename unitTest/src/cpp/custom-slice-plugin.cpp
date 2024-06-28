#include "custom-slice-plugin.hpp"
#include <cuda_runtime_api.h>
#include <vector>
#include <cstring>
#include <cassert>
#include <iostream>
using namespace nvinfer1;

namespace custom
{

REGISTER_TENSORRT_PLUGIN(CustomSlicePluginCreator);

PluginFieldCollection   CustomSlicePluginCreator::mFC {};
std::vector<PluginField> CustomSlicePluginCreator::mAttrs;

CustomSlicePlugin::CustomSlicePlugin(const std::string &name, int start, int size) 
: mName(name), mStart(start), mSize(size) {}

CustomSlicePlugin::CustomSlicePlugin(const std::string &name, const void* buffer, size_t length) 
: mName(name) 
{
    const char* d = static_cast<const char*>(buffer);
    mStart = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mSize = *reinterpret_cast<const int*>(d);
}

CustomSlicePlugin::~CustomSlicePlugin() {}

const char* CustomSlicePlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* CustomSlicePlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t CustomSlicePlugin::getNbOutputs() const noexcept
{
    return 1;
}

size_t CustomSlicePlugin::getSerializationSize() const noexcept
{
    return 2 * sizeof(int);
}

void CustomSlicePlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<int*>(d) = mStart; d += sizeof(int);
    *reinterpret_cast<int*>(d) = mSize;
}

void CustomSlicePlugin::destroy() noexcept
{
    delete this;
}

const char* CustomSlicePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void CustomSlicePlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

int32_t CustomSlicePlugin::initialize() noexcept
{
    return 0;
}

void CustomSlicePlugin::terminate() noexcept {}

size_t CustomSlicePlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t CustomSlicePlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    int numElements = mSize * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
    cudaMemcpyAsync(output, input + mStart * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3], numElements * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    return 0;
}

nvinfer1::DataType CustomSlicePlugin::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

DimsExprs CustomSlicePlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs output(inputs[0]);
    output.d[1] = exprBuilder.constant(mSize);
    return output;
}

bool CustomSlicePlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
}

nvinfer1::IPluginV2DynamicExt* CustomSlicePlugin::clone() const noexcept
{
    return new CustomSlicePlugin(mName, mStart, mSize);
}

void CustomSlicePlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {}

void CustomSlicePlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator* gpuAllocator) noexcept {}

void CustomSlicePlugin::detachFromContext() noexcept {}

CustomSlicePluginCreator::CustomSlicePluginCreator()
{
    mAttrs.emplace_back(PluginField("start", nullptr, PluginFieldType::kINT32, 1));
    mAttrs.emplace_back(PluginField("size", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mAttrs.size();
    mFC.fields   = mAttrs.data();
}

CustomSlicePluginCreator::~CustomSlicePluginCreator() {}

const char* CustomSlicePluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* CustomSlicePluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const char* CustomSlicePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

IPluginV2* CustomSlicePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept 
{
    int start = 0;
    int size = 0;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; i++)
    {
        if (strcmp(fields[i].name, "start") == 0)
        {
            start = *static_cast<const int*>(fields[i].data);
        }
        else if (strcmp(fields[i].name, "size") == 0)
        {
            size = *static_cast<const int*>(fields[i].data);
        }
    }
    return new CustomSlicePlugin(name, start, size);
}

IPluginV2* CustomSlicePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new CustomSlicePlugin(name, serialData, serialLength);
}

void CustomSlicePluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const PluginFieldCollection* CustomSlicePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

} // namespace custom
