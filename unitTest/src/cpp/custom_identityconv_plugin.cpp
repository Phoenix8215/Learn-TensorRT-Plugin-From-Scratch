#include "custom_identityconv_plugin.hpp"
#include <cuda_runtime_api.h>
#include <vector>
#include <cstring>
#include <cassert>

using namespace nvinfer1;

namespace custom
{

REGISTER_TENSORRT_PLUGIN(CustomIdentityConvPluginCreator);

PluginFieldCollection   CustomIdentityConvPluginCreator::mFC {};
std::vector<PluginField> CustomIdentityConvPluginCreator::mAttrs;

CustomIdentityConvPlugin::CustomIdentityConvPlugin(const std::string &name, int channels, int kernel_size, int stride, int padding, int group) 
: mName(name), mChannels(channels), mKernelSize(kernel_size), mStride(stride), mPadding(padding), mGroup(group) {}

CustomIdentityConvPlugin::CustomIdentityConvPlugin(const std::string &name, const void* buffer, size_t length) 
: mName(name) 
{
    const char* d = static_cast<const char*>(buffer);
    mChannels = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mKernelSize = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mStride = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mPadding = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mGroup = *reinterpret_cast<const int*>(d);
}

CustomIdentityConvPlugin::~CustomIdentityConvPlugin() {}

const char* CustomIdentityConvPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* CustomIdentityConvPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t CustomIdentityConvPlugin::getNbOutputs() const noexcept
{
    return 1;
}

size_t CustomIdentityConvPlugin::getSerializationSize() const noexcept
{
    return 5 * sizeof(int);
}

void CustomIdentityConvPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<int*>(d) = mChannels; d += sizeof(int);
    *reinterpret_cast<int*>(d) = mKernelSize; d += sizeof(int);
    *reinterpret_cast<int*>(d) = mStride; d += sizeof(int);
    *reinterpret_cast<int*>(d) = mPadding; d += sizeof(int);
    *reinterpret_cast<int*>(d) = mGroup;
}

void CustomIdentityConvPlugin::destroy() noexcept
{
    delete this;
}

const char* CustomIdentityConvPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void CustomIdentityConvPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

int32_t CustomIdentityConvPlugin::initialize() noexcept
{
    return 0;
}

void CustomIdentityConvPlugin::terminate() noexcept {}

size_t CustomIdentityConvPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t CustomIdentityConvPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    int numElements = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
    cudaMemcpyAsync(output, input, numElements * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    return 0;
}

DimsExprs CustomIdentityConvPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool CustomIdentityConvPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
}

nvinfer1::IPluginV2DynamicExt* CustomIdentityConvPlugin::clone() const noexcept
{
    return new CustomIdentityConvPlugin(mName, mChannels, mKernelSize, mStride, mPadding, mGroup);
}

void CustomIdentityConvPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {}

void CustomIdentityConvPlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator* gpuAllocator) noexcept {}

void CustomIdentityConvPlugin::detachFromContext() noexcept {}

CustomIdentityConvPluginCreator::CustomIdentityConvPluginCreator()
{
    mAttrs.emplace_back(PluginField("channels", nullptr, PluginFieldType::kINT32, 1));
    mAttrs.emplace_back(PluginField("kernel_size", nullptr, PluginFieldType::kINT32, 1));
    mAttrs.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
    mAttrs.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 1));
    mAttrs.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mAttrs.size();
    mFC.fields = mAttrs.data();
}

const char* CustomIdentityConvPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* CustomIdentityConvPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* CustomIdentityConvPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* CustomIdentityConvPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    int channels = 0;
    int kernel_size = 0;
    int stride = 0;
    int padding = 0;
    int group = 0;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (strcmp(fc->fields[i].name, "channels") == 0)
        {
            channels = *static_cast<const int*>(fc->fields[i].data);
        }
        if (strcmp(fc->fields[i].name, "kernel_size") == 0)
        {
            kernel_size = *static_cast<const int*>(fc->fields[i].data);
        }
        if (strcmp(fc->fields[i].name, "stride") == 0)
        {
            stride = *static_cast<const int*>(fc->fields[i].data);
        }
        if (strcmp(fc->fields[i].name, "padding") == 0)
        {
            padding = *static_cast<const int*>(fc->fields[i].data);
        }
        if (strcmp(fc->fields[i].name, "group") == 0)
        {
            group = *static_cast<const int*>(fc->fields[i].data);
        }
    }
    return new CustomIdentityConvPlugin(name, channels, kernel_size, stride, padding, group);
}

IPluginV2* CustomIdentityConvPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new CustomIdentityConvPlugin(name, serialData, serialLength);
}

void CustomIdentityConvPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* CustomIdentityConvPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

} // namespace custom
