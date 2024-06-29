#include <cstring>
#include <map>
#include "custom-Mish-plugin.hpp"

/* Mish的核函数接口部分 */
void mishKernelLauncher(const float* inputs, float* outputs, const int nElements, cudaStream_t stream);

using namespace nvinfer1;

namespace custom {

REGISTER_TENSORRT_PLUGIN(MishPluginCreator);

PluginFieldCollection MishPluginCreator::mFC{};
std::vector<PluginField> MishPluginCreator::mAttrs;

MishPlugin::MishPlugin(const std::string& name)
    : mName(name) {}

MishPlugin::MishPlugin(const std::string& name, const void* buffer, size_t length)
    : mName(name) {}

MishPlugin::~MishPlugin() {}

const char* MishPlugin::getPluginType() const noexcept {
    return PLUGIN_NAME;
}

const char* MishPlugin::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

int32_t MishPlugin::getNbOutputs() const noexcept {
    return 1;
}

size_t MishPlugin::getSerializationSize() const noexcept {
    return 0;
}

void MishPlugin::serialize(void* buffer) const noexcept {}

void MishPlugin::destroy() noexcept {
    delete this;
}

const char* MishPlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

void MishPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

int MishPlugin::initialize() noexcept {
    return 0;
}

void MishPlugin::terminate() noexcept {}

size_t MishPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t MishPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    int nElements = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++) {
        nElements *= inputDesc[0].dims.d[i];
    }

    mishKernelLauncher(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]), nElements, stream);

    return 0;
}

nvinfer1::DataType MishPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept {
    return inputTypes[0];
}

DimsExprs MishPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept {
    return inputs[0];
}

bool MishPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    switch (pos) {
        case 0:
            return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
        case 1:
            return inOut[1].type == DataType::kFLOAT && inOut[1].format == TensorFormat::kLINEAR;
        default:
            return false;
    }
}

nvinfer1::IPluginV2DynamicExt* MishPlugin::clone() const noexcept {
    auto p = new MishPlugin(mName);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

void MishPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {}

void MishPlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator* gpuAllocator) noexcept {}

void MishPlugin::detachFromContext() noexcept {}

MishPluginCreator::MishPluginCreator() {
    mFC.nbFields = mAttrs.size();
    mFC.fields = mAttrs.data();
}

MishPluginCreator::~MishPluginCreator() {}

const char* MishPluginCreator::getPluginName() const noexcept {
    return PLUGIN_NAME;
}

const char* MishPluginCreator::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

const char* MishPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

IPluginV2* MishPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    return new MishPlugin(name);
}

IPluginV2* MishPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    return new MishPlugin(name, serialData, serialLength);
}

void MishPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const PluginFieldCollection* MishPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

}  // namespace custom
