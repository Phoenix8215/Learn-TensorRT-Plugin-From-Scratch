#include <cuda_runtime_api.h>
#include <cassert>
#include <cstring>
#include <vector>
#include "custom-BatchNormalization-plugin.hpp"

/* BatchNormalization的核函数接口部分 */
void batchNormalizationKernelLauncher(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, float epsilon, int n, int c, int h, int w, cudaStream_t stream);

using namespace nvinfer1;

namespace custom {

REGISTER_TENSORRT_PLUGIN(CustomBatchNormalizationPluginCreator);

PluginFieldCollection CustomBatchNormalizationPluginCreator::mFC{};
std::vector<PluginField> CustomBatchNormalizationPluginCreator::mAttrs;

CustomBatchNormalizationPlugin::CustomBatchNormalizationPlugin(const std::string& name, float epsilon, std::vector<float> scale, std::vector<float> bias, std::vector<float> mean, std::vector<float> variance)
    : mName(name), mEpsilon(epsilon), mScale(scale), mBias(bias), mMean(mean), mVariance(variance) {}

CustomBatchNormalizationPlugin::CustomBatchNormalizationPlugin(const std::string& name, const void* buffer, size_t length)
    : mName(name) {
    const char *d = static_cast<const char*>(buffer), *a = d;
    mEpsilon = read<float>(d);
    size_t size = read<size_t>(d);
    mScale.assign(d, d + size);
    d += size * sizeof(float);
    size = read<size_t>(d);
    mBias.assign(d, d + size);
    d += size * sizeof(float);
    size = read<size_t>(d);
    mMean.assign(d, d + size);
    d += size * sizeof(float);
    size = read<size_t>(d);
    mVariance.assign(d, d + size);
    assert(d == a + length);
}

CustomBatchNormalizationPlugin::~CustomBatchNormalizationPlugin() {}

const char* CustomBatchNormalizationPlugin::getPluginType() const noexcept {
    return PLUGIN_NAME;
}

const char* CustomBatchNormalizationPlugin::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

int32_t CustomBatchNormalizationPlugin::getNbOutputs() const noexcept {
    return 1;
}

size_t CustomBatchNormalizationPlugin::getSerializationSize() const noexcept {
    return sizeof(mEpsilon) + sizeof(size_t) + mScale.size() * sizeof(float) +
           sizeof(size_t) + mBias.size() * sizeof(float) +
           sizeof(size_t) + mMean.size() * sizeof(float) +
           sizeof(size_t) + mVariance.size() * sizeof(float);
}

void CustomBatchNormalizationPlugin::serialize(void* buffer) const noexcept {
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mEpsilon);
    write(d, mScale.size());
    std::memcpy(d, mScale.data(), mScale.size() * sizeof(float));
    d += mScale.size() * sizeof(float);
    write(d, mBias.size());
    std::memcpy(d, mBias.data(), mBias.size() * sizeof(float));
    d += mBias.size() * sizeof(float);
    write(d, mMean.size());
    std::memcpy(d, mMean.data(), mMean.size() * sizeof(float));
    d += mMean.size() * sizeof(float);
    write(d, mVariance.size());
    std::memcpy(d, mVariance.data(), mVariance.size() * sizeof(float));
    d += mVariance.size() * sizeof(float);
    assert(d == a + getSerializationSize());
}

void CustomBatchNormalizationPlugin::destroy() noexcept {
    delete this;
}

const char* CustomBatchNormalizationPlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

void CustomBatchNormalizationPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

int32_t CustomBatchNormalizationPlugin::initialize() noexcept {
    return 0;
}

void CustomBatchNormalizationPlugin::terminate() noexcept {}

size_t CustomBatchNormalizationPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t CustomBatchNormalizationPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    int n = inputDesc[0].dims.d[0];
    int c = inputDesc[0].dims.d[1];
    int h = inputDesc[0].dims.d[2];
    int w = inputDesc[0].dims.d[3];

    batchNormalizationKernelLauncher(
        static_cast<const float*>(inputs[0]),
        static_cast<float*>(outputs[0]),
        mScale.data(),
        mBias.data(),
        mMean.data(),
        mVariance.data(),
        mEpsilon,
        n, c, h, w, stream);

    return 0;
}

nvinfer1::DataType CustomBatchNormalizationPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept {
    return inputTypes[0];
}

DimsExprs CustomBatchNormalizationPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept {
    return inputs[0];
}

bool CustomBatchNormalizationPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    switch (pos) {
        case 0:
            return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
        case 1:
            return inOut[1].type == DataType::kFLOAT && inOut[1].format == TensorFormat::kLINEAR;
        default:
            return false;
    }
}

nvinfer1::IPluginV2DynamicExt* CustomBatchNormalizationPlugin::clone() const noexcept {
    return new CustomBatchNormalizationPlugin(mName, mEpsilon, mScale, mBias, mMean, mVariance);
}

void CustomBatchNormalizationPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {}

void CustomBatchNormalizationPlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator* gpuAllocator) noexcept {}

void CustomBatchNormalizationPlugin::detachFromContext() noexcept {}

CustomBatchNormalizationPluginCreator::CustomBatchNormalizationPluginCreator() {
    mAttrs.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    mAttrs.emplace_back(PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1));
    mAttrs.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));
    mAttrs.emplace_back(PluginField("mean", nullptr, PluginFieldType::kFLOAT32, 1));
    mAttrs.emplace_back(PluginField("variance", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mAttrs.size();
    mFC.fields = mAttrs.data();
}

CustomBatchNormalizationPluginCreator::~CustomBatchNormalizationPluginCreator() {}

const char* CustomBatchNormalizationPluginCreator::getPluginName() const noexcept {
    return PLUGIN_NAME;
}

const char* CustomBatchNormalizationPluginCreator::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

const char* CustomBatchNormalizationPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

IPluginV2* CustomBatchNormalizationPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    float epsilon = 1e-5;
    std::vector<float> scale, bias, mean, variance;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; i++) {
        if (strcmp(fields[i].name, "epsilon") == 0) {
            epsilon = *(static_cast<const float*>(fields[i].data));
        } else if (strcmp(fields[i].name, "scale") == 0) {
            scale.assign(static_cast<const float*>(fields[i].data), static_cast<const float*>(fields[i].data) + fields[i].length);
        } else if (strcmp(fields[i].name, "bias") == 0) {
            bias.assign(static_cast<const float*>(fields[i].data), static_cast<const float*>(fields[i].data) + fields[i].length);
        } else if (strcmp(fields[i].name, "mean") == 0) {
            mean.assign(static_cast<const float*>(fields[i].data), static_cast<const float*>(fields[i].data) + fields[i].length);
        } else if (strcmp(fields[i].name, "variance") == 0) {
            variance.assign(static_cast<const float*>(fields[i].data), static_cast<const float*>(fields[i].data) + fields[i].length);
        }
    }
    return new CustomBatchNormalizationPlugin(name, epsilon, scale, bias, mean, variance);
}

IPluginV2* CustomBatchNormalizationPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    return new CustomBatchNormalizationPlugin(name, serialData, serialLength);
}

void CustomBatchNormalizationPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const PluginFieldCollection* CustomBatchNormalizationPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

}  // namespace custom
