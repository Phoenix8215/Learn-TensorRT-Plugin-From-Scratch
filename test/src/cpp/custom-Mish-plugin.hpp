#ifndef __CUSTOM_MISH_PLUGIN_HPP
#define __CUSTOM_MISH_PLUGIN_HPP

#include <NvInfer.h>
#include <string>
#include <vector>
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"

using namespace nvinfer1;

namespace custom {
static const char* PLUGIN_NAME{"Mish"};
static const char* PLUGIN_VERSION{"1"};

class MishPlugin : public IPluginV2DynamicExt {
public:
    MishPlugin() = delete;
    MishPlugin(const std::string& name);
    MishPlugin(const std::string& name, const void* buffer, size_t length);
    ~MishPlugin();

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    size_t getSerializationSize() const noexcept override;
    const char* getPluginNamespace() const noexcept override;
    DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* input, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;

    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOuts, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    void attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;

private:
    const std::string mName;
    std::string mNamespace;
};

class MishPluginCreator : public IPluginCreator {
public:
    MishPluginCreator();
    ~MishPluginCreator();

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    const char* getPluginNamespace() const noexcept override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mAttrs;
    std::string mNamespace;
};

}  // namespace custom

#endif  // __CUSTOM_MISH_PLUGIN_HPP
