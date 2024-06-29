#include <cuda_runtime_api.h>
#include <vector>
#include <cstring>
#include <cassert>
#include <iostream>
#include "custom_identityconv_plugin.hpp"
#include <sstream>
#include "utils.hpp"

using namespace nvinfer1;

namespace custom
{
// Write values into buffer
template <typename Type, typename BufferType>
void write(BufferType*& buffer, Type const& val) {
    static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
    std::memcpy(buffer, &val, sizeof(Type));
    buffer += sizeof(Type);
}

// Read values from buffer
template <typename OutType, typename BufferType>
OutType read(BufferType const*& buffer) {
    static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
    OutType val{};
    std::memcpy(&val, static_cast<void const*>(buffer), sizeof(OutType));
    buffer += sizeof(OutType);
    return val;
}

REGISTER_TENSORRT_PLUGIN(IdentityConvPluginCreator);

PluginFieldCollection IdentityConvPluginCreator::mFC {};
std::vector<PluginField> IdentityConvPluginCreator::mPluginAttributes;

IdentityConvPlugin::IdentityConvPlugin(const std::string& name, IdentityConvParameters params)
    : mName(name), mParams{params} {}

IdentityConvPlugin::IdentityConvPlugin(const std::string &name, const void* buffer, size_t length) 
: mName(name) {
    deserialize(static_cast<uint8_t const*>(buffer), length);
}

void IdentityConvPlugin::deserialize(uint8_t const* data, size_t length) {
    // In our simple use case, even though there is no parameter used for this
    // plugin, we deserialize and serialize some attributes for demonstration
    // purposes.
    uint8_t const* d{data};
    mParams.group = read<int32_t>(d);
    mParams.dtype = read<nvinfer1::DataType>(d);
    mParams.channelSize = read<int32_t>(d);
    mParams.height = read<int32_t>(d);
    mParams.width = read<int32_t>(d);
    mParams.dtypeBytes = read<size_t>(d);
    PLUGIN_ASSERT(d == data + length);
}

IdentityConvPlugin::~IdentityConvPlugin() {}

const char* IdentityConvPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* IdentityConvPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t IdentityConvPlugin::getNbOutputs() const noexcept
{
    return 1;
}

size_t IdentityConvPlugin::getSerializationSize() const noexcept
{
    return sizeof(int32_t) * 4 + sizeof(nvinfer1::DataType) + sizeof(size_t);
}

void IdentityConvPlugin::serialize(void* buffer) const noexcept
{
    char* d{reinterpret_cast<char*>(buffer)};
    char* const a{d};
    // Be cautious, the order has to match deserialization.
    write(d, mParams.group);
    write(d, mParams.dtype);
    write(d, mParams.channelSize);
    write(d, mParams.height);
    write(d, mParams.width);
    write(d, mParams.dtypeBytes);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void IdentityConvPlugin::destroy() noexcept
{
    delete this;
}

const char* IdentityConvPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void IdentityConvPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

int32_t IdentityConvPlugin::initialize() noexcept
{
    // 在这个阶段，配置已经确定，并且推理引擎正在创建，
    // 因此插件可以设置其内部数据结构并为执行做准备。
    // 这种准备可能包括初始化库、分配内存等。然而，
    // 在我们的例子中，我们不需要进行任何准备工作。
    return 0;
}

void IdentityConvPlugin::terminate() noexcept {}

size_t IdentityConvPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t IdentityConvPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    size_t const inputSize{
        static_cast<size_t>(inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1] *
                            inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3])};
    size_t const inputSizeBytes{inputSize * mParams.dtypeBytes};
    cudaError_t const status{cudaMemcpyAsync(outputs[0], inputs[0],
                                             inputSizeBytes,
                                             cudaMemcpyDeviceToDevice, stream)};
    return status;
}

nvinfer1::DataType IdentityConvPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

DimsExprs IdentityConvPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::Dims dimsOutput;
    PLUGIN_ASSERT(inputs[0].nbDims == 4);
    // Identity operation.
    // Just copy the dimensions from the input tensor.
    dimsOutput.nbDims = inputs[0].nbDims;
    dimsOutput.d[0] = inputs[0].d[0];
    dimsOutput.d[1] = inputs[0].d[1];
    dimsOutput.d[2] = inputs[0].d[2];
    dimsOutput.d[3] = inputs[0].d[3];

    return dimsOutput;
}

bool IdentityConvPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // 在此方法中，输入的编号为 0..(nbInputs-1)，而输出的编号为 nbInputs..(nbInputs+nbOutputs-1)。
    // 根据这种编号方式，pos 是 InOut 数组的一个索引，其中 0 <= pos < nbInputs+nbOutputs。
    PLUGIN_ASSERT(nbInputs == 2 && nbOutputs == 1 &&
                  pos < nbInputs + nbOutputs);
    bool isValidCombination = false;

    // Suppose we support only a limited number of format configurations.
    isValidCombination |=
        (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[pos].type == nvinfer1::DataType::kFLOAT);
    isValidCombination |=
        (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[pos].type == nvinfer1::DataType::kHALF);
    // Make sure the input tensor and output tensor types and formats are same.
    isValidCombination &=
        (pos < nbInputs || (inOut[pos].format == inOut[0].format &&
                            inOut[pos].type == inOut[0].type));

    return isValidCombination;
}

nvinfer1::IPluginV2DynamicExt* IdentityConvPlugin::clone() const noexcept
{
    // It's possible to encounter errors during cloning.
    // For example, if the memory to allocate is insufficient, exceptions can be
    // thrown.
    try {
        nvinfer1::IPluginV2DynamicExt* const plugin{new IdentityConvPlugin{mParams}};
        plugin->setPluginNamespace(mNamespace);
        return plugin;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}
void IdentityConvPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {
        // 此函数传达以下信息：输入和输出的数量、维度和数据类型，各输入和输出的广播信息，
        // 所选插件格式，以及最大批处理大小。插件在此阶段设置其内部状态，
        // 并根据给定配置选择最合适的算法和数据结构。注意：在此 API 中不允许进行资源分配，因为这可能导致资源泄漏。
        PLUGIN_ASSERT(nbInput == 2);
        PLUGIN_ASSERT(nbOutput == 1);
        PLUGIN_ASSERT(in[0].dims.nbDims == 3);
        PLUGIN_ASSERT(out[0].dims.nbDims == 3);
        PLUGIN_ASSERT(in[0].dims.d[0] == out[0].dims.d[0]);
        PLUGIN_ASSERT(in[0].dims.d[1] == out[0].dims.d[1]);
        PLUGIN_ASSERT(in[0].dims.d[2] == out[0].dims.d[2]);
        PLUGIN_ASSERT(in[0].dims.d[3] == out[0].dims.d[3]);
        PLUGIN_ASSERT(in[0].type == out[0].type);

        mParams.dtype = in[0].type;
        mParams.channelSize = in[0].dims.d[0];
        mParams.height = in[0].dims.d[1];
        mParams.width = in[0].dims.d[2];

        if (mParams.dtype == nvinfer1::DataType::kINT8) {
            mParams.dtypeBytes = 1;
        } else if (mParams.dtype == nvinfer1::DataType::kHALF) {
            mParams.dtypeBytes = 2;
        } else if (mParams.dtype == nvinfer1::DataType::kFLOAT) {
            mParams.dtypeBytes = 4;
        } else {
            PLUGIN_ASSERT(false);
        }
        // 请注意，此成员函数仅在引擎构建期间被调用。
}

void IdentityConvPlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator* gpuAllocator) noexcept {}

void IdentityConvPlugin::detachFromContext() noexcept {}

IdentityConvPluginCreator::IdentityConvPluginCreator()
{
    // 声明 ONNX 属性，ONNX 解析器将从包含 IdentityConv 节点的 ONNX 模型中收集这些属性。

    // 在我们的示例中，
    // attrs={
    //     "kernel_shape": [1, 1],
    //     "strides": [1, 1],
    //     "pads": [0, 0, 0, 0],
    //     "group": num_groups
    // }
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "kernel_shape", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("strides", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("pads", nullptr, PluginFieldType::kINT32, 4));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("group", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

IdentityConvPluginCreator::~IdentityConvPluginCreator() {}

const char* IdentityConvPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* IdentityConvPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const char* IdentityConvPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

IPluginV2* IdentityConvPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept 
{
    // The attributes from the ONNX node will be parsed and passed via fc.
    try {
        nvinfer1::PluginField const* fields{fc->fields};
        int32_t nbFields{fc->nbFields};

        PLUGIN_VALIDATE(nbFields == 4);

        std::vector<int32_t> kernelShape{};
        std::vector<int32_t> strides{};
        std::vector<int32_t> pads{};
        int32_t group{};

        for (int32_t i{0}; i < nbFields; ++i) {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "kernel_shape")) {
                PLUGIN_VALIDATE(fields[i].type ==
                                nvinfer1::PluginFieldType::kINT32);
                int32_t const* const kernelShapeData{
                    static_cast<int32_t const*>(fields[i].data)};
                for (int32_t j{0}; j < fields[i].length; ++j) {
                    kernelShape.push_back(kernelShapeData[j]);
                }
            }
            if (!strcmp(attrName, "strides")) {
                PLUGIN_VALIDATE(fields[i].type ==
                                nvinfer1::PluginFieldType::kINT32);
                int32_t const* const stridesData{
                    static_cast<int32_t const*>(fields[i].data)};
                for (int32_t j{0}; j < fields[i].length; ++j) {
                    strides.push_back(stridesData[j]);
                }
            }
            if (!strcmp(attrName, "pads")) {
                PLUGIN_VALIDATE(fields[i].type ==
                                nvinfer1::PluginFieldType::kINT32);
                int32_t const* const padsData{
                    static_cast<int32_t const*>(fields[i].data)};
                for (int32_t j{0}; j < fields[i].length; ++j) {
                    pads.push_back(padsData[j]);
                }
            }
            if (!strcmp(attrName, "group")) {
                PLUGIN_VALIDATE(fields[i].type ==
                                nvinfer1::PluginFieldType::kINT32);
                PLUGIN_VALIDATE(fields[i].length == 1);
                group = *(static_cast<int32_t const*>(fields[i].data));
            }
        }

        // Log the attributes parsed from ONNX node.
        std::stringstream ss;
        ss << "Plugin Attributes:";
        logInfo(ss.str().c_str());

        ss.str("");
        ss << "kernel_shape: ";
        for (auto const& val : kernelShape) {
            ss << val << " ";
        }
        logInfo(ss.str().c_str());

        ss.str("");
        ss << "strides: ";
        for (auto const& val : strides) {
            ss << val << " ";
        }
        logInfo(ss.str().c_str());

        ss.str("");
        ss << "pads: ";
        for (auto const& val : pads) {
            ss << val << " ";
        }
        logInfo(ss.str().c_str());

        ss.str("");
        ss << "group: " << group;
        logInfo(ss.str().c_str());

        IdentityConvParameters const params{.group = group};

        IdentityConvPlugin* const plugin{new IdentityConvPlugin{mName.c_str(),params}};
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* IdentityConvPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    try {
        IdentityConvPlugin* plugin = new IdentityConvPlugin{serialData, serialLength};
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

void IdentityConvPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const PluginFieldCollection* IdentityConvPluginCreator::getFieldNames() noexcept
{
    // This is only used in the build phase.
    return &mFC;
}

} // namespace custom
