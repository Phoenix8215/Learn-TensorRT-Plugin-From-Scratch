#include "custom-scalar-plugin.hpp"
#include "utils.hpp"
#include <cstring>
#include <map>
#include <cassert>
/* customScalar的核函数接口部分 */
void customScalarImpl(const float* inputs, float* outputs, const float scalar, const float scale, const int nElements, cudaStream_t stream);

using namespace nvinfer1;

namespace custom {
/******************************************************************/
/********************注册PluginCreator*****************************/
/******************************************************************/
REGISTER_TENSORRT_PLUGIN(CustomScalarPluginCreator);

/******************************************************************/
/*********************静态变量类外初始化*******************************/
/******************************************************************/
PluginFieldCollection CustomScalarPluginCreator::mFC {};
std::vector<PluginField> CustomScalarPluginCreator::mAttrs {};

/******************************************************************/
/*********************CustomScalarPlugin实现部分***********************/
/******************************************************************/
/*
 * 我们在编译的过程中会有大概有三次创建插件实例的过程
 * 1. parse阶段: 第一次读取onnx来parse这个插件。会读取参数信息并转换为TensorRT格式
 * 2. clone阶段: parse完了以后，TensorRT为了去优化这个插件会复制很多副本出来来进行很多优化测试。也可以在推理的时候供不同的context创建插件的时候使用
 * 3. deseriaze阶段: 将序列化好的Plugin进行反序列化的时候也需要创建插件的实例
 */
CustomScalarPlugin::CustomScalarPlugin(const std::string& name, float scalar, float scale)
    : mName(name) {
    mParams.scalar = scalar;
    mParams.scale = scale;
}

CustomScalarPlugin::CustomScalarPlugin(const std::string& name, const void* buffer, size_t length)
    : mName(name) {
    memcpy(&mParams, buffer, sizeof(mParams));
}

CustomScalarPlugin::~CustomScalarPlugin() {
    /* 这里的析构函数不需要做任何事情，生命周期结束的时候会自动调用terminate和destroy */
    return;
}

// CustomScalarPlugin::~CustomScalarPlugin() = default;

const char* CustomScalarPlugin::getPluginType() const noexcept {
    /* 一般来说所有插件的实现差不多一致 */
    return PLUGIN_NAME;
}

const char* CustomScalarPlugin::getPluginVersion() const noexcept {
    /* 一般来说所有插件的实现差不多一致 */
    return PLUGIN_VERSION;
}

int32_t CustomScalarPlugin::getNbOutputs() const noexcept {
    /* 一般来说所有插件的实现差不多一致 */
    return 1;
}

size_t CustomScalarPlugin::getSerializationSize() const noexcept {
    /* 如果把所有的参数给放在mParams中的话, 一般来说所有插件的实现差不多一致 */
    return sizeof(mParams);
}

const char* CustomScalarPlugin::getPluginNamespace() const noexcept {
    /* 一般来说所有插件的实现差不多一致 */
    return mNamespace.c_str();
}

DataType CustomScalarPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept {
    /* 一般来说所有插件的实现差不多一致 */
    return inputTypes[0];
}

DimsExprs CustomScalarPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept {
    
    return inputs[0];
}

// 假设我们有一个插件，它将输入张量的每个维度加一：
// {
//     assert(nbInputs == 1); // 确保只有一个输入
//     DimsExprs outputDims;
//     outputDims.nbDims = inputs[0].nbDims;
    
//     for (int i = 0; i < inputs[0].nbDims; ++i)
//     {
//         // 每个维度加一
//         outputDims.d[i] = exprBuilder.operation(DimensionOperation::kSUM, *inputs[0].d[i], *exprBuilder.constant(1));
//     }
    
//     return outputDims;
// }

size_t CustomScalarPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept {
    /* 一般来说会使用builder创建时用的workspaceSize所以这里一般什么都不做 */
    return 0;
}

int32_t CustomScalarPlugin::initialize() noexcept {
    /*
    这个一般会根据情况而定，建议每个插件都有一个自己的实现
    主要初始化一些提前开辟空间的参数，一般是一些cuda操作需要的参数(例如conv操作需要执行卷积操作，
    我们就需要提前开辟weight和bias的显存)，假如我们的算子需要这些参数，则在这里需要提前开辟显存。

    需要注意的是，如果插件算子需要开辟比较大的显存空间，不建议自己去申请显存空间，
    可以使用Tensorrt官方接口传过来的workspace指针来获取显存空间。因为如果这个插件被一个网络调用了很多次，
    而这个插件op需要开辟很多显存空间，那么TensorRT在构建network的时候会根据这个插件被调用的次数开辟很多显存，
    很容易导致显存溢出。
    */
    return 0;
}

void CustomScalarPlugin::terminate() noexcept {
    /*
     * 这个是析构函数调用的函数。一般和initialize配对的使用
     * initialize分配多少内存，这里就释放多少内存
     */
    return;
}

void CustomScalarPlugin::serialize(void* buffer) const noexcept {
    /* 序列化也根据情况而定，每个插件自己定制 */
    memcpy(buffer, &mParams, sizeof(mParams));
    return;
}

void CustomScalarPlugin::destroy() noexcept {
    /* 一般来说所有插件的实现差不多一致 */
    delete this;
    return;
}

int32_t CustomScalarPlugin::enqueue(
    const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
    /*
     * Plugin的核心的地方。每个插件都有一个自己的定制方案
     * Plugin直接调用kernel的地方
     * 当然C++写的op也可以放进来，不过因为是CPU执行，速度就比较慢了
     */
    int nElements = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++) {
        nElements *= inputDesc[0].dims.d[i];
    }

    customScalarImpl(
        static_cast<const float*>(inputs[0]),
        static_cast<float*>(outputs[0]),
        mParams.scalar,
        mParams.scale,
        nElements,
        stream);

    return 0;
}

IPluginV2DynamicExt* CustomScalarPlugin::clone() const noexcept {
    /* 克隆一个Plugin对象，所有的插件的实现都差不多*/
    // 将这个plugin对象克隆一份给TensorRT的builder、network或者engine。这个成员函数会调用上述说到的第二个构造函数
    try
    {
        auto p = new CustomScalarPlugin(mName, &mParams, sizeof(mParams));
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch(const std::exception& e)
    {
        LOGE("ERROR detected when clone plugin: %s", e.what());
    }
    return nullptr;
    
}

bool CustomScalarPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    /*
     * 设置这个Plugin支持的Datatype以及TensorFormat, 每个插件都有自己的定制
     * 作为案例展示，这个customScalar插件只支持FP32，如果需要扩展到FP16以及INT8，需要在这里设置
     */
    // 检查输入
    if (pos < nbInputs) {
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }

    // 检查输出
    if (pos < nbInputs + nbOutputs) {
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
}

void CustomScalarPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {
    /* 一般不需要做任何使用，所有插件实现都差不多 */
    // 配置这个插件op，判断输入和输出类型数量是否正确。
    // 官方还提到通过这个配置信息可以告知TensorRT去选择合适的算法(algorithm)去调优这个模型。
    // 检查输入和输出的数量是否正确
    assert(nbInputs == 1);
    assert(nbOutputs == 1);

    // 检查输入和输出的格式和数据类型
    assert(in[0].desc.type == DataType::kFLOAT);
    assert(out[0].desc.type == DataType::kFLOAT);
    assert(in[0].desc.format == TensorFormat::kLINEAR);
    assert(out[0].desc.format == TensorFormat::kLINEAR);

    // 检查输入和输出的维度是否兼容
    // 此处假设输入和输出的维度相同
    assert(in[0].desc.dims.nbDims == out[0].desc.dims.nbDims);
    for (int i = 0; i < in[0].desc.dims.nbDims; ++i) {
        assert(in[0].desc.dims.d[i] == out[0].desc.dims.d[i]);
    }

    // 根据需要设置插件的内部状态或分配资源
    // 在这个示例中，我们不需要额外的配置
    return;
}
void CustomScalarPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    /* 所有插件的实现都差不多 */
    mNamespace = pluginNamespace;
    return;
}
void CustomScalarPlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator* gpuAllocator) noexcept {
    /* 一般不需要做任何使用，所有插件实现都差不多 */
    /*
    // 初始化 cuDNN 和 cuBLAS 句柄
    mCudnnHandle = contextCudnn;
    mCublasHandle = contextCublas;

    // 使用 gpuAllocator 分配 GPU 内存
    size_t size = 1024 * 1024; // 例如分配 1MB 的内存
    mGpuMemory = gpuAllocator->allocate(size, 0);
    */
    return;
}
void CustomScalarPlugin::detachFromContext() noexcept {
    /* 一般不需要做任何使用，所有插件实现都差不多 */
    /*
    // 释放 cuDNN 和 cuBLAS 句柄
    mCudnnHandle = nullptr;
    mCublasHandle = nullptr;

    // 释放 GPU 内存
    if (mGpuMemory)
    {
        // 这里假设 gpuAllocator 提供了 deallocate 函数
        // 如果没有提供，需要自行实现
        gpuAllocator->deallocate(mGpuMemory);
        mGpuMemory = nullptr;
    }
    */
    return;
}

/******************************************************************/
/*********************CustomScalarPluginCreator部分********************/
/******************************************************************/

CustomScalarPluginCreator::CustomScalarPluginCreator() {
    /*
     * 每个插件的Creator构造函数需要定制，主要就是获取参数以及传递参数
     * 初始化creator中的PluginField以及PluginFieldCollection
     * - PluginField::            负责获取onnx中的参数
     * - PluginFieldCollection：  负责将onnx中的参数传递给Plugin
     */

    mAttrs.emplace_back(PluginField("scalar", nullptr, PluginFieldType::kFLOAT32, 1));
    mAttrs.emplace_back(PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mAttrs.size();
    mFC.fields = mAttrs.data();
}

CustomScalarPluginCreator::~CustomScalarPluginCreator() {
    /* 一般不需要做任何使用，所有插件实现都差不多 */
}


// CustomScalarPluginCreator::~CustomScalarPluginCreator() = default;

const char* CustomScalarPluginCreator::getPluginName() const noexcept {
    /* 所有插件实现都差不多 */
    return PLUGIN_NAME;
}

const char* CustomScalarPluginCreator::getPluginVersion() const noexcept {
    /* 所有插件实现都差不多 */
    return PLUGIN_VERSION;
}

const char* CustomScalarPluginCreator::getPluginNamespace() const noexcept {
    /* 所有插件实现都差不多 */
    return mNamespace.c_str();
}

IPluginV2* CustomScalarPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    /*
     * 通过Creator创建一个Plugin的实现，这个时候会通过mFC中取出需要的参数, 并实例化一个Plugin
     * 这个案例中，参数有scalar和scale两个参数。从mFC中取出来对应的数据来初始化这个paramMap
     */
    try
    {
        float scalar = 1.0f;  // default value
        float scale = 1.0f;   // default value

        // 提取字段数据
        for (int i = 0; i < fc->nbFields; ++i) {
            if (std::string(fc->fields[i].name) == "scalar" && fc->fields[i].type == PluginFieldType::kFLOAT32) {
                scalar = *(static_cast<const float*>(fc->fields[i].data));
            } else if (std::string(fc->fields[i].name) == "scale" && fc->fields[i].type == PluginFieldType::kFLOAT32) {
                scale = *(static_cast<const float*>(fc->fields[i].data));
            }
        }

        return new CustomScalarPlugin(name, scalar, scale);
    }
    catch(const std::exception& e)
    {
        LOGE("ERROR detected when create plugin: %s", e.what());
    }
    return nullptr;

}

IPluginV2* CustomScalarPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    /* 反序列化插件其实就是实例化一个插件，所有插件实现都差不多 */
    return new CustomScalarPlugin(name, serialData, serialLength);
}

void CustomScalarPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    /* 所有插件实现都差不多 */
    mNamespace = pluginNamespace;
    return;
}

const PluginFieldCollection* CustomScalarPluginCreator::getFieldNames() noexcept {
    /* 所有插件实现都差不多 */
    return &mFC;
}

}  // namespace custom
