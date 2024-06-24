#include <cstring>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <type_traits>
#include <assert.h>

#include "model.hpp"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "utils.hpp"
#include "cuda_runtime.h"
#include "math.h"

float input_5x5[] = {
    0.7576, 0.2793, 0.4031, 0.7347, 0.0293,
    0.7999, 0.3971, 0.7544, 0.5695, 0.4388,
    0.6387, 0.5247, 0.6826, 0.3051, 0.4635,
    0.4550, 0.5725, 0.4980, 0.9371, 0.6556,
    0.3138, 0.1980, 0.4162, 0.2843, 0.3398};



using namespace std;

class Logger : public nvinfer1::ILogger{
public:
    virtual void log (Severity severity, const char* msg) noexcept override{
        string str;
        switch (severity){
            case Severity::kINTERNAL_ERROR: str = RED    "[fatal]" CLEAR;break; // 编译时自变量可以进行拼接
            case Severity::kERROR:          str = RED    "[error]" CLEAR;break;
            case Severity::kWARNING:        str = BLUE   "[warn]"  CLEAR;break;
            case Severity::kINFO:           str = YELLOW "[info]"  CLEAR;break;
            case Severity::kVERBOSE:        str = PURPLE "[verb]"  CLEAR;break;
        }
        printf("%s %s\n", str.c_str(), msg);
    }   

};

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using make_unique = std::unique_ptr<T, InferDeleter>;

Model::Model(string path, precision prec){
    if (getFileType(path) == ".onnx")
        mOnnxPath = path;
    else if (getFileType(path) == ".weights")
        mWtsPath = path;
    else 
        LOGE("ERROR: %s, wrong weight or model type selected. Program terminated", getFileType(path).c_str());

    if (prec == precision::FP16) {
        mPrecision = nvinfer1::DataType::kHALF;
    } else if (prec == precision::INT8) {
        mPrecision = nvinfer1::DataType::kINT8;
    } else {
        mPrecision = nvinfer1::DataType::kFLOAT;
    }

    mEnginePath = getEnginePath(path, prec);
}



bool Model::build() {
    if (mOnnxPath != "") {
        return build_from_onnx();
    } else {
        // return build_from_weights();
        return false;
    }
}


bool Model::build_from_onnx(){
    if (fileExists(mEnginePath)){
        LOG("%s has been generated!", mEnginePath.c_str());
        return true;
    } else {
        LOG("%s not found. Building engine...", mEnginePath.c_str());
    }
    Logger logger;
    auto builder       = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network       = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config        = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser        = make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    config->setMaxWorkspaceSize(1<<28);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    if (!parser->parseFromFile(mOnnxPath.c_str(), 1)){
        LOGE("ERROR: failed to %s", mOnnxPath.c_str());
        return false;
    }

    if (builder->platformHasFastFp16() && mPrecision == nvinfer1::DataType::kHALF) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    } else if (builder->platformHasFastInt8() && mPrecision == nvinfer1::DataType::kINT8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }

    auto engine        = make_unique<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    auto f = fopen(mEnginePath.c_str(), "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);


    return true;
};

bool Model::infer(){
    /*
        我们在infer需要做的事情
        1. 读取model => 创建runtime, engine, context
        2. 把数据进行host->device传输
        3. 使用context推理
        4. 把数据进行device->host传输
    */

    /* 1. 读取model => 创建runtime, engine, context */
    if (!fileExists(mEnginePath)) {
        LOGE("ERROR: %s not found", mEnginePath.c_str());
        return false;
    }

    vector<unsigned char> modelData;
    modelData = loadFile(mEnginePath);
    
    Logger logger;
    auto runtime     = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine      = make_unique<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelData.data(), modelData.size()));
    auto context     = make_unique<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    auto input_dims   = context->getBindingDimensions(0);
    auto output_dims  = context->getBindingDimensions(1);

    LOG("input dim shape is:  %s", printDims(input_dims).c_str());
    LOG("output dim shape is: %s", printDims(output_dims).c_str());

    /* 2. 创建流 */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* 2. 初始化input，以及在host/device上分配空间 */
    init_data(input_dims, output_dims);

    /* 2. host->device的数据传递*/
    cudaMemcpyAsync(mInputDevice, mInputHost, mInputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

    /* 3. 模型推理, 最后做同步处理 */
    float* bindings[]{mInputDevice, mOutputDevice};
    bool success = context->enqueueV2((void**)bindings, stream, nullptr);

    /* 4. device->host的数据传递 */
    cudaMemcpyAsync(mOutputHost, mOutputDevice, mOutputSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    LOG("input data is:  %s", printTensor(mInputHost, mInputSize / sizeof(float), input_dims).c_str());
    LOG("output data is: %s", printTensor(mOutputHost, mOutputSize / sizeof(float), output_dims).c_str());
    LOG("finished inference");
    return true;
}


void Model::init_data(nvinfer1::Dims input_dims, nvinfer1::Dims output_dims){
    mInputSize  = getDimSize(input_dims) * sizeof(float);
    mOutputSize = getDimSize(output_dims) * sizeof(float);

    cudaMallocHost(&mInputHost, mInputSize);
    cudaMallocHost(&mOutputHost, mOutputSize);

    mInputHost = input_5x5;

    cudaMalloc(&mInputDevice, mInputSize);
    cudaMalloc(&mOutputDevice, mOutputSize);
    
}

