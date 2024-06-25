import ctypes
import os
import numpy as np
import tensorrt as trt
import logging

from trt_model import test_logger, console_handler, file_handler
from trt_model import build_network, inference, validation

# CPU实现
def CustomBatchNormCPU(inputH, mean, variance, gamma, beta, epsilon):
    normalized = (inputH - mean) / np.sqrt(variance + epsilon)
    return gamma * normalized + beta

# def getCustomBatchNormPlugin(mean, variance, gamma, beta, epsilon) -> trt.tensorrt.IPluginV2:
#     for c in trt.get_plugin_registry().plugin_creator_list:
#         print(f"Found plugin creator: {c.name}")
#         if c.name == "customBatchNormalization":
#             parameterList = []
#             parameterList.append(trt.PluginField("mean", mean.astype(np.float32), trt.PluginFieldType.FLOAT32))
#             parameterList.append(trt.PluginField("variance", variance.astype(np.float32), trt.PluginFieldType.FLOAT32))
#             parameterList.append(trt.PluginField("gamma", gamma.astype(np.float32), trt.PluginFieldType.FLOAT32))
#             parameterList.append(trt.PluginField("beta", beta.astype(np.float32), trt.PluginFieldType.FLOAT32))
#             parameterList.append(trt.PluginField("epsilon", np.array([epsilon], dtype=np.float32), trt.PluginFieldType.FLOAT32))
#             return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
#     return None

def getCustomBatchNormPlugin(mean, variance, scale, bias, epsilon) -> trt.tensorrt.IPluginV2:
    for c in trt.get_plugin_registry().plugin_creator_list:
        print(f"Found plugin creator: {c.name}")
        if c.name == "customBatchNormalization":
            parameterList = []
            parameterList.append(trt.PluginField("mean", mean.astype(np.float32), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("variance", variance.astype(np.float32), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("scale", scale.astype(np.float32), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("bias", bias.astype(np.float32), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("epsilon", np.array([epsilon], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None


def customBatchNormTest(shape, epsilon):
    current_path = os.path.dirname(__file__)
    soFile       = current_path + "/../../lib/custom-plugin.so"
    ctypes.cdll.LoadLibrary(soFile)

    # 生成随机输入和BatchNorm参数
    inputH = np.random.randn(*shape).astype(np.float32)
    mean = np.random.randn(shape[1]).astype(np.float32)
    variance = np.random.rand(shape[1]).astype(np.float32)
    scale = np.random.rand(shape[1]).astype(np.float32)
    bias = np.random.rand(shape[1]).astype(np.float32)

    plugin = getCustomBatchNormPlugin(mean, variance, scale, bias, epsilon)
    if plugin is None:
        test_logger.error("Failed to create customBatchNormalization plugin")
        return

    name = plugin.plugin_type
    trtFile = current_path + "/../../models/engine/%s-Dim%s.engine" % (name, str(len(shape)))
    testCase = "<shape=%s,epsilon=%f>" % (shape, epsilon)

    test_logger.info("Test '%s':%s" % (name, testCase))

    #################################################################
    ################### 从这里开始是builder的部分 ######################
    #################################################################
    engine = build_network(trtFile, shape, plugin)
    if engine is None:
        exit()

    #################################################################
    ################### 从这里开始是infer的部分 ########################
    #################################################################
    nInput, nIO, bufferH = inference(engine, shape)

    #################################################################
    ################# 从这里开始是validation的部分 #####################
    #################################################################
    outputCPU = CustomBatchNormCPU(bufferH[:nInput], mean, variance, scale, bias, epsilon)
    res = validation(nInput, nIO, bufferH, outputCPU)

    if res:
        test_logger.info("Test '%s':%s finish!\n" % (name, testCase))
    else:
        test_logger.error("Test '%s':%s failed!\n" % (name, testCase))
        exit()

def unit_test():
    customBatchNormTest([32, 3], 1e-5)
    customBatchNormTest([32, 32, 32], 1e-5)
    customBatchNormTest([16, 16, 16, 16], 1e-5)

if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    np.random.seed(1)

    test_logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    test_logger.info("Starting unit test...")
    unit_test()
    test_logger.info("All tests are passed!!")
