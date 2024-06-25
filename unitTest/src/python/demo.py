import ctypes
import os 
import numpy as np
import tensorrt as trt
import logging

from trt_model import test_logger, console_handler, file_handler
from trt_model import build_network, inference, validation

def getMishPlugin() -> trt.tensorrt.IPluginV2:
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "Mish":
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def MishCPU(inputH):
    return [inputH[0] * np.tanh(np.log1p(np.exp(inputH[0])))]


def customMishTest(shape):
    current_path = os.path.dirname(__file__)
    soFile = current_path + "/../../lib/custom-plugin.so"
    ctypes.cdll.LoadLibrary(soFile)

    plugin = getMishPlugin()
    name = plugin.plugin_type
    trtFile = current_path + "/../../models/engine/%s-Dim%s.engine" % (name, str(len(shape)))
    testCase = "<shape=%s>" % (shape,)

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
    outputCPU = MishCPU(bufferH[:nInput])
    res = validation(nInput, nIO, bufferH, outputCPU)

    if res:
        test_logger.info("Test '%s':%s finish!\n" % (name, testCase))
    else:
        test_logger.error("Test '%s':%s failed!\n" % (name, testCase))
        exit()

# 单元测试入口
def unit_test():
    customMishTest([32])
    customMishTest([32, 32])
    customMishTest([16, 16, 16])
    customMishTest([8, 8, 8, 8])

if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    np.random.seed(1)

    test_logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    test_logger.info("Starting unit test...")
    unit_test()
    test_logger.info("All tests are passed!!")