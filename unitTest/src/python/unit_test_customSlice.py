import ctypes
import os 
import numpy as np
import tensorrt as trt
import logging

from trt_model import test_logger, console_handler, file_handler
from trt_model import build_network, inference, validation

# CPU 实现
def CustomSliceCPU(inputH, start, size):
    return [inputH[0][:, start:start + size, :, :]]

def getCustomSlicePlugin(start, size) -> trt.tensorrt.IPluginV2:
    for c in trt.get_plugin_registry().plugin_creator_list:
        print(c.name)
        if c.name == "customSlice":
            parameterList = []
            parameterList.append(trt.PluginField("start", np.int32(start), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("size", np.int32(size), trt.PluginFieldType.INT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def customSliceTest(shape, start, size):
    current_path = os.path.dirname(__file__)
    soFile       = current_path + "/../../lib/custom-plugin.so"
    trtFile      = current_path + "/../../models/engine/model-Dim%s.engine" % str(len(shape))
    testCase     = "<shape=%s,start=%d,size=%d>" % (shape, start, size)

    ctypes.cdll.LoadLibrary(soFile)
    plugin = getCustomSlicePlugin(start, size)
    test_logger.info("Test '%s':%s" % (plugin.plugin_type, testCase))

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
    outputCPU = CustomSliceCPU(bufferH[:nInput], start, size)
    res = validation(nInput, nIO, bufferH, outputCPU)

    if res:
        test_logger.info("Test '%s':%s finish!\n" % (plugin.plugin_type, testCase))
    else:
        test_logger.error("Test '%s':%s failed!\n" % (plugin.plugin_type, testCase))
        exit()

def unit_test():
    customSliceTest([4, 3, 32, 32], 1, 1)
    # customSliceTest([8, 3, 32, 32], 0, 2)
    # customSliceTest([1, 3, 32, 32], 1, 2)
    # customSliceTest([1, 3, 32, 32], 2, 1)

if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    np.random.seed(1)

    test_logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    test_logger.info("Starting unit test...")
    unit_test()
    test_logger.info("All tests are passed!!")
