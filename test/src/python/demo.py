import ctypes
import os 
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
test_logger = logging.getLogger("IdentityConvTest")

def getCustomIdentityConvPlugin(channels, kernel_size, stride, padding, group) -> trt.tensorrt.IPluginV2:
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "customIdentityConv":
            parameterList = []
            parameterList.append(trt.PluginField("channels", np.int32(channels), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("kernel_size", np.int32(kernel_size), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("stride", np.int32(stride), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("padding", np.int32(padding), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("group", np.int32(group), trt.PluginFieldType.INT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def build_engine(onnx_file_path, plugin):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30

    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as model:
        parser.parse(model.read())

    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    network.add_plugin_v2([input_tensor], plugin)
    engine = builder.build_engine(network, config)
    
    return engine

def infer(engine, input_data):
    context = engine.create_execution_context()
    input_shape = (1, 3, 480, 960)
    output_shape = (1, 3, 480, 960)
    
    # 分配 GPU 内存
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(input_data.nbytes)
    bindings = [int(d_input), int(d_output)]
    
    # 创建流
    stream = cuda.Stream()
    
    # 将输入数据拷贝到 GPU
    cuda.memcpy_htod_async(d_input, input_data, stream)
    
    # 执行推理
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    # 将输出数据拷贝回 CPU
    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    
    # 同步流
    stream.synchronize()
    
    return output_data

if __name__ == "__main__":
    onnx_file_path = "identity_neural_network.onnx"
    
    # 定义输入数据
    input_data = np.random.rand(1, 3, 480, 960).astype(np.float32)
    
    # 加载插件库
    ctypes.cdll.LoadLibrary("path_to_your_plugin.so")
    
    # 获取插件实例
    plugin = getCustomIdentityConvPlugin(channels=3, kernel_size=1, stride=1, padding=0, group=3)
    
    # 构建 TensorRT 引擎
    engine = build_engine(onnx_file_path, plugin)
    
    # 进行推理
    output_data = infer(engine, input_data)
    
    test_logger.info(f"Input shape: {input_data.shape}")
    test_logger.info(f"Output shape: {output_data.shape}")
    test_logger.info(f"Output data: {output_data}")
