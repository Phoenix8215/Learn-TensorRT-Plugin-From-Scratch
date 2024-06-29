## still work in process
This repository provides a step-by-step guide on how to write TensorRT plugins and conduct unit tests to ensure that the program's output aligns with our expectations.🙈

This repository will be regularly updated and contains the following custom plugins:
- Scalar
- Slice
- LeakyRELU
- Mish

The test directory contains unit tests for various plugins, while the src folder mainly demonstrates parsing ONNX files with custom operators during model inference, allowing them to be compiled by TensorRT for accelerated execution on the GPU.🤧