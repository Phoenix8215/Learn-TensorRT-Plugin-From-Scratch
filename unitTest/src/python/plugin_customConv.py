import os
import torch
import torch.nn as nn
from torch.onnx.symbolic_helper import _get_tensor_sizes
import onnx
import onnxsim



class DummyIdentityConvOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, weight, kernel_shape, strides, pads, group):
        args = [input, weight]
        kwargs = {
            "kernel_shape_i": kernel_shape,
            "strides_i": strides,
            "pads_i": pads,
            "group_i": group
        }
        output_type = input.type().with_sizes(_get_tensor_sizes(input))
        return g.op("CustomTorchOps::IdentityConv", *args,
                    **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, input, weight, kernel_shape, strides, pads, group):
        return input


class DummyIdentityConv(nn.Module):
    def __init__(self, channels):
        super(DummyIdentityConv, self).__init__()
        self.kernel_shape = (1, 1)
        self.strides = (1, 1)
        self.pads = (0, 0, 0, 0)
        self.group = channels
        self.weight = torch.ones(channels, 1, 1, 1)
        self.weight.requires_grad = False

    def forward(self, x):
        x = DummyIdentityConvOp.apply(x, self.weight, self.kernel_shape,
                                      self.strides, self.pads, self.group)
        return x




class IdentityNeuralNetwork(nn.Module):
    def __init__(self, channels):
        super(IdentityNeuralNetwork, self).__init__()
        self.conv1 = DummyIdentityConv(channels)

    def forward(self, x):
        x = self.conv1(x)
        return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Define the export_slice_onnx function
def export_identity_onnx(input, model):
    current_path = os.path.dirname(__file__)
    file = current_path + "/../../models/onnx/sample_customConv.onnx"
    torch.onnx.export(
        model=model,
        args=(input,),
        f=file,
        input_names=["input0"],
        output_names=["output0"],
        opset_version=15)
    print("Finished normal onnx export")

    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)

    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, file)


def eval(input, model):
    output = model(input)
    print("------from infer------")
    print(input)
    print("\n")
    print(output)

if __name__ == "__main__":
    setup_seed(1)
    input = torch.tensor([[[
        [0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
        [0.7999, 0.3971, 0.7544, 0.5695, 0.4388],
        [0.6387, 0.5247, 0.6826, 0.3051, 0.4635],
        [0.4550, 0.5725, 0.4980, 0.9371, 0.6556],
        [0.3138, 0.1980, 0.4162, 0.2843, 0.3398]]]], dtype=torch.float32)

    model = IdentityNeuralNetwork(input.size(1))
    model.eval()
    

    eval(input, model)


    export_identity_onnx(input, model)
