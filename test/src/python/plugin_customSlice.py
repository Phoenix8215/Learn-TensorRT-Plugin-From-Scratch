import torch
import torch.onnx
import torch.nn as nn
import onnx
import onnxsim
import os


class CustomSliceImpl(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, start, end):
        return g.op("Slice", x, axes_i=[1], starts_i=[int(start)], ends_i=[int(end)])

    @staticmethod
    def forward(ctx, x, start, end):
        return x[:, start:end, :, :]

class CustomSlice(nn.Module):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, x):
        return CustomSliceImpl.apply(x, self.start, self.end)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, (3, 3), padding=1)
        self.slice1 = CustomSlice(0, 2)  # First 2 channels
        self.slice2 = CustomSlice(2, 4)  # Next 2 channels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=1.)
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.05)
                nn.init.constant_(m.bias, 0.05)

    def forward(self, x):
        x = self.conv(x)
        slice1 = self.slice1(x)
        slice2 = self.slice2(x)
        return slice1, slice2

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def export_norm_onnx(input, model):
    current_path = os.path.dirname(__file__)
    file = current_path + "/sample_customSlice.onnx"
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0_1", "output0_2"],
        opset_version = 11)
    print("Finished normal onnx export")

    # check the exported onnx model
    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)

    # use onnx-simplifier to simplify the onnx
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, file)

def eval(input, model):
    output1, output2 = model(input)
    print("------from infer------")
    print("Input:")
    print(input)
    print("\nSlice 1:")
    print(output1)
    print("\nSlice 2:")
    print(output2)

if __name__ == "__main__":
    setup_seed(1)
    torch.set_printoptions(precision=4, sci_mode=False)
    input = torch.tensor([[[[0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
                            [0.7999, 0.3971, 0.7544, 0.5695, 0.4388],
                            [0.6387, 0.5247, 0.6826, 0.3051, 0.4635],
                            [0.4550, 0.5725, 0.4980, 0.9371, 0.6556],
                            [0.3138, 0.1980, 0.4162, 0.2843, 0.3398]],

                           [[0.4576, 0.1793, 0.3031, 0.6347, 0.1293],
                            [0.5999, 0.2971, 0.6544, 0.4695, 0.3388],
                            [0.4387, 0.4247, 0.5826, 0.2051, 0.3635],
                            [0.3550, 0.4725, 0.3980, 0.8371, 0.5556],
                            [0.2138, 0.0980, 0.3162, 0.1843, 0.2398]],

                           [[0.1576, 0.0793, 0.2031, 0.5347, 0.2293],
                            [0.3999, 0.1971, 0.5544, 0.3695, 0.2388],
                            [0.2387, 0.3247, 0.4826, 0.1051, 0.2635],
                            [0.2550, 0.3725, 0.2980, 0.7371, 0.4556],
                            [0.1138, 0.1980, 0.2162, 0.0843, 0.1398]],

                           [[0.0576, 0.0793, 0.1031, 0.1347, 0.3293],
                            [0.2999, 0.0971, 0.4544, 0.2695, 0.1388],
                            [0.1387, 0.2247, 0.3826, 0.0051, 0.1635],
                            [0.1550, 0.2725, 0.1980, 0.6371, 0.3556],
                            [0.0138, 0.2980, 0.1162, 0.1843, 0.0398]]]])

    model = Model()
    model.eval() 
    
    # 计算
    eval(input, model)

    # 导出onnx
    export_norm_onnx(input, model);
