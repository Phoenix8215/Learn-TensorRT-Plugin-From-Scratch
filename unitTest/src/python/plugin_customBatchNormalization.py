import torch
import torch.onnx
import torch.nn as nn
import onnx
import onnxsim
import os

class CustomBatchNormImpl(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, mean, variance, gamma, beta, epsilon):
        return g.op("custom::BatchNorm", x, mean, variance, gamma, beta, epsilon_f=epsilon)

    @staticmethod
    def forward(ctx, x, mean, variance, gamma, beta, epsilon):
        ctx.save_for_backward(x, mean, variance, gamma, beta)
        ctx.epsilon = epsilon
        mean = mean.view(1, -1, 1, 1)
        variance = variance.view(1, -1, 1, 1)
        gamma = gamma.view(1, -1, 1, 1)
        beta = beta.view(1, -1, 1, 1)
        normalized = (x - mean) / torch.sqrt(variance + epsilon)
        return gamma * normalized + beta

class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # 注册一个缓冲区 (buffer),用来存储不需要学习的参数，它们作为模型的一部分保存和加载。
        # 缓冲区与参数的区别在于，缓冲区不会在优化过程中更新，它们是固定的值，
        # 但仍然是模型的一部分，并且会出现在模型的 state_dict 中
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        return CustomBatchNormImpl.apply(x, self.running_mean, self.running_var, self.gamma, self.beta, self.epsilon)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, (3, 3), padding=1)
        self.bn = CustomBatchNorm(3)
    
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
        x = self.bn(x)
        return x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def export_bn_onnx(input, model):
    current_path = os.path.dirname(__file__)
    file = current_path + "/../../models/onnx/sample_batchnorm.onnx"
    torch.onnx.export(
        model=model, 
        args=(input,),
        f=file,
        input_names=["input0"],
        output_names=["output0"],
        opset_version=15)
    print("Finished BatchNorm onnx export")

    # check the exported onnx model
    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)

    # use onnx-simplifier to simplify the onnx
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
    input = torch.tensor([[[[0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
                            [0.7999, 0.3971, 0.7544, 0.5695, 0.4388],
                            [0.6387, 0.5247, 0.6826, 0.3051, 0.4635],
                            [0.4550, 0.5725, 0.4980, 0.9371, 0.6556],
                            [0.3138, 0.1980, 0.4162, 0.2843, 0.3398]]]], dtype=torch.float32)

    model = Model()
    model.eval() 
    
    # 计算
    eval(input, model)

    # 导出onnx
    export_bn_onnx(input, model)
