import numpy as np
import torch
import onnx
import onnxruntime
import os, sys
from model import LeNet

DIR_WTS = './weights'
WTS = 'mnist_net.pt'

DIR_ONNX = './onnx'
ONNX = 'mnist_net.onnx'


def initNet(weight_path):
    net = LeNet()
    net.load_state_dict((torch.load(weight_path)))
    net.eval()
    return net


def pytorch2onnx(net, batch_size, onnx_path):
    dummy_input = torch.randn(batch_size, 1, 28, 28, requires_grad=True)
    torch_out = net(dummy_input)
    print("Exporting ONNX Model ...")
    torch.onnx.export(model=net, args=dummy_input, f=onnx_path, export_params=True,
                      input_names=["input"], output_names=["output"], opset_version=13)
    print("ONNX Model has been exported.")
    return dummy_input, torch_out


def onnx_checker(onnx_path):
    print("ONNX Checking ...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX checking Done.")


def ort_checker(onnx_path, dummy_input, torch_out):
    print("ONNXRuntime chekcing ...")
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name:to_numpy(dummy_input)}
    ort_outputs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(to_numpy(torch_out), ort_outputs[0], rtol=1e-03, atol=1e-06)
    print("Exported model has been tested with ONNXRuntime, and the result is good!")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy


if __name__ == '__main__':
    batch_size = 1
    path = os.path.join(DIR_WTS, WTS)
    onnx_path = os.path.join(DIR_ONNX, ONNX)
    if not os.path.exists(path):
        print("Please have the LeNet weights in defined directory.")
        sys.exit()
    if not os.path.exists(DIR_ONNX):
        os.makedirs(DIR_ONNX)
    net = initNet(path)
    dummy_input, torch_out = pytorch2onnx(net, batch_size, onnx_path)
    onnx_checker(onnx_path)
    ort_checker(onnx_path, dummy_input, torch_out)