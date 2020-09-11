# encoding:utf-8
# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
from models import *
from PIL import Image

####### 所需库及版本信息 ########
# torch 1.6.0
# onnx 1.7.0
# onnxruntime 1.3.0



if __name__ == '__main__':
    model_def = 'config/face_mask.cfg'
    img_size = 416
    regu_mode = 0
    active_mode = 0
    model = Darknet(model_def, img_size=img_size, regu_mode=regu_mode, active_mode=active_mode)
    weights_path = 'checkpoints/yolov3_ckpt_9.pth'
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    # Input to the model
    x = torch.randn(1, 3, 416, 416, requires_grad=True)
    torch_out = model(x)
    print(torch_out)
    import pdb
    pdb.set_trace()
    # convert to onnx
    # Export the model
    torch.onnx.export(model,                 # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "yolov3.onnx",             # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  #dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                  #              'output' : {0 : 'batch_size'}})
                  )
    # 验证模型有效性
    import onnx
    pdb.set_trace()
    onnx_model = onnx.load("yolov3.onnx")
    onnx.checker.check_model(onnx_model)

    # 验证输出有效性
    import onnxruntime

    ort_session = onnxruntime.InferenceSession("yolov3.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
