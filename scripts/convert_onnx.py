import onnx
import torch
from onnxsim import simplify

import internimage

model_name = "internimage_t_1k_224"
args = {"drop_path_rate": 0.1, "core_op": "DCNv3_pytorch"}
image_size = 224

model = internimage.create_model(model_name, pretrained=True, **args)
print(model)
dummy_input = torch.randn(1, 3, image_size, image_size)

torch.onnx.export(
    model,
    dummy_input,
    model_name + ".onnx",
    # verbose=True,
    opset_version=17,
    export_params=True,
    # do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    # dynamic_axes={
    #     "input": {0: "batch_size"},
    #     "output": {0: "batch_size"},
    # },  # 動的軸（dynamic axes）の指定
)
model_org = onnx.load(model_name + ".onnx")
onnx.checker.check_model(model_org)
onnx.shape_inference.infer_shapes(model_org)
model_onnx, check = simplify(model_org)
print("simple check: ", check)
onnx.save(model_onnx, model_name + "_sim" + ".onnx")
