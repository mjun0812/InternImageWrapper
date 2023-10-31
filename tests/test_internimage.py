import time

import torch

import internimage

torch.manual_seed(3)


class TestInternImage:
    model_cpp = internimage.create_model("internimage_t_1k_224", pretrained=False).to("cuda")
    model_torch = internimage.create_model(
        "internimage_t_1k_224", pretrained=False, core_op="DCNv3_pytorch"
    ).to("cuda")

    def test_print(self):
        print(self.model_cpp)
        print(self.model_torch)

    @torch.no_grad()
    def test_forward(self):
        inputs = torch.rand((1, 3, 224, 224)).to("cuda")
        print(inputs.shape)

        output_cpp = self.model_cpp(inputs)
        output_torch = self.model_torch(inputs)

        fwdok = torch.allclose(output_cpp, output_torch, rtol=1e-2, atol=1e-3)
        max_abs_err = (output_cpp - output_torch).abs().max()
        max_rel_err = ((output_cpp - output_torch).abs() / output_torch.abs()).max()
        print(">>> forward float")
        print(
            f"* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
        )

    @torch.no_grad()
    def test_time(self):
        num_iter = 200

        inputs = torch.rand((1, 3, 224, 224)).to("cuda")

        times_cpp = []
        for _ in range(num_iter):
            t = time.time()
            output_cpp = self.model_cpp(inputs)
            torch.cuda.synchronize()
            times_cpp.append(time.time() - t)
        times_cpp = sum(times_cpp[num_iter // 2 :]) / (num_iter // 2)

        times_torch = []
        for _ in range(num_iter):
            t = time.time()
            output_torch = self.model_torch(inputs)
            torch.cuda.synchronize()
            times_torch.append(time.time() - t)
        times_torch = sum(times_torch[num_iter // 2 :]) / (num_iter // 2)

        print(f"forward cpp time: {times_cpp * 1000}ms")
        print(f"forward torch time: {times_torch * 1000}ms")
