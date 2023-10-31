import torch
from torch.profiler import ProfilerActivity, profile, record_function

import internimage

torch.manual_seed(3)


class TestInternImage:
    model = internimage.create_model("internimage_t_1k_224", pretrained=False).to("cuda")

    def test_create_model(self):
        model_names = [
            "internimage_t_1k_224",
            "internimage_s_1k_224",
            "internimage_b_1k_224",
            "internimage_l_22kto1k_384",
            "internimage_xl_22kto1k_384",
            "internimage_h_22kto1k_384",
            "internimage_h_22kto1k_640",
            "internimage_g_22kto1k_512",
        ]
        for name in model_names:
            model = internimage.create_model(name, pretrained=False)
            model = internimage.create_model(
                name, pretrained=False, features_only=True, out_indices=[2, 3]
            )
            print("successful create model:", name)

    def test_forward(self):
        inputs = torch.rand((1, 3, 224, 224)).to("cuda")
        out = self.model(inputs)
        print(f"{torch.cuda.memory_reserved() / 1e9} GB")

    def test_fp16(self):
        inputs = torch.rand((1, 3, 224, 224)).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = self.model(inputs)
            print(f"{torch.cuda.memory_reserved() / 1e9} GB")

    def test_profile(self):
        inputs = torch.rand((1, 3, 224, 224)).to("cuda")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
        ) as prof:
            self.model(inputs)
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
