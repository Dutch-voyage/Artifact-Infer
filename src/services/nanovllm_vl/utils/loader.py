import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    assert packed_modules_mapping is not None
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for file_tensor_name in f.keys(): # name is the name of dict in safetensors
                name = file_tensor_name
                if "rotary_emb.inv_freq" in name:
                    continue
                for param_name, weight_name, shard_id in packed_modules_mapping:
                    # param_name is the substring in the safetensors' names
                    # weight_name is the substring in the model's defined parameter names
                    if weight_name not in name:
                        continue
                    if (
                        "visual" in name
                        and "up_proj" not in name
                        and "gate_proj" not in name
                    ):
                        continue
                    name = name.replace(weight_name, param_name)
                    param = model.get_parameter(name)
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, f.get_tensor(file_tensor_name), shard_id)
                    break
                else:
                    if "visual" in name:
                        # adapt to VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                    param = model.get_parameter(name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(file_tensor_name))
