import torch
import torch.nn as nn 
import torch.nn.functional as F

import os
from glob import glob
from safetensors import safe_open

from torch.linalg import pinv, lstsq


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

def load_model(model: nn.Module, path: str):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if "layers.0" not in weight_name:
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace("model.layers.0.self_attn.", "").replace(k, v)
                        print(param_name)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    continue

class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.num_heads = self.total_num_heads
        self.num_kv_heads = self.total_num_kv_heads
        input_size = hidden_size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            self.q_weight = loaded_weight.clone()
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
            self.k_weight = loaded_weight.clone()
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            self.v_weight = loaded_weight.clone()
        param_data = param_data.narrow(0, shard_offset, shard_size)
        param_data.copy_(loaded_weight)

class placeholder(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = QKVParallelLinear(
            hidden_size=2560,
            head_size=128,
            total_num_heads=32,
            total_num_kv_heads=8,
            bias=False,
        )
    
    def forward(self):
        pass

def test_inv():
    model = placeholder()
    load_model(model, "/home/yyx/models/Qwen3-4B")
    model = model.to("cuda")
    print(model.qkv_proj.q_weight.shape)
    print(model.qkv_proj.k_weight.shape)
    print(model.qkv_proj.v_weight.shape)

    Kinv_Q, residuals, rank, sigular_values = lstsq(model.qkv_proj.k_weight.float().T, model.qkv_proj.q_weight.float().T)# .solution.T.to("cuda")
    
    print(residuals)
    Kinv_Q = Kinv_Q.T.to("cuda")
    
    x = torch.randn(512, 2560).to("cuda")
    qkv = model.qkv_proj(x)
    
    q_size = 32 * 128
    kv_size = 8 * 128
    q, k, v  = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_approx = F.linear(k, Kinv_Q, None)
    
    print(q)
    print(q_approx)

    assert torch.allclose(q, q_approx, atol=1e-5), "Inverse query failed"
    
    print("test inverse qk passed")

if __name__ == "__main__":
    test_inv()