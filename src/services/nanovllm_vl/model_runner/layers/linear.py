import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 0)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias)
        
    def pad_or_narrow_weight(self, loaded_weight: torch.Tensor, input_dim: int, start_idx: int, shard_size: int) -> torch.Tensor:
        # Padding with zeros for special case such as qwen2_5_VL's mlp which is not 8-aligned
        valid_size = max(loaded_weight.shape[input_dim] - start_idx, 0)

        if valid_size > 0:
            loaded_slice = loaded_weight.narrow(input_dim, start_idx, valid_size)
            pad_shape = list(loaded_weight.shape)
            pad_shape[input_dim] = shard_size - valid_size
            pad = torch.zeros(
                pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
            )
            return torch.cat([loaded_slice, pad], dim=input_dim)

        # All padding
        pad_shape = list(loaded_weight.shape)
        pad_shape[input_dim] = shard_size
        return torch.zeros(
            pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
        )
    

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size        
        start_idx = self.tp_rank * shard_size
        
        # TODO look into how the merged ColumnParallelLinear is sharded, this part is for Qwen2_5 VL only, where the intermediate dim is not 8 aligned
        end_idx = start_idx + shard_size
        if end_idx > loaded_weight.size(self.tp_dim):
            loaded_weight = self.pad_or_narrow_weight(
                loaded_weight, self.tp_dim, start_idx, shard_size
            )
        else:
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)  
        
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        
        param_data.copy_(loaded_weight)


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
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str=None):
        param_data = param.data
        if loaded_shard_id is None:
            # check if this is correct
            ColumnParallelLinear.weight_loader(self, param, loaded_weight)
            return
        
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, -1)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        shard_size = get_actual_shard_size(
            shard_size,
            start_idx,
            loaded_weight.size(self.tp_dim),
        )
        
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.narrow(self.tp_dim, shard_size, param_data.size(self.tp_dim) - shard_size).zero_()
        param_data = param_data.narrow(self.tp_dim, 0, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y

def get_actual_shard_size(shard_size, weight_start, weight_end):
    if weight_end < weight_start:
        return 0

    return min(shard_size, weight_end - weight_start)