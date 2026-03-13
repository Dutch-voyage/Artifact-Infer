from copy import copy
from enum import Enum, auto
from itertools import count
import itertools
from src.services.nanovllm_v1 import sampling_params
import torch

from ..sampling_params import SamplingParams

def torch_rotl_uint8(x: torch.Tensor, k: int) -> torch.Tensor:
    assert x.device.type == "cpu" 
    assert x.dtype is torch.uint8
    return ((x << k) | (x >> (8 - k))).to(torch.uint8)


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class BlockTable:
    num_layers: int = 36
    num_kv_heads: int = 8
    layer_head_to_table: dict[int, list[int]] = {}
    
    def init_block_table(self, num_layers: int, num_kv_heads: int):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        for layer_id in range(num_layers):
            for head_id in range(num_kv_heads):
                self.layer_head_to_table[layer_id * num_kv_heads + head_id] = []
    
    
class Sequence:
    query_window_size = 128
    block_size = 1
    num_kv_heads = 8
    num_layers = 36
    counter = count()
    cuda_graph_counter = count()
    
    def __init__(self):
        self.block_table: list[int] = []
        self.headwise_mask_layer_transpose: torch.Tensor = torch.zeros((self.num_layers, self.num_kv_heads, 1), device="cuda", dtype=torch.uint8)
        self.query_block_id: int = -1
        self.num_tokens: int = 0
        self.last_block_num_tokens: int = 1
        self.num_blocks_head: torch.Tensor = torch.zeros((self.num_kv_heads, ), device="cuda", dtype=torch.int32)
        self.num_prompt_tokens: int = 0
        self.num_cached_tokens: int = 0
        self.next_mask = torch.ones((self.num_kv_heads,), device="cpu", dtype=torch.uint8)
        self.count_to_block_id = {i: [] for i in range(self.num_kv_heads)}
        self.block_id_to_count = {}
    
    @classmethod
    def for_capture(cls, block_table: list[int]):
        seq = cls()
        seq.seq_id = next(Sequence.cuda_graph_counter)
        seq.block_table = block_table
        seq.headwise_mask_layer_transpose = torch.ones((cls.num_layers, cls.num_kv_heads, len(block_table)), device="cuda", dtype=torch.uint8)        
            
        seq.num_tokens = len(block_table) * cls.block_size
        seq.num_blocks_head = torch.ones((seq.num_kv_heads,), device="cuda", dtype=torch.int32) * len(block_table)
        return seq
    
    @classmethod
    def from_prompt(cls, token_ids: list[int], sampling_params = SamplingParams(), kvcache_block_size = 1, query_window_size = 128):
        seq = cls()
        seq.block_size = kvcache_block_size
        seq.query_window_size = query_window_size
        seq.seq_id = next(Sequence.counter)
        seq.status = SequenceStatus.WAITING
        seq.token_ids = copy(token_ids)
        seq.logits = []
        seq.last_token = token_ids[-1]
        seq.num_tokens = len(seq.token_ids)
        # seq.num_blocks = (seq.num_tokens + seq.block_size - 1) // seq.block_size
        # NOTE the block size is always 1 here 
        seq.num_blocks_head = torch.ones((seq.num_kv_heads,), device="cuda", dtype=torch.int32) * seq.num_tokens
        seq.num_prompt_tokens = len(token_ids)
        seq.num_cached_tokens = 0
        
        # seq.next_mask = torch_rotl_uint8(0b00000001, seq.num_tokens)
        seq.next_mask = torch.ones((cls.num_kv_heads,), device="cpu", dtype=torch.uint8)
    
        
        seq.block_table = []
        seq.headwise_mask_layer_transpose = torch.zeros((cls.num_layers, cls.num_kv_heads, 1), device="cuda", dtype=torch.uint8)
        seq.temperature = sampling_params.temperature
        seq.top_k = sampling_params.top_k
        seq.top_p = sampling_params.top_p
        seq.min_p = sampling_params.min_p
        seq.max_tokens = sampling_params.max_tokens
        seq.ignore_eos = sampling_params.ignore_eos
    
        return seq

    def copy_(self, seq):
        self.block_size = seq.block_size
        self.query_window_size = seq.query_window_size
        self.seq_id = seq.seq_id
        self.status = seq.status
        self.token_ids = copy(seq.token_ids)
        self.logits = copy(seq.logits)
        self.last_token = seq.token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(seq.token_ids)
        self.num_cached_tokens = 0
        
        self.block_table = copy(seq.block_table)
        self.temperature = seq.temperature
        self.top_k = seq.top_k
        self.top_p = seq.top_p
        self.min_p = seq.min_p
        self.max_tokens = seq.max_tokens
        self.ignore_eos = seq.ignore_eos
    
    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_blocks_max_heads(self):
        return int(torch.max(self.num_blocks_head).item())

    @property
    def query_window_num_tokens(self):
        return min(self.query_window_size, self.num_tokens)

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    # @property
    # def num_blocks(self):
    #     return (self.num_tokens + self.block_size - 1) // self.block_size

    # @property
    # def last_block_num_tokens(self):
    #     return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # what is this interface for? 
    def block(self, i):
        assert -1 <= i < self.num_tokens
        if i == -1:
            return self.token_ids[-self.block_size:]
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
    
    def append_logits(self, logits: list[float]):
        self.logits.extend(logits)

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
