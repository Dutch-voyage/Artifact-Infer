import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    log_path: str = "./no_compress_logs"
    max_num_batched_tokens: int = 262144
    max_num_seqs: int = 64
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.80
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 1
    query_window_size: int = 128
    layer_budget: int = 128 + 1024
    layer_upper_budget: int = 2048
    lower_bound_budget: int = 128
    num_kvcache_blocks: int = -1
    
    compress_method: str = "oMerge"
    
    if_compress_kvcache: bool = False
    if_fake_compress: bool = False
    if_log_compress: bool = False
    if_log_lse_in_attn: bool = False
    if_log_num_topp: bool = False
    lse_preserve_merge: bool = False # merge only take effect when "steps between_cache_compressions" > "query_window_size"
    
    return_logits:  bool = False
    p_attn: float = 0.90
    
    attn_reduce_method: str = "raw"

    steps_between_cache_compressions: int = 128

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
    
    def to_dict(self):
        return self.__dict__
