import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    log_path: str = "./no_compress_logs"
    max_num_batched_tokens: int = 262144
    max_num_seqs: int = 128
    lazy_max_num_seqs: int = -1
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 1
    query_window_size: int = 128
    layer_budget: int = 128 + 1024
    num_kvcache_blocks: int = -1
    if_compress_kvcache: bool = False
    compress_method: str = "oMerge"
    if_log_lse: bool = False
    if_log_num_topp: bool = False
    return_logits:  bool = False
    p_attn: float = 0.99

    if_fake_compress: bool = False
    steps_between_cache_compressions: int = 128

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
