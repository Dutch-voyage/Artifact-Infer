from dataclasses import dataclass


TOPK_ALL = 1 << 30

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = TOPK_ALL, 
    top_p: float = 1.0
    min_p: float = 0.0