import torch
import torch.nn as nn
import torch.nn.functional as F

from src.services.nanovllm_v8.utils.logging import append_item_to_log
from .utils import compute_attention_scores, update_log

class NoCompress:
    def __init__(
        self,
        config, 
        budget=128,
        window_size=8,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.sink_size = 1
        self.window_size = window_size - self.sink_size
        
        
        self.if_log_compress = config.if_log_compress
        self.p_attn =  config.p_attn

    def update_kv(
        self,
        query_states,
        key_states,
        value_states,
        *args, 
    ):
        head_dim = query_states.shape[-1]
        q_cache_len = query_states.shape[-2]    
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len < self.budget:
            return {
                "key_states": key_states, 
                "value_states": value_states,
            }
        else:
            attn_weights = compute_attention_scores(query_states, key_states)            
            attn_cache = (
                nn.functional.softmax(
                    attn_weights[:, :, :, self.sink_size: -self.window_size],
                    dim=-1,
                    dtype=torch.float32,
                )
                .mean(dim=-2)
                .to(query_states.dtype)
            )
            # [bsz, num_kv_heads, q_cache_len, kv_cache_len - sink_size - window_size]
            if self.if_log_compress:
                update_log(
                    attn_cache, 
                    key_states, 
                    value_states, 
                    query_states, 
                    self.p_attn, 
                    self.sink_size, 
                    self.window_size
                )
                # T [num_kv_heads, q_cache_len]
                # append_item_to_log("temperatures", T.reshape(-1).cpu())
            
            return {"key_states": key_states, 
                    "value_states": value_states, 
                    }