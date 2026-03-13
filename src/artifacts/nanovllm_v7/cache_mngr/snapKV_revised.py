import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import compute_attention_scores, update_log
from .lse_preserve_merge import merge_fixed_budget
from src.services.nanovllm_v7.utils.logging import append_item_to_log
from .binary_search import binary_search_T, gradient_descent_T

class SnapKV:
    def __init__(
        self,
        config, 
        budget=128,
        window_size=8,
        kernel_size=7,
        record_kept_token_indices=False,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.sink_size = 0
        self.window_size = window_size - self.sink_size
        self.kernel_size = kernel_size

        self.lse_preserve_merge = config.lse_preserve_merge
        self.if_log_compress = config.if_log_compress
        self.p_attn = config.p_attn
        self.if_temperatured = config.if_temperatured
        
        # for recording kept token indices
        self.record_kept_token_indices = record_kept_token_indices

    def update_kv(
        self,
        query_states,
        key_states,
        value_states,
        *args, 
    ):
        bsz, num_heads, q_cache_len, head_dim = query_states.shape
        kv_cache_len = key_states.shape[-2]
        num_kv_heads = key_states.shape[1]
        
        if kv_cache_len < self.budget:
            return {
                "key_states": key_states, 
                "value_states": value_states,
            }
        else:
            attn_weights = compute_attention_scores(query_states, key_states)

            raw_attn_weights = attn_weights[:, :, :, self.sink_size : -self.window_size] # .view(-1, kv_cache_len - self.window_size)
            
            def transform(attn_weights):
                transformed_attn = F.max_pool1d(
                    attn_weights,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )
                return transformed_attn

            if self.if_temperatured:
                # attn_weights, T = binary_search_T(raw_attn_weights, self.p_attn, transform)
                attn_weights, T = gradient_descent_T(raw_attn_weights, self.p_attn, transform)
            else:
                raw_attn_weights = raw_attn_weights.view(-1, kv_cache_len - self.window_size - self.sink_size)
                attn_weights = transform(raw_attn_weights)
                attn_weights = attn_weights.view(bsz, num_kv_heads, q_cache_len, -1)
                T = None
            
            attn_cache = (
                nn.functional.softmax(
                    attn_weights,
                    dim=-1,
                    dtype=torch.float32,
                )
                .mean(dim=-2)
                .to(query_states.dtype)
            ) 
                        
            if self.if_log_compress:
                update_log(attn_cache, 
                           key_states, 
                           value_states, 
                           query_states, 
                           self.p_attn, 
                           self.sink_size, 
                           self.window_size)
                # append_item_to_log("temperatures", T)
            
            if self.lse_preserve_merge:
                k_compress, v_compress = merge_fixed_budget(
                    attn_cache,
                    raw_attn_weights.softmax(dim=-1).mean(-2), 
                    self.budget - self.window_size - self.sink_size,
                    key_states[:, :, self.sink_size : -self.window_size, :],
                    value_states[:, :, self.sink_size : -self.window_size, :],
                )
            else:
                indices = attn_cache.topk(
                    self.budget - self.window_size - self.sink_size, dim=-1
                ).indices
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

                k_compress = key_states[:, :, self.sink_size: -self.window_size, :].gather(
                    dim=2, index=indices
                )
                v_compress = value_states[:, :, self.sink_size: -self.window_size, :].gather(
                    dim=2, index=indices
                )
            
            k_sink = key_states[:, :, : self.sink_size, :]
            v_sink = value_states[:, :, : self.sink_size, :]
            k_cur = key_states[:, :, -self.window_size :, :]
            v_cur = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([k_sink, k_compress, k_cur], dim=2)
            value_states = torch.cat([v_sink, v_compress, v_cur], dim=2)
            
            return {"key_states": key_states, "value_states": value_states}