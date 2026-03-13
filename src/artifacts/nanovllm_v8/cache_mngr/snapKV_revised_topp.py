import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import compute_attention_scores, update_log, gather_selected_kv
from .lse_preserve_merge import merge_fixed_budget
from src.services.nanovllm_v8.utils.logging import append_item_to_log
from .binary_search import binary_search_T, gradient_descent_T

from flashinfer.sampling import top_p_renorm_probs
from flashinfer.quantization import segment_packbits

class SnapKV:
    def __init__(
        self,
        config, 
        budget=128,
        window_size=8,
        kernel_size=7,
        *args, 
        **kwargs, 
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.sink_size = 1
        self.window_size = window_size - self.sink_size
        self.window_size = 32
        
        self.kernel_size = kernel_size

        self.lse_preserve_merge = config.lse_preserve_merge
        self.if_log_compress = config.if_log_compress
        self.p_attn = config.p_attn
        
        self.num_kv_heads = config.hf_config.num_key_value_heads
        
        self.mask_indptr = torch.arange(0, self.num_kv_heads + 1).to("cuda")
        self.block_indices = torch.arange(0, config.max_model_len).to("cuda")
        
        self.temperatures = {}
        
    def update_kv(
        self,
        query_states,
        key_states,
        value_states,
        effective_kv_head_lens=None,
        effective_mask=None, 
        seq_id=None, 
        *args, 
        **kwargs,
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
            
            if effective_kv_head_lens is not None:
                
                indices = torch.arange(
                    kv_cache_len, device=key_states.device
                ).view(1, 1, -1)
                
                lengths = effective_kv_head_lens.unsqueeze(-1)
                
                effective_mask = indices < lengths.to(indices.device)

                attn_weights = attn_weights.masked_fill(~effective_mask.unsqueeze(2), float("-inf"))
                
            if effective_mask is not None:
                effective_mask = effective_mask.unsqueeze(0).unsqueeze(2).to(key_states.device)
                attn_weights = attn_weights.masked_fill(~effective_mask, float("-inf"))
                
                indices = torch.arange(kv_cache_len, device=key_states.device).unsqueeze(0).expand(bsz, -1)
                
                masked_indices = indices.masked_fill(~effective_mask, -1)
                
                _, window_indices = torch.topk(masked_indices, k=self.window_size, dim=-1)
                
                window_indices = window_indices.squeeze(0).squeeze(1)


            raw_attn_weights = attn_weights
            
            def transform(attn_weights):
                transformed_attn = F.max_pool1d(
                    attn_weights,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )
                return transformed_attn

            if seq_id not in self.temperatures.keys():  
                attn_weights, T = gradient_descent_T(
                    raw_attn_weights,
                    self.p_attn,
                    transform,
                )
                self.temperatures[seq_id] = T
            else:
                T = self.temperatures[seq_id]
                raw_logits = raw_attn_weights.reshape(-1, raw_attn_weights.shape[-1])
                raw_logits /= T.unsqueeze(-1)
                attn_weights = transform(raw_logits).view(bsz, num_kv_heads, q_cache_len, -1)
            
            attn_weights = attn_weights.masked_fill(~effective_mask, float("-inf"))
            
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
                append_item_to_log("temperatures", T)
            
            if self.lse_preserve_merge:
                k_compress, v_compress = merge_fixed_budget(
                    attn_cache,
                    raw_attn_weights.softmax(dim=-1).mean(-2), 
                    self.budget - self.window_size - self.sink_size,
                    key_states[:, :, self.sink_size : -self.window_size, :],
                    value_states[:, :, self.sink_size : -self.window_size, :],
                )
            else:
                selected_mask_full = torch.zeros(num_kv_heads, kv_cache_len, dtype=torch.bool, device=key_states.device)
                
                attn_topp_normed = top_p_renorm_probs(attn_cache.view(-1, attn_cache.shape[-1]), top_p=self.p_attn)
                
                unselected_mask = (attn_topp_normed == torch.zeros_like(attn_topp_normed))
                
                unselected_mask = unselected_mask.reshape(num_kv_heads, -1)
                
                selected_mask = ~unselected_mask
                
                selected_mask_full = selected_mask
                
                # save the top budget indices
                indices_desc_topk = attn_cache.squeeze(0).topk(self.budget - self.window_size - self.sink_size, dim=-1).indices
                selected_mask_full.scatter_(-1, indices_desc_topk, True)
                
                selected_mask_full[..., :self.sink_size] = True
                selected_mask_full[..., -self.window_size:] = True
                
                mask_indptr = torch.arange(0, num_kv_heads + 1).to(selected_mask.device) * kv_cache_len
                
                packed_selected_mask, mask_indptr_new = segment_packbits(selected_mask_full.view(-1), mask_indptr, bitorder="little")
                
                packed_selected_mask = packed_selected_mask.view(8, -1)
            
            # k_sink = key_states[:, :, : self.sink_size, :]
            # v_sink = value_states[:, :, : self.sink_size, :]
            # k_cur = key_states[:, :, -self.window_size :, :]
            # v_cur = value_states[:, :, -self.window_size :, :]
            # key_states = torch.cat([k_sink, k_compress, k_cur], dim=2)
            # value_states = torch.cat([v_sink, v_compress, v_cur], dim=2)
            
            return {"key_states": key_states, "value_states": value_states, "packed_selected_mask": packed_selected_mask}

