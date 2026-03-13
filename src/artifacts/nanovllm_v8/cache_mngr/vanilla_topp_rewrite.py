import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import compute_attention_scores, update_log, gather_selected_kv
from .lse_preserve_merge import merge_fixed_budget
from src.services.nanovllm_v8.utils.logging import append_item_to_log
from .binary_search import binary_search_T, gradient_descent_T

from flashinfer.sampling import top_p_renorm_probs
from flashinfer.quantization import segment_packbits

class VanillaToppKV:
    def __init__(
        self,
        config, 
        budget=128,
        upper_budget=2048, 
        window_size=8,
        kernel_size=7,
        record_kept_token_indices=False,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.upper_budget = upper_budget
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
                                        
            attn_weights = attn_weights.masked_fill(~effective_mask, float("-inf"))
            
            attn_cache = (
                nn.functional.softmax(
                    attn_weights,
                    dim=-1,
                    dtype=torch.float32,
                )
                # .mean(dim=-2)
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
                    attn_weights.softmax(dim=-1).mean(-2), 
                    self.budget - self.window_size - self.sink_size,
                    key_states[:, :, self.sink_size : -self.window_size, :],
                    value_states[:, :, self.sink_size : -self.window_size, :],
                )
            else:
                attn_topp_normed = top_p_renorm_probs(attn_cache.view(-1, attn_cache.shape[-1]), top_p=self.p_attn)
                       
                unselected_mask_topp = (attn_topp_normed == torch.zeros_like(attn_topp_normed))
                
                unselected_mask_topp = unselected_mask_topp.reshape(num_kv_heads, q_cache_len, -1)
                
                selected_mask_topp = (1 - torch.prod(unselected_mask_topp, dim=-2)) * 2
                
                k = min(self.upper_budget - self.sink_size - self.window_size, attn_cache.shape[-1])
                
                unselected_mask_topk = torch.ones(num_kv_heads, q_cache_len, kv_cache_len, dtype=torch.int32, device=key_states.device)
                
                # save the top budget indices
                indices_desc_topk = attn_cache.squeeze(0).topk(k, dim=-1).indices
                unselected_mask_topk.scatter_(-1, indices_desc_topk, 0)
                
                selected_mask_topk = (1 - torch.prod(unselected_mask_topk)) 

                selected_mask = selected_mask_topk + selected_mask_topp
                
                selected_mask[..., :self.sink_size] = 0
                
                selected_mask.scatter_(-1, window_indices, 0)
                selected_mask.masked_fill_((~effective_mask).squeeze(0).squeeze(1), 0)

                key_sink = key_states[:, :, : self.sink_size, :]
                value_sink = value_states[:, :, : self.sink_size, :]
                
                key_window = torch.gather(
                    key_states,
                    2,
                    window_indices.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, -1, head_dim),
                )
                value_window = torch.gather(
                    value_states,
                    2,
                    window_indices.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, -1, head_dim),
                )
                
                key_states, num_selected = gather_selected_kv(key_states, selected_mask.unsqueeze(0))
                value_states, _ = gather_selected_kv(value_states, selected_mask.unsqueeze(0))
                
                key_states = torch.cat([key_sink, key_window, key_states], dim=2)
                value_states = torch.cat([value_sink, value_window, value_states], dim=2)
                
                num_blocks_head = num_selected.squeeze(0) + self.sink_size + self.window_size
                
                num_blocks_head = torch.minimum(num_blocks_head, torch.ones_like(num_blocks_head) * self.upper_budget)
                
                num_blocks_max_heads = num_blocks_head.max()
                
                key_states = key_states[..., :num_blocks_max_heads, :]
                value_states = value_states[..., :num_blocks_max_heads, :]
                                
                max_blocks = selected_mask.shape[-1]
                block_indices = self.block_indices[:max_blocks].to(selected_mask.device)
                organized_selected_mask = block_indices.unsqueeze(0) < num_blocks_head.unsqueeze(1)
                
                mask_indptr = self.mask_indptr * max_blocks
                packed_selected_mask, _ = segment_packbits(organized_selected_mask.view(-1), mask_indptr, bitorder="little")

                packed_selected_mask = packed_selected_mask.view(8, -1)
                
                key_states = key_states.transpose(1, 2).squeeze(0).contiguous()
                value_states = value_states.transpose(1, 2).squeeze(0).contiguous()
                                        
            return {"key_states": key_states, "value_states": value_states, "packed_selected_mask": packed_selected_mask, "num_blocks_this_layer": num_blocks_max_heads}
