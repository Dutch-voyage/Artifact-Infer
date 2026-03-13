import torch
import torch.nn as nn
import torch.nn.functional as F

from src.services.nanovllm_v8.utils.logging import append_item_to_log

from .utils import cal_similarity, compute_attention_scores, update_log
from .lse_preserve_merge import merge_fixed_budget, merge_multi_to_one
from .binary_search import count_topp_selected, calculate_mass_differentiable_linear
from flashinfer.sampling import top_p_renorm_probs
from flashinfer.quantization import segment_packbits

num_kv_heads = 8

def gradient_descent_T_linear(raw_probs, shifted_logits, anchor_p, lr=1e-2, max_iters=10):
    bsz, num_heads, kv_cache_len = shifted_logits.shape
    # bsz, num_heads, kv_cache_len = shifted_logits.shape
    shifted_logits = shifted_logits.reshape(-1, kv_cache_len)
    raw_probs = raw_probs.reshape(-1, kv_cache_len)
    T = torch.full((bsz, num_kv_heads), 0.5, device="cuda", dtype=torch.float32, requires_grad=True)
    # T = torch.full((raw_probs.shape[0],), 0.5, device="cuda", dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([T], lr=lr)
    
    num_selected_oracle = count_topp_selected(raw_probs, anchor_p)
    T = T.repeat(1, num_heads // num_kv_heads).reshape(-1)
        
    # print(num_selected_oracle)
        
    for iter in range(max_iters):
        # Use differentiable version for gradient-based optimization
        cu_mass = calculate_mass_differentiable_linear(shifted_logits, T, num_selected_oracle)
        loss = torch.mean((cu_mass - anchor_p) ** 2)
        # loss = torch.mean(torch.abs(cu_mass - anchor_p))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Ensure T stays positive and reasonable
        with torch.no_grad():
            T.clamp_(min=1e-3, max=10.0)
        if torch.abs(cu_mass - anchor_p).max() < 1e-2:
            break
    
    # print(cu_mass.mean().item())
    
    # print(count_topp_selected(F.softmax(shifted_logits, dim=-1), anchor_p))
    
    # T = T.detach().reshape(bsz, num_heads)
    # shifted_probs = shifted_probs.reshape(bsz, num_heads, q_cache_len, kv_cache_len)
    T = T.unsqueeze(-1).repeat(1, shifted_logits.shape[0] // T.shape[0]).reshape(-1)
    
    shifted_logits /= T.unsqueeze(-1)
    shifted_logits -= shifted_logits.amax(dim=-1, keepdim=True)
    shifted_probs = torch.softmax(shifted_logits, dim=-1)
    
    # print(count_topp_selected(F.softmax(shifted_logits, dim=-1), anchor_p))
    # print("-" * 100)
    
    shifted_probs = shifted_probs.reshape(bsz, num_heads, kv_cache_len)
    return shifted_probs, T

class RKV:
    def __init__(
        self,
        config,
        budget=1024,
        upper_budget=2048, 
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.2,
        retain_direction="last",
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        # self.budget = budget
        self.budget = budget
        self.upper_budget = upper_budget
        self.sink_size = 1
        self.window_size = window_size - self.sink_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        
        self.lse_preserve_merge = config.lse_preserve_merge
        self.p_attn = config.p_attn
        self.if_log_compress = config.if_log_compress
        
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

            attn_weights_sum = (
                nn.functional.softmax(
                    attn_weights,
                    dim=-1,
                    dtype=torch.float32,
                )
                .mean(dim=-2)
                .to(query_states.dtype)
            ).detach()

            # TODO: Softmax then reduce head

            # attn_cache = F.max_pool1d(
            #     attn_weights_sum,
            #     kernel_size=self.kernel_size,
            #     padding=self.kernel_size // 2,
            #     stride=1,
            # )

            attn_cache = attn_weights_sum

            similarity_cos = -cal_similarity(
                key_states,
                normalization=True,
                retain_ratio=self.retain_ratio,
                retain_direction=self.retain_direction,
            ).unsqueeze(0).detach()
                        
            shifted_probs = attn_cache * self.mix_lambda + similarity_cos * (
                1 - self.mix_lambda
            )

            # shifted_probs, T = gradient_descent_T_linear(
            #     attn_weights_sum,
            #     shifted_probs,
            #     self.p_attn,
            # )
            
            shifted_probs -= shifted_probs.amin(dim=-1, keepdim=True).detach()
            shifted_probs = shifted_probs / shifted_probs.sum(dim=-1, keepdim=True)
            shift_logits = torch.log(shifted_probs + 1e-10)
    
            if seq_id not in self.temperatures:            
                attn_cache, T = gradient_descent_T_linear(
                    attn_weights_sum,
                    shift_logits,
                    self.p_attn,
                )
                self.temperatures[seq_id] = T
            else:
                T = self.temperatures[seq_id]
                shift_logits = shift_logits.view(-1, kv_cache_len)
                attn_cache = shift_logits / T.unsqueeze(-1)
                attn_cache = attn_cache.view(-1, num_kv_heads, kv_cache_len)
                attn_cache = attn_cache.masked_fill(~(effective_mask.squeeze(-2)), float("-inf"))
                attn_cache = F.softmax(attn_cache, dim=-1)
            
            if self.if_log_compress:
                update_log(
                    attn_cache,
                    key_states,
                    value_states,
                    query_states,
                    self.p_attn,
                    self.sink_size,
                    self.window_size,
                )
                append_item_to_log("temperatures", T.reshape(-1).cpu())
            
            if self.lse_preserve_merge:
                # k_compress, v_compress = merge_fixed_budget(
                #     attn_cache,
                #     attn_weights_sum, 
                #     self.budget - self.window_size - self.sink_size,
                #     key_states[:, :, self.sink_size : -self.window_size, :],
                #     value_states[:, :, self.sink_size : -self.window_size, :],
                # )
                k_compress, v_compress = merge_multi_to_one(
                    attn_cache,
                    attn_weights_sum, 
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
                
                k = min(self.budget - self.window_size - self.sink_size, attn_cache.shape[-1])
                # save the top budget indices
                indices_desc_topk = attn_cache.squeeze(0).topk(k, dim=-1).indices
                
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
