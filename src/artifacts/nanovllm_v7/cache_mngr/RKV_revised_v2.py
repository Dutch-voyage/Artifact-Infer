import torch
import torch.nn as nn
import torch.nn.functional as F

from src.services.nanovllm_v7.utils.logging import append_item_to_log

from .utils import cal_similarity, compute_attention_scores, update_log
from .lse_preserve_merge import merge_fixed_budget, merge_multi_to_one
from .binary_search import binary_search_T_linear, gradient_descent_T_linear


class RKV:
    def __init__(
        self,
        config,
        budget=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.2,
        retain_direction="last",
        record_kept_token_indices=False,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.sink_size = 1
        self.window_size = window_size - self.sink_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        
        self.lse_preserve_merge = config.lse_preserve_merge
        self.p_attn = config.p_attn
        self.if_log_compress = config.if_log_compress
        

    def update_kv(
        self,
        query_states,
        key_states,
        value_states,
        *args,
    ):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len < self.budget:
            return {
                "key_states": key_states,
                "value_states": value_states,
            }
        else:
            attn_weights = compute_attention_scores(query_states, key_states)

            attn_weights_sum = (
                nn.functional.softmax(
                    attn_weights[:, :, :, self.sink_size : -self.window_size],
                    dim=-1,
                    dtype=torch.float32,
                )
                .mean(dim=-2)
                .to(query_states.dtype)
            )

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
            )[:, self.sink_size : -self.window_size].unsqueeze(0)
            
            shifted_probs = attn_cache * self.mix_lambda + similarity_cos * (
                1 - self.mix_lambda
            )

            # shifted_probs, T = gradient_descent_T_linear(
            #     attn_weights_sum,
            #     shifted_probs,
            #     self.p_attn,
            # )

            attn_cache, T = gradient_descent_T_linear(
                attn_weights_sum,
                shifted_probs,
                self.p_attn,
            )

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
