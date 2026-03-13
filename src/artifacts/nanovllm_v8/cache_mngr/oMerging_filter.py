import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, einsum

import math


class OrthMerging:
    def __init__(
        self,
        budget=128,
        window_size=8,
        num_layers=32,
        num_heads=8,
        max_seq_len=8192,
        steps_between_compression=128,
    ):
        self.filter_topk = steps_between_compression // 2
        self.budget = budget
        self.window_size = window_size
        self.steps_between_compression = steps_between_compression
        self.cache_pos_to_seq_pos = torch.zeros(max_seq_len, dtype=torch.int16).expand(
            (num_layers, num_heads, budget + steps_between_compression, max_seq_len)
        ).clone()
        self.cache_pos_to_seq_pos[..., :budget, :budget] = torch.eye(budget)
        self.layer_to_filtered_mask = {}

    def initial_indices(self, seq, layer_id):
        self.cache_pos_to_seq_pos[
            layer_id, :,
            -self.steps_between_compression :,
            seq.num_tokens - self.steps_between_compression : seq.num_tokens,
        ] = torch.eye(self.steps_between_compression)
    
    def reset_indices(self):
        self.cache_pos_to_seq_pos.fill_(0)
        self.cache_pos_to_seq_pos[..., :self.budget, :self.budget] = torch.eye(self.budget)
        self.layer_to_filtered_mask = {}
    
    def update_kv(
        self,
        query_states,
        key_states,
        value_states,
        seq,
        layer_id,
    ):
        # if seq_id not in self.seq_to_per_layer_buffer:
        #     self.seq_to_per_layer_buffer[seq_id] = {
        #         layer_id: []
        #         for layer_id in range(self.num_layers)
        #     }

        bsz, num_q_heads, q_len, head_dim = query_states.shape

        _, num_kv_heads, kv_len, _ = key_states.shape

        if kv_len < self.budget:
            return {
                "key_states": key_states, 
                "value_states": value_states,
            }
        
        topk = kv_len - self.budget
        assert topk * 2 <= kv_len, "budget too small compared to current kv length"

        query_states = rearrange(
            query_states, "b (h g) l d -> b g h l d", g=num_q_heads // num_kv_heads
        )

        # apply mean pooling over query groups
        score = torch.einsum(
            "b g h l d, b h m d -> b g h l m",
            query_states,
            key_states,
        ).mean(dim=1) / math.sqrt(head_dim)

        score = torch.exp(score - score.amax(dim=-1, keepdim=True))

        # option 1: compute kv_len * kv_len matrix, here orthogonal means the the mininum index in this matrix,

        # option 2: select the maximum and minimum score after pooling on q_len .
        score = score.mean(dim=-2)  # shape: (bsz, num_kv_heads, kv_len)
        unselected_mask = torch.ones(
            (bsz, num_kv_heads, kv_len), device=key_states.device, dtype=torch.bool
        )

        filter_mask = self.layer_to_filtered_mask.get(layer_id, None)
        if filter_mask is None:
            filter_mask = torch.zeros(
                (bsz, num_kv_heads, kv_len), device=key_states.device, dtype=torch.bool
            )
            # self.layer_to_filtered_mask[layer_id] = filter_mask
        filter_mask = filter_mask[:, :, :kv_len]
        max_indices = (torch.where(~filter_mask, score, -float("inf"))).topk(
            topk, dim=-1
        ).indices  # shape: (bsz, num_kv_heads, topk)
        unselected_mask.scatter_(-1, max_indices, 0)
        min_indices = (
            (torch.where(unselected_mask & (~filter_mask), -score, -float("inf")))
            .topk(topk, dim=-1)
            .indices
        )  #
        unselected_mask.scatter_(-1, min_indices, 0)

        # no intersection between max_indices and min_indices
        assert (max_indices.unsqueeze(-1) != min_indices.unsqueeze(-2)).all()

        unselected_indices = unselected_mask.nonzero(as_tuple=True)

        k_kept = key_states[unselected_indices].view(
            bsz, num_kv_heads, kv_len - 2 * topk, head_dim
        )
        v_kept = value_states[unselected_indices].view(
            bsz, num_kv_heads, kv_len - 2 * topk, head_dim
        )

        score_max = torch.gather(score, -1, max_indices).unsqueeze(-1)
        score_min = torch.gather(score, -1, min_indices).unsqueeze(-1)

        k_merged = (score_max / score_max + score_min) * torch.gather(
            key_states, 2, max_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        ) + (score_min / score_max + score_min) * torch.gather(
            key_states, 2, min_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )

        # score_merged = torch.einsum(
        #     "b g h l d, b h m d -> b g h l m",
        #     query_states,
        #     k_merged,
        # ).mean(dim=1) / math.sqrt(head_dim)

        # score_merged = torch.exp(score_merged - score_merged.amax(dim=2, keepdim=True)).mean(dim=-2).unsqueeze(-1)

        v_merged = (score_max / score_max + score_min) * torch.gather(
            value_states, 2, max_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        ) + (score_min / score_max + score_min) * torch.gather(
            value_states, 2, min_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )

        k_final = torch.cat([k_kept, k_merged], dim=2)
        v_final = torch.cat([v_kept, v_merged], dim=2)

        self.initial_indices(seq, layer_id)
        new_cache_pos_to_seq_pos = torch.zeros_like(
            self.cache_pos_to_seq_pos[layer_id]
        )
        new_cache_pos_to_seq_pos[:, : k_kept.shape[2], :] = (
            self.cache_pos_to_seq_pos[unselected_indices].view(num_kv_heads, kv_len - 2 * topk, -1)
        )
        new_cache_pos_to_seq_pos[
            :, k_kept.shape[2] : k_final.shape[2], :
        ] = torch.gather(
            self.cache_pos_to_seq_pos,
            -2,
            max_indices.unsqueeze(-1).expand(
                -1, -1, -1, self.cache_pos_to_seq_pos.shape[-1]
            ),
        ) + torch.gather(
            self.cache_pos_to_seq_pos,
            -2,
            min_indices.unsqueeze(-1).expand(
                -1, -1, -1, self.cache_pos_to_seq_pos.shape[-1]
            ),
        )
        self.cache_pos_to_seq_pos[layer_id] = new_cache_pos_to_seq_pos
        
        maxindices_seq_pos_per_cache = score.topk(self.filter_topk, dim=-1).indices
        filter_mask = torch.zeros(
            (bsz, num_kv_heads, self.budget + self.steps_between_compression), device=key_states.device, dtype=torch.bool
        )
        filter_mask.scatter_(-1, maxindices_seq_pos_per_cache, 1)
        self.layer_to_filtered_mask[layer_id] = filter_mask

        return {"key_states": k_final, 
                "value_states": v_final, 
                }
