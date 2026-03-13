import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, einsum

import math

def generate_orthogonal_indices(corr_score, topk):
    """
    Optimized GPU version that finds orthogonal indices in parallel.
    Uses a greedy approach with batched operations for better GPU utilization.
    """
    kv_len, _ = corr_score.shape
    device = corr_score.device

    # Create a mask for used indices
    used_mask = torch.zeros(kv_len, dtype=torch.bool, device=device)

    # Initialize result tensors
    i_indices = torch.zeros(topk, dtype=torch.long, device=device)
    j_indices = torch.zeros(topk, dtype=torch.long, device=device)

    # Work with a copy of the correlation matrix to avoid modifying original
    corr_work = corr_score.clone()

    # Set diagonal to infinity to avoid selecting same indices
    corr_work.fill_diagonal_(float('inf'))

    for step in range(topk):
        # Apply used mask by setting used indices to infinity
        corr_work[used_mask, :] = float('inf')
        corr_work[:, used_mask] = float('inf')

        # Find the minimum value and its indices in parallel
        # Get the flattened index of minimum value
        flat_idx = torch.argmin(corr_work)

        # Convert flat index to 2D indices
        min_i = flat_idx // kv_len
        min_j = flat_idx % kv_len

        # Store the indices
        i_indices[step] = min_i
        j_indices[step] = min_j

        # Mark these indices as used
        used_mask[min_i] = True
        used_mask[min_j] = True

    return i_indices, j_indices


def generate_orthogonal_indices_batch(corr_scores, topk):
    """
    Batch version for processing multiple heads simultaneously.
    corr_scores: (num_heads, kv_len, kv_len)
    Returns: (num_heads, topk), (num_heads, topk)
    """
    num_heads, kv_len, _ = corr_scores.shape
    device = corr_scores.device

    # Create batch masks for used indices
    used_masks = torch.zeros(num_heads, kv_len, dtype=torch.bool, device=device)

    # Initialize result tensors
    i_indices_batch = torch.zeros(num_heads, topk, dtype=torch.long, device=device)
    j_indices_batch = torch.zeros(num_heads, topk, dtype=torch.long, device=device)

    # Work with a copy of the correlation matrix
    corr_work = corr_scores.clone()

    # Set diagonals to infinity for all heads
    diag_mask = torch.eye(kv_len, dtype=torch.bool, device=device).unsqueeze(0).expand(num_heads, -1, -1)
    corr_work[diag_mask] = float('inf')

    for step in range(topk):
        # Apply used masks
        # Expand used_masks for broadcasting
        used_masks_expanded = used_masks.unsqueeze(2) | used_masks.unsqueeze(1)
        corr_work[used_masks_expanded] = float('inf')

        # Find minimum values and indices for each head
        flat_indices = torch.argmin(corr_work.view(num_heads, -1), dim=1)

        # Convert flat indices to 2D
        min_i = flat_indices // kv_len
        min_j = flat_indices % kv_len

        # Store the indices
        i_indices_batch[:, step] = min_i
        j_indices_batch[:, step] = min_j

        # Update used masks
        used_masks.scatter_(1, min_i.unsqueeze(1), True)
        used_masks.scatter_(1, min_j.unsqueeze(1), True)
        
        # used_masks[torch.arange(num_heads, device=device), min_i] = True
        # used_masks[torch.arange(num_heads, device=device), min_j] = True

    return i_indices_batch, j_indices_batch
    
    

class OrthMerging:
    def __init__(
        self,
        budget=128,
        window_size=8,
    ):
        self.budget = budget
        self.window_size = window_size
        self.seq_to_per_layer_buffer = {}

    def update_kv(
        self,
        query_states,
        key_states,
        value_states,
        # seq_id,
        # layer_id,
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
        
        query_states = rearrange(
            query_states, "b (h g) l d -> b g h l d", g=num_q_heads // num_kv_heads
        )

        # apply mean pooling over query groups
        score = torch.einsum(
            "b g h l d, b h m d -> b g h l m",
            query_states,
            key_states,
        ).mean(dim=1) / math.sqrt(head_dim)

        score_exp = torch.exp(
            score - score.amax(dim=-1, keepdim=True)
        )  # shape: (bsz, num_kv_heads, q_len, kv_len)

        # option 1: compute kv_len * kv_len matrix, here orthogonal means the the mininum index in this matrix,

        corr_score = score_exp.transpose(
            -2, -1
        ) @ (1.0 / score_exp) # shape: (bsz, num_kv_heads, kv_len, kv_len)
        # get the i, j indices of the topk minimum values in the (-1, -2) dims of the score matrix
        assert bsz == 1

        # Use batch version for better GPU utilization
        i_indices, j_indices = generate_orthogonal_indices_batch(corr_score[0], topk)
        i_indices = i_indices.unsqueeze(0)  # shape: (1, num_kv_heads, topk)
        j_indices = j_indices.unsqueeze(0)
        
        unselected_mask = torch.ones((bsz, num_kv_heads, kv_len), device=key_states.device, dtype=torch.bool)
        unselected_mask.scatter_(-1, i_indices, 0)
        unselected_mask.scatter_(-1, j_indices, 0)
        unselected_indices = unselected_mask.nonzero(as_tuple=True)
        
        k_kept = key_states[unselected_indices].view(bsz, num_kv_heads, kv_len - 2 * topk, head_dim)
        v_kept = value_states[unselected_indices].view(bsz, num_kv_heads, kv_len - 2 * topk, head_dim)
        
        k_first = torch.gather(
            key_states, 2, i_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )
        v_first = torch.gather(
            value_states, 2, i_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )

        k_second = torch.gather(
            key_states, 2, j_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )
        v_second = torch.gather(
            value_states, 2, j_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )

        score_reduced = score_exp.mean(dim=-2)  # shape: (bsz, num_kv_heads, kv_len)

        score_first = torch.gather(score_reduced, -1, i_indices).unsqueeze(-1)
        score_second = torch.gather(score_reduced, -1, j_indices).unsqueeze(-1)

        # print(f"[1]: {score_first / (score_second + score_first)}\n")
        # print(f"[2]: {score_first / (score_first + score_second)}\n")
        # print(f"[3] : {score_second / (score_second + score_first)}\n")
        # print(f"[4] : {score_second / (score_first + score_second)}\n")
        
        # print("-" * 100)

        k_merged = (score_first / (score_first + score_second)) * k_first + (score_second / (score_second + score_first)) * k_second

        score_kept  = torch.einsum(
            "b g h l d, b h m d -> b g h l m",
            query_states,
            k_kept,
        ).mean(dim=1) / math.sqrt(head_dim)
        
        score_merged_first = torch.einsum(
            "b g h l d, b h m d -> b g h l m",
            query_states,
            k_first,
        ).mean(dim=1) / math.sqrt(head_dim)
        
        score_merged_second = torch.einsum(
            "b g h l d, b h m d -> b g h l m",
            query_states,
            k_second,
        ).mean(dim=1) / math.sqrt(head_dim)
        
        # it should be maximum(kept, first, second) however second is ususally larger than first
        score_merged_second -= torch.maximum(score_kept.amax(dim=-1, keepdim=True), score_merged_second.amax(dim=-1, keepdim=True))

        score_merged_first -= torch.maximum(score_kept.amax(dim=-1, keepdim=True), score_merged_second.amax(dim=-1, keepdim=True))

        score_merged = torch.einsum(
            "b g h l d, b h m d -> b g h l m",
            query_states, 
            k_merged,
        ).mean(dim=1) / math.sqrt(head_dim)

        score_merged -= torch.maximum(score_kept.amax(dim=-1, keepdim=True), score_merged.amax(dim=-1, keepdim=True))
        
        score_first = torch.exp(
            score_merged_first - score_merged
        ).mean(-2).unsqueeze(-1)
        
        score_second = torch.exp(
            score_merged_second - score_merged
        ).mean(-2).unsqueeze(-1)

        # score_merged = score_merged_first - score_merged_second

        # score_merged_first = score_second.squeeze(-1).unsqueeze(-2) * score_merged

        # score_merged_first = torch.exp(
        #     score_merged_first
        # ).mean(-2).unsqueeze(-1)

        # score_merged_second = - score_first.squeeze(-1).unsqueeze(-2) * score_merged

        # score_merged_second = torch.exp(
        #     score_merged_second
        # ).mean(-2).unsqueeze(-1)
                
        # v_merged = score_first * v_first + score_second * v_second

        v_merged = (score_first / (score_first + score_second)) * v_first + (score_second / (score_second + score_first)) * v_second
        k_final = torch.cat([k_kept, k_merged], dim=2)
        v_final = torch.cat([v_kept, v_merged], dim=2)

        return {"key_states": k_final, 
                "value_states": v_final, 
                }
