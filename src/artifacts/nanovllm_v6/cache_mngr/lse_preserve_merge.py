
import torch

def merge_fixed_budget(score, raw_score, topk, key_states, value_states):
    bsz, num_kv_heads, kv_len, head_dim = key_states.shape
    
    num_selected = kv_len - topk
    
    unselected_mask = torch.ones(
        (bsz, num_kv_heads, kv_len), device=key_states.device, dtype=torch.bool
    )
    
    idx_desc = raw_score.topk(
        num_selected, dim=-1
    ).indices  # shape: (bsz, num_kv_heads, topk)
    unselected_mask.scatter_(-1, idx_desc, 0)
    idx_asc = (
        (torch.where(unselected_mask, -score, -float("inf")))
        .topk(num_selected, dim=-1)
        .indices
    )  #
    # idx_asc = idx_asc.flip(-1)
    assert (idx_desc.unsqueeze(-1) != idx_asc.unsqueeze(-2)).all()
    unselected_mask.scatter_(-1, idx_asc, 0)
    
    unselected_indices = unselected_mask.nonzero(as_tuple=True)
    
    k_kept = key_states[unselected_indices].view(
        bsz, num_kv_heads, topk - num_selected, head_dim
    )
    v_kept = value_states[unselected_indices].view(
        bsz, num_kv_heads, topk - num_selected, head_dim
    )

    score_max = torch.gather(raw_score, -1, idx_desc).unsqueeze(-1)
    score_min = torch.gather(raw_score, -1, idx_asc).unsqueeze(-1)

    k_merged = (score_max / (score_max + score_min)) * torch.gather(
        key_states, 2, idx_desc.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    ) + (score_min / (score_max + score_min)) * torch.gather(
        key_states, 2, idx_asc.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    )
    
    v_merged = (score_max / (score_max + score_min)) * torch.gather(
        value_states, 2, idx_desc.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    ) + (score_min / (score_max + score_min)) * torch.gather(
        value_states, 2, idx_asc.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    )
    
    k_final = torch.cat([k_kept, k_merged], dim=2)
    v_final = torch.cat([v_kept, v_merged], dim=2)
    
    return k_final, v_final