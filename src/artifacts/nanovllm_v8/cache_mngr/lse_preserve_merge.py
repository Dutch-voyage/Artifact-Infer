
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

m = 1

def merge_multi_to_one(score, raw_score, topk, key_states, value_states):
    bsz, num_kv_heads, kv_len, head_dim = key_states.shape
    num_selected = kv_len - topk
    
    # 1. Identify Indices
    # Top m scores (Destination / Survivors)
    idx_desc = score.topk(m, dim=-1).indices.flip(-1)
    
    # Identify Unselected for exclusion
    unselected_mask = torch.ones(
        (bsz, num_kv_heads, kv_len), device=key_states.device, dtype=torch.bool
    )
    unselected_mask.scatter_(-1, idx_desc, 0)
    
    # Bottom 'num_selected' scores from the remaining (Sources / Merged)
    # We pick the smallest scores to merge into the top scores
    idx_asc = (
        (torch.where(unselected_mask, -score, -float("inf")))
        .topk(num_selected, dim=-1)
        .indices
    )
    
    # 2. Gather Candidates (Corrected Order: Sources -> Destinations)
    # We place 'asc' (low scores) BEFORE 'desc' (high scores) so 'desc' can attend to 'asc'.
    score_desc = raw_score.gather(-1, idx_desc).unsqueeze(-1)
    score_asc = raw_score.gather(-1, idx_asc).unsqueeze(-1)
    
    # N = num_selected + m
    a_0 = torch.cat([score_asc, score_desc], dim=2) 
    
    k_0 = torch.cat([
        key_states.gather(2, idx_asc.unsqueeze(-1).expand(-1, -1, -1, head_dim)), 
        key_states.gather(2, idx_desc.unsqueeze(-1).expand(-1, -1, -1, head_dim))
    ], dim=2)
    
    v_0 = torch.cat([
        value_states.gather(2, idx_asc.unsqueeze(-1).expand(-1, -1, -1, head_dim)), 
        value_states.gather(2, idx_desc.unsqueeze(-1).expand(-1, -1, -1, head_dim))
    ], dim=2)

    # 3. Calculate Log-Space Weights (The Math Fix)
    # LS shape: (B, H, N, 1). Note dim=2 is sequence length.
    LS = torch.log(torch.cumsum(a_0, dim=2))
    
    # We only want the outputs corresponding to the last 'm' tokens (the desc tokens)
    # Target LS shape: (B, H, m, 1)
    LS_target = LS[..., -m:, :] 
    
    # Inputs: ln_a_0 shape (B, H, N, 1) -> Transpose to (B, H, 1, N) for broadcasting
    ln_a_0 = torch.log(a_0).transpose(-2, -1)
    
    # ln_w shape: (B, H, m, N)
    # Formula: ln(w_ij) = ln(a_j) - LS_i
    ln_w = ln_a_0 - LS_target
    
    # 4. Causal Masking
    # We need a mask (m, N) where row i can see columns j <= (offset + i)
    # The 'desc' tokens are at indices [num_selected, num_selected+1, ...] in the sequence a_0.
    N = a_0.shape[2]
    device = key_states.device
    
    cols = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
    rows = torch.arange(m, device=device).unsqueeze(1) + num_selected # (m, 1)
    
    mask = cols <= rows # Valid if input_index <= output_index
    
    ln_w = ln_w.masked_fill(~mask, -float('inf'))
    
    w = torch.exp(ln_w) # (B, H, m, N)
    
    # 5. Compute Compressed States
    # (B, H, m, N) @ (B, H, N, D) -> (B, H, m, D)
    
    k_out = w @ k_0
    # k_out = key_states.gather(2, idx_desc.unsqueeze(-1).expand(-1, -1, -1, head_dim))
    v_out = w @ v_0
    
    # 6. Handle "Kept" (Middle) Tokens
    # These are tokens not in Top-m AND not in Bottom-K. They are kept as is.
    unselected_mask.scatter_(-1, idx_asc, 0)
    unselected_indices = unselected_mask.nonzero(as_tuple=True)
    
    k_kept = key_states[unselected_indices].view(bsz, num_kv_heads, -1, head_dim)
    v_kept = value_states[unselected_indices].view(bsz, num_kv_heads, -1, head_dim)
    
    # Final concatenation: Compressed High-Scores + Unaffected Middle-Scores
    k_final = torch.cat([k_out, k_kept], dim=2)
    v_final = torch.cat([v_out, v_kept], dim=2)
    
    return k_final, v_final