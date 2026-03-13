import torch
import math
from src.services.nanovllm_v8.utils.logging import append_num_topp, append_selected_indices, append_item_to_log
from flashinfer.sampling import top_p_renorm_probs
from functools import partial
import triton
import triton.language as tl

def gather_selected_kv(kv_cache, indicator):
    """
    Args:
        kv_cache: [bsz, num_kv_heads, kv_len, head_dim]
        indicator: [bsz, num_kv_heads, kv_len] (bool or 0/1) indicating selection
    
    Returns:
        selected_kv: [bsz, num_kv_heads, max_selected_len, head_dim]
    """
    bsz, num_kv_heads, kv_len, head_dim = kv_cache.shape

    # 1. Calculate the target length (max selections across the batch/heads)
    #    We assume the user wants the tensor padded to the longest selection in the batch.
    num_selected = indicator.to(torch.bool).sum(dim=-1)  # [bsz, num_kv_heads]
    max_selected_len = num_selected.max().item()
    
    # Optional: If nothing is selected, return empty tensor
    if max_selected_len == 0:
        return torch.zeros((bsz, num_kv_heads, 0, head_dim), 
                           device=kv_cache.device, dtype=kv_cache.dtype), num_selected

    # 2. Get gather indices using Stable Sort
    #    descending=True puts 1s (True) before 0s (False).
    #    stable=True ensures selected tokens remain in their original relative order (time order).
    sorted_indices = torch.argsort(indicator, dim=-1, descending=True, stable=True)
    
    #    Truncate to the max length needed
    gather_indices = sorted_indices[..., :max_selected_len] # [bsz, num_kv_heads, max_selected]

    # 3. Gather the KV cache
    #    We need to expand indices to match the head_dim
    gather_indices_expanded = gather_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    
    selected_kv = torch.gather(kv_cache, 2, gather_indices_expanded)

    # # 4. Zero out padding (The 'gather' pulled unselected tokens into the padding slots)
    # #    Create a mask: [1, 1, max_selected] < [bsz, num_heads, 1]
    # range_vector = torch.arange(max_selected_len, device=kv_cache.device).view(1, 1, -1)
    # mask = range_vector < num_selected.unsqueeze(-1)
    
    # #    Apply mask
    # selected_kv = selected_kv * mask.unsqueeze(-1)

    return selected_kv, num_selected

"""
The spin lock design is adapted from Cut-Cross-Entropy
https://github.com/apple/ml-cross-entropy/blob/b7a02791b234e187b524fb1dba6a812d521b203a/cut_cross_entropy/cce_lse_forward.py#L12
"""

@triton.jit
def gather_from_topp_kernel(
    selected_indices_ptr,
    num_selected_ptr,
    key_states_ptr,
    value_states_ptr,
    out_key_ptr,
    out_value_ptr,
    lock_ptr, 
    head_dim: tl.constexpr,
    N: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_b = pid // num_pid_n 
    pid_n = pid % num_pid_n

    # Load normalized probabilities
    
    indices_offsets = pid_b * N + pid_n * BN + tl.arange(0, BN)
    indices_mask = pid_n * BN + tl.arange(0, BN) < N
        
    selected_indices = tl.load(selected_indices_ptr + indices_offsets, mask=indices_mask, other=-1)

    # Create mask for elements with prob > 0
    topp_mask = selected_indices > -1
    # Count selected elements in this block
    num_selected = tl.sum(topp_mask, axis=0)
    
    # tl.device_print("num_selected", num_selected)
    
    if num_selected == 0:
        return 
    
    this_lock = lock_ptr + pid_b
    while tl.atomic_cas(this_lock, 0, 1) == 1: 
        pass
    
    num_selected_prev = tl.load(num_selected_ptr + pid_b)
    # print(num_selected_prev)
    
    # Load key and value for this selected position
    k_val = tl.load(key_states_ptr + pid_b * N * head_dim + selected_indices[:, None] * tl.arange(0, head_dim)[None, :], mask=topp_mask[:, None])
    v_val = tl.load(value_states_ptr + pid_b * N * head_dim + selected_indices[:, None] * tl.arange(0, head_dim)[None, :], mask=topp_mask[:, None])

    pos = num_selected_prev + tl.arange(0, BN)   
    # Store in consecutive positions in output
    tl.store(out_key_ptr + pid_b * N * head_dim + pos[:, None] * tl.arange(0, head_dim)[None, :], k_val, mask=topp_mask[:, None])
    tl.store(out_value_ptr + pid_b * N * head_dim + pos[:, None] * tl.arange(0, head_dim)[None, :], v_val, mask=topp_mask[:, None])

    num_selected += num_selected_prev
    tl.store(num_selected_ptr + pid_b, num_selected)
    
    tl.debug_barrier()
    tl.atomic_xchg(this_lock, 0)

    
def gather_from_topp(
    selected_indices, 
    key_states,
    value_states,
):
    bsz, num_kv_heads, kv_len, head_dim = key_states.shape
    selected_indices = selected_indices.to(torch.int16).contiguous()
    key_states = key_states.view(-1, kv_len, head_dim)
    value_states = value_states.view(-1, kv_len, head_dim)
    
    out_key_states = torch.zeros_like(key_states)
    out_value_states = torch.zeros_like(value_states)
    num_selected = torch.zeros((selected_indices.shape[0],), dtype=torch.int16, device=selected_indices.device)
    
    locks = torch.full((selected_indices.shape[0],), 0, dtype=torch.uint32).to(selected_indices.device)
    num_locks = locks.shape[0]
    BN = 32
        
    gather_from_topp_kernel[(selected_indices.shape[0] * triton.cdiv(selected_indices.shape[1], BN), )](
        selected_indices,
        num_selected,
        key_states,
        value_states,
        out_key_states,
        out_value_states,
        locks, 
        head_dim,
        kv_len,
        BN
    )
    
    out_key_states = out_key_states.view(bsz, num_kv_heads, kv_len, head_dim)
    out_value_states = out_value_states.view(bsz, num_kv_heads, kv_len, head_dim)
    num_selected = num_selected.view(bsz, num_kv_heads)
    
    return out_key_states, out_value_states, num_selected

"""
Advised by gemini pro to remove spin locks and use atomic operations for gathering from topp
However, this may cause illegal memory access sometimes, need to investigate further.
"""

@triton.jit
def gather_from_topp_kernel_by_gemini(
    selected_indices_ptr,
    num_selected_ptr,
    key_states_ptr,
    value_states_ptr,
    out_key_ptr,
    out_value_ptr,
    # lock_ptr, # REMOVED: No longer needed
    head_dim: tl.constexpr,
    N: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_b = pid // num_pid_n 
    pid_n = pid % num_pid_n

    # 1. Load Indices
    indices_offsets = pid_b * N + pid_n * BN + tl.arange(0, BN)
    indices_mask = pid_n * BN + tl.arange(0, BN) < N
    
    # Load indices, default to -1.
    selected_indices = tl.load(selected_indices_ptr + indices_offsets, mask=indices_mask, other=-1)

    # 2. Identify Valid Selections
    topp_mask = selected_indices > -1
    
    # Early exit if nothing selected in this block
    # casting to int32 is safer for accumulation
    topp_mask_i32 = topp_mask.to(tl.int32)
    num_selected_in_block = tl.sum(topp_mask_i32, axis=0)
    
    if num_selected_in_block == 0:
        return 

    # 3. Global Reservation (The "Lock-Free" Magic)
    # atomic_add returns the value *before* the addition, which is our start offset.
    # We use int32 for counters to ensure atomic hardware support.
    global_offset = tl.atomic_add(num_selected_ptr + pid_b, num_selected_in_block)

    # 4. Local Compaction (CRITICAL FIX)
    # We need to squeeze sparse valid items into dense output slots.
    # cumsum gives us [1, 1, 2, 2, 3...] for mask [T, F, T, F, T...]
    # Subtract 1 to get 0-based offsets: [0, 0, 1, 1, 2...]
    local_offsets = tl.cumsum(topp_mask_i32, axis=0) - 1
    
    # Calculate final destination indices: Global Start + Local Compacted Offset
    write_indices = global_offset + local_offsets

    # 5. Gather and Scatter
    # Pointer Arithmetic:
    # Use tl.where to ensure we don't calculate garbage pointers for -1 indices, 
    # though the mask protects the load, this is cleaner.
    safe_indices = tl.where(topp_mask, selected_indices, 0)
    
    # Offsets for reading (Sparse)
    # [BN, head_dim]
    dim_offsets = tl.arange(0, head_dim)[None, :]
    base_read = pid_b * N * head_dim
    read_ptrs = base_read + safe_indices[:, None] * head_dim + dim_offsets

    # Offsets for writing (Compacted)
    base_write = pid_b * N * head_dim
    write_ptrs = base_write + write_indices[:, None] * head_dim + dim_offsets
    
    # Mask for the 2D block [BN, head_dim]
    io_mask = topp_mask[:, None] # Broadcast to head_dim

    # Load Key/Values
    k_val = tl.load(key_states_ptr + read_ptrs, mask=io_mask)
    v_val = tl.load(value_states_ptr + read_ptrs, mask=io_mask)

    # Store to compacted positions
    tl.store(out_key_ptr + write_ptrs, k_val, mask=io_mask)
    tl.store(out_value_ptr + write_ptrs, v_val, mask=io_mask)

def gather_from_topp_by_gemini(
    selected_indices, 
    key_states,
    value_states,
):
    bsz, num_kv_heads, kv_len, head_dim = key_states.shape
    selected_indices = selected_indices.to(torch.int16).contiguous()
    
    # Flatten view for simpler pointer math in kernel
    key_states_flat = key_states.view(-1, kv_len, head_dim)
    value_states_flat = value_states.view(-1, kv_len, head_dim)
    
    out_key_states = torch.zeros_like(key_states_flat)
    out_value_states = torch.zeros_like(value_states_flat)
    
    # Use int32 for atomic counters (safer than int16 on GPU)
    num_selected = torch.zeros((bsz * num_kv_heads,), dtype=torch.int32, device=selected_indices.device)
    
    BN = 32
    
    grid = (selected_indices.shape[0] * triton.cdiv(selected_indices.shape[1], BN), )
    
    gather_from_topp_kernel_by_gemini[grid](
        selected_indices,
        num_selected,
        key_states_flat,
        value_states_flat,
        out_key_states,
        out_value_states,
        # lock argument removed
        head_dim,
        kv_len,
        BN
    )
    
    out_key_states = out_key_states.view(bsz, num_kv_heads, kv_len, head_dim)
    out_value_states = out_value_states.view(bsz, num_kv_heads, kv_len, head_dim)
    num_selected = num_selected.view(bsz, num_kv_heads)
    
    return out_key_states, out_value_states, num_selected

@triton.jit
def gather_num_selected_kernel(
    selected_indices_ptr,
    num_selected_ptr,
    lock_ptr, 
    N: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_b = pid // num_pid_n 
    pid_n = pid % num_pid_n

    # Load normalized probabilities
    
    indices_offsets = pid_b * N + pid_n * BN + tl.arange(0, BN)
    indices_mask = pid_n * BN + tl.arange(0, BN) < N
        
    selected_indices = tl.load(selected_indices_ptr + indices_offsets, mask=indices_mask, other=-1)

    # Create mask for elements with prob > 0
    topp_mask = selected_indices > -1
    # Count selected elements in this block
    num_selected = tl.sum(topp_mask, axis=0)
    
    # tl.device_print("num_selected", num_selected)
    
    if num_selected == 0:
        return 
    
    this_lock = lock_ptr + pid_b
    while tl.atomic_cas(this_lock, 0, 1) == 1: 
        pass
    
    num_selected_prev = tl.load(num_selected_ptr + pid_b)
    # print(num_selected_prev)

    num_selected += num_selected_prev
    tl.store(num_selected_ptr + pid_b, num_selected)
    
    tl.debug_barrier()
    tl.atomic_xchg(this_lock, 0)

# gather_from_topp = gather_from_topp_by_gemini

def gather_num_selected(
    selected_indices, 
):
    bsz, kv_len = selected_indices.shape
    selected_indices = selected_indices.to(torch.int16).contiguous()
    num_selected = torch.zeros((selected_indices.shape[0],), dtype=torch.int16, device=selected_indices.device)
    
    locks = torch.full((selected_indices.shape[0],), 0, dtype=torch.uint32).to(selected_indices.device)
    BN = 32
    
    gather_num_selected_kernel[(selected_indices.shape[0] * triton.cdiv(selected_indices.shape[1], BN), )](
        selected_indices,
        num_selected,
        locks, 
        kv_len,
        BN
    )
    
    num_selected = num_selected.view(bsz)
    
    return num_selected

@triton.jit
def gather_num_selected_kernel_gemini(
    selected_indices_ptr,
    num_selected_ptr,
    # lock_ptr removed: Not needed
    N: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate row (pid_b) and block-within-row (pid_n)
    num_pid_n = tl.cdiv(N, BN)
    pid_b = pid // num_pid_n 
    pid_n = pid % num_pid_n

    # 1. Load Data
    indices_offsets = pid_b * N + pid_n * BN + tl.arange(0, BN)
    indices_mask = pid_n * BN + tl.arange(0, BN) < N
        
    selected_indices = tl.load(selected_indices_ptr + indices_offsets, mask=indices_mask, other=-1)

    # 2. Count Valid Elements
    topp_mask = selected_indices > -1
    
    # Cast to int32 for summation to ensure compatibility with atomic_add
    num_selected_block = tl.sum(topp_mask.to(tl.int32), axis=0)
    
    # Optimization: Skip memory write if count is 0
    if num_selected_block == 0:
        return 
    
    # 3. Atomic Update (Lock-Free)
    # Directly add this block's count to the global total for this row
    tl.atomic_add(num_selected_ptr + pid_b, num_selected_block)


def gather_num_selected_gemini(
    selected_indices, 
):
    bsz, kv_len = selected_indices.shape
    selected_indices = selected_indices.to(torch.int16).contiguous()
    
    # IMPORTANT: Use int32 for the counter to ensure safe atomic operations on all GPU architectures.
    # (int16 atomics support is limited/hardware-dependent)
    num_selected = torch.zeros((bsz,), dtype=torch.int32, device=selected_indices.device)
    
    BN = 32
    
    # Grid size covers total number of blocks needed
    grid = (bsz * triton.cdiv(kv_len, BN), )
    
    gather_num_selected_kernel_gemini[grid](
        selected_indices,
        num_selected,
        # lock argument removed
        N=kv_len,
        BN=BN
    )
    
    return num_selected

# gather_num_selected = gather_num_selected_gemini

def update_log(attn_cache, key_states, value_states, query_states, p_attn, sink_size=0, window_size=0):
    q_cache_len = query_states.shape[-2]
    kv_len = key_states.shape[-2]
    attn_topp_normed = top_p_renorm_probs(attn_cache.view(-1, attn_cache.shape[-1]), top_p=p_attn)
    
    selected_indices = torch.vmap(partial(torch.nonzero_static, size=attn_cache.shape[-1]), in_dims=(0,))(attn_topp_normed).squeeze(-1)
    
    out_key_states, out_value_states, num_selected = gather_from_topp(
        selected_indices,
        key_states[:, :, sink_size: kv_len - window_size, :],
        value_states[:, :, sink_size: kv_len - window_size, :],
    )
    # print(num_selected)
    
    attn_weights = compute_attention_scores(query_states, key_states, pooling="none") # .squeeze(0)
    # [bsz, num_kv_heads, num_groups, q_cache_len, kv_cache_len]
    
    # number_selected: [bsz, num_kv_heads]
    
    # 1. Create a range of indices [0, 1, ..., kv_len-1]
    # Reshape to [1, 1, 1, 1, kv_len] to align with the last dimension of attn_weights
    indices = torch.arange(kv_len, device=attn_weights.device).view(1, 1, 1, 1, -1)
    
    # 2. Reshape num_selected to broadcast against attn_weights
    # Current: [bsz, num_kv_heads]
    # Target:  [bsz, num_kv_heads, 1, 1, 1] (Broadcasting over num_groups and q_len)
    num_selected_reshaped = num_selected.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    lse = torch.logsumexp(attn_weights, dim=-1).view(-1, q_cache_len).transpose(0, 1)
    
    # 3. Create the mask
    # masked_fill expects True where the value should be REPLACED (masked out).
    # So we want True where index >= num_selected (valid indices are < num_selected)
    mask = (indices >= (num_selected_reshaped + sink_size)) & (indices < (kv_len - window_size))
    
    attn_weights = torch.masked_fill(attn_weights, mask, float("-inf"))
    
    lse_topp = torch.logsumexp(attn_weights, dim=-1).view(-1, q_cache_len).transpose(0, 1) # [window_len, num_heads]
    
    # print(lse.max().item())
    # print("------" * 10)
    
    append_item_to_log("lse_log", lse.float().detach().cpu())
    append_item_to_log("lse_topp_log", lse_topp.float().detach().cpu())
    append_num_topp(num_selected)
    append_selected_indices(selected_indices)
    
    return num_selected

def compute_attention_scores(query_states, key_states, pooling="max"):
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    query_group_size = q_heads // kv_heads

    if query_group_size == 1:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
    else:
        # shape: [batch_size, kv_heads, query_group_size, q_len, head_dim]
        query_states = query_states.view(
            batch_size, kv_heads, query_group_size, q_len, head_dim
        )

        # shape: [batch_size, kv_heads, 1, kv_len, head_dim]
        key_states = key_states.unsqueeze(2)

        # shape: [batch_size, kv_heads, query_group_size, q_len, kv_len]
        attn_weights = torch.matmul(
            query_states, key_states.transpose(3, 4)
        ) / math.sqrt(head_dim)

        # apply pooling over query_group_size dimension
        if pooling == "mean":
            attn_weights = attn_weights.mean(dim=2)
        elif pooling == "max":
            attn_weights = attn_weights.max(dim=2).values
        elif pooling == "none":
            attn_weights = attn_weights
        else:
            raise ValueError("Pooling method not supported")

    return attn_weights

@torch.no_grad()
def cal_similarity(
    key_cache,
    threshold=0.5,
    aggregation="mean",
    normalization=False,
    retain_num=None,
    retain_ratio=None,
    retain_direction="last",
):
    k = key_cache[0]
    num_heads = k.shape[0]

    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2)).detach()

    for h in range(num_heads):
        similarity_cos[h].fill_diagonal_(0.0)

    # shape: [num_heads, seq_len, seq_len]
    similarity_mask = similarity_cos > threshold

    if retain_ratio is not None and retain_num is not None:
        raise ValueError("retain_ratio and retain_num cannot be used together")
    if retain_ratio is None and retain_num is None:
        raise ValueError("retain_ratio or retain_num must be provided")
    if retain_num is not None:
        k = retain_num if retain_num is not None else 1
    else:
        seq_len = similarity_mask.size(-1)
        k = int(seq_len * retain_ratio)  # 改为直接使用比例

    indices = torch.where(
        similarity_mask,
        torch.arange(similarity_mask.size(-1), device=similarity_mask.device),
        torch.zeros_like(similarity_mask, dtype=torch.long),
    )

    if retain_direction == "last":
        # find the last True index in each row
        similarity_retain = torch.max(indices, dim=-1)[0]
    elif retain_direction == "first":
        # find the first True index in each row
        similarity_retain = torch.min(indices, dim=-1)[0]
    elif retain_direction == "last_percent":
        # 保留位置在后百分比的元素
        similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]
    elif retain_direction == "first_percent":
        # 保留位置在前百分比的元素
        similarity_retain = torch.topk(indices, k=k, dim=-1, largest=False)[0][:, :, -1]

    # create indices for zeroing
    batch_idx = (
        torch.arange(num_heads).unsqueeze(1).repeat(1, similarity_retain.size(1))
    )
    seq_idx = torch.arange(similarity_retain.size(1)).unsqueeze(0).repeat(num_heads, 1)

    # zero the specified positions in similarity_cos
    similarity_cos[batch_idx, seq_idx, similarity_retain] = 0

    if aggregation == "mean":
        similarity_cos = similarity_cos.mean(dim=1)
    elif aggregation == "max":
        similarity_cos = similarity_cos.max(dim=1).values
    

    if normalization:
        similarity_cos = similarity_cos.softmax(dim=-1)

    return similarity_cos

def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M 

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    selected_keys = key_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]

    # kv-drop
    all_indices = torch.arange(k_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, k_len)
    all_indices_flattened = all_indices.flatten()  # [bsz * num_heads * (k_len-window_size)]
    selected_indices_flattened = indices.flatten()  # [bsz * num_heads * topk_len]
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected] 
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] * all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(all_indices.shape[0], all_indices.shape[1], drop_len) # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # kv-recent
    recent_keys = key_states[:, :, -window_size:, :]

    ##### apply merge #####
    # prepare for merge
    k_hh_pruned = drop_keys  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_recent = torch.cat([recent_keys, selected_keys], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    v_hh_pruned = drop_values  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_recent = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
    max_values, max_indices = similarity.max(dim=-1)

    # pivot merge
    if merge=="pivot":
        print("Pivot merge")
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    else:
        raise ValueError('Merge method not supported')
        
    # TODO: other merge strategies
    # average merge
    # weight merge

    return k_hh_recent, v_hh_recent
