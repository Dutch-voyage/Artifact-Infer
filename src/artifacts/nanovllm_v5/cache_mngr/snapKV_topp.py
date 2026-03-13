import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import compute_attention_scores
from src.services.nanovllm_v5.utils.logging import append_num_topp, append_selected_indices
from flashinfer.sampling import top_p_renorm_probs
import triton 
import triton.language as tl
from functools import partial

@triton.jit
def gather_from_topp_kernel(
    selected_indices_ptr,
    num_selected_ptr,
    key_states_ptr,
    value_states_ptr,
    out_key_ptr,
    out_value_ptr,
    lock_ptr, 
    num_locks: tl.constexpr,
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
    topp_mask = selected_indices > 0
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
        num_locks, 
        head_dim,
        kv_len,
        BN
    )
    
    out_key_states = out_key_states.view(bsz, num_kv_heads, kv_len, head_dim)
    out_value_states = out_value_states.view(bsz, num_kv_heads, kv_len, head_dim)
    num_selected = num_selected.view(bsz, num_kv_heads)
    
    return out_key_states, out_value_states, num_selected


class SnapKV:
    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        record_kept_token_indices=False,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.p_attn = kwargs["p_attn"]

        # for recording kept token indices
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []

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
                    attn_weights[:, :, -self.window_size :, : -self.window_size],
                    dim=-1,
                    dtype=torch.float32,
                )
                .mean(dim=-2)
                .to(query_states.dtype)
            )

            attn_cache = attn_weights_sum
            # attn_cache = F.max_pool1d(
            #     attn_weights_sum,
            #     kernel_size=self.kernel_size,
            #     padding=self.kernel_size // 2,
            #     stride=1,
            # )
            
            # shape: (bsz, num_kv_heads, budget - window_size)
            # indices = attn_cache.topk(self.budget - self.window_size, dim=-1).indices
            
            attn_topp_normed = top_p_renorm_probs(attn_cache.view(-1, attn_cache.shape[-1]), top_p=self.p_attn)
            
            selected_indices = torch.vmap(partial(torch.nonzero_static, size=attn_cache.shape[-1]), in_dims=(0,))(attn_topp_normed).squeeze(-1)
            # torch.set_printoptions(threshold=10000)

            # print(selected_indices)
            
            out_key_states, out_value_states, num_selected = gather_from_topp(
                selected_indices,
                key_states[:, :, : -self.window_size, :],
                value_states[:, :, : -self.window_size, :],
            )
            
            append_num_topp(num_selected)
            append_selected_indices(selected_indices)
            
            # indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            # k_past_compress = key_states[:, :, : -self.window_size, :].gather(
            #     dim=2, index=indices
            # )
            # v_past_compress = value_states[:, :, : -self.window_size, :].gather(
            #     dim=2, index=indices
            # )
            # k_cur = key_states[:, :, -self.window_size :, :]
            # v_cur = value_states[:, :, -self.window_size :, :]
            # key_states = torch.cat([k_past_compress, k_cur], dim=2)
            # value_states = torch.cat([v_past_compress, v_cur], dim=2)
            
            return {"key_states": key_states, 
                    "value_states": value_states, 
                    "num_selected": num_selected, }