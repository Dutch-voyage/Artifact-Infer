from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import compute_attention_scores, gather_from_topp, update_log
from .lse_preserve_merge import merge_fixed_budget
from src.services.nanovllm_v8.utils.logging import append_item_to_log
from .binary_search import binary_search_T, gradient_descent_T

from flashinfer.sampling import top_p_renorm_probs
from flashinfer.quantization import segment_packbits

import triton
import triton.language as tl

class VanillaToppKV:
    def __init__(
        self,
        config, 
        budget=128,
        window_size=8,
        kernel_size=7,
        record_kept_token_indices=False,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.sink_size = 1
        self.window_size = window_size - self.sink_size
        self.kernel_size = kernel_size

        self.lse_preserve_merge = config.lse_preserve_merge
        self.if_log_compress = config.if_log_compress
        self.p_attn = config.p_attn
        
        # for recording kept token indices
        self.record_kept_token_indices = record_kept_token_indices

    def update_kv(
        self,
        query_states,
        key_states,
        value_states,
        effective_kv_head_lens=None,
        *args, 
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
                # 1. Create a sequence tensor: [0, 1, ..., kv_cache_len-1] on the correct device
                # Shape: (1, 1, kv_cache_len)
                indices = torch.arange(
                    kv_cache_len, device=key_states.device
                ).view(1, 1, -1)

                # 2. Reshape lengths for broadcasting
                # Shape: (bsz, num_heads, 1)
                lengths = effective_kv_head_lens.unsqueeze(-1)

                # 3. Create the mask in one go using broadcasting
                # Result Shape: (bsz, num_heads, kv_cache_len)
                # True where index < length (valid), False where index >= length (padding)
                effective_mask = indices < lengths.to(indices.device)
                # 4. Apply the mask
                # We invert the mask (~) because masked_fill fills where the condition is True.
                # We unsqueeze dim 2 to match attn_weights shape (bsz, num_heads, q_len, kv_len)
                attn_weights = attn_weights.masked_fill(~effective_mask.unsqueeze(2), float("-inf"))
                        
            # raw_attn_weights = attn_weights[:, :, :, self.sink_size : -self.window_size]# .view(-1, kv_cache_len - self.window_size)
                        
            attn_cache = (
                nn.functional.softmax(
                    attn_weights,
                    dim=-1,
                    dtype=torch.float32,
                )
                .to(query_states.dtype)
            ) 
            
            # if self.if_log_compress:
            #     update_log(attn_cache, 
            #                key_states, 
            #                value_states, 
            #                query_states, 
            #                self.p_attn, 
            #                self.sink_size, 
            #                self.window_size)
            
            # if self.lse_preserve_merge:
            #     k_compress, v_compress = merge_fixed_budget(
            #         attn_cache,
            #         raw_attn_weights.softmax(dim=-1).mean(-2), 
            #         self.budget - self.window_size - self.sink_size,
            #         key_states[:, :, self.sink_size : -self.window_size, :],
            #         value_states[:, :, self.sink_size : -self.window_size, :],
            #     )
            # else:
            attn_topp_normed = top_p_renorm_probs(attn_cache.view(-1, attn_cache.shape[-1]), top_p=self.p_attn)
            
            unselected_mask = (attn_topp_normed == torch.zeros_like(attn_topp_normed))
            # selected_mask = torch.zeros_like(selected_indices, dtype=torch.bool).squeeze(0) # bsz = 1 in the current implementation
                    
            # selected_mask[..., :self.sink_size] = True
            
            # selected_mask[..., -self.window_size:] = True
            
            # scatter_with_mask(torch.ones_like(selected_indices, dtype=torch.bool), selected_indices, selected_mask)
            
            unselected_mask = unselected_mask.reshape(num_kv_heads, q_cache_len, -1)
                        
            selected_mask = ~(torch.prod(unselected_mask.to(torch.int32), dim=-2).to(torch.bool))
            
            # save the top budget indices
            indices_desc_topk = attn_cache.mean(-2).squeeze(0).topk(self.budget - self.window_size, dim=-1).indices
            selected_mask.scatter_(-1, indices_desc_topk, True)
            
            selected_mask[..., :self.sink_size] = True
            selected_mask[..., -self.window_size:] = True
            
            mask_indptr = torch.arange(0, num_kv_heads + 1).to(selected_mask.device) * kv_cache_len
                        
            packed_selected_mask, mask_indptr_new = segment_packbits(selected_mask.view(-1), mask_indptr, bitorder="little")
            
            packed_selected_mask = packed_selected_mask.view(8, -1)
            
            return {"key_states": key_states, "value_states": value_states, "packed_selected_mask": packed_selected_mask}

@triton.jit
def scatter_with_mask_kernel(
    src_ptr,
    index_ptr,
    output_ptr,
    
    stride_src_b,
    stride_index_b,
    stride_output_b,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # 1. Identify the location of this program instance in the grid
    pid_N = tl.program_id(axis=0)
    pid_B = tl.program_id(axis=1)
    
    block_start_N = pid_N * BLOCK_SIZE
    N_offsets = block_start_N + tl.arange(0, BLOCK_SIZE)
    
    # 2. Create a mask to guard against out-of-bounds memory accesses
    #    (essential if n_elements is not a multiple of BLOCK_SIZE)
    mask = N_offsets < N

    src_B_start = src_ptr + pid_B * stride_src_b
    index_B_start = index_ptr + pid_B * stride_index_b
    output_B_start = output_ptr + pid_B * stride_output_b

    # 3. Load the indices and the source values
    #    We use the mask here to ensure we don't read off the end of the tensor
    index_vals = tl.load(index_B_start + N_offsets, mask=mask, other=-1)
    src_vals = tl.load(src_B_start + N_offsets, mask=mask, other=0.0)

    # 4. Check the condition: Is the index NOT -1?
    #    We also combine this with the boundary mask.
    #    valid_mask is True only if:
    #      a) We are within the tensor boundaries
    #      b) The index value is NOT -1
    valid_mask = mask & (index_vals != -1)
    
    target_ptrs = output_B_start + index_vals
    
    tl.store(target_ptrs, src_vals, mask=valid_mask)

def scatter_with_mask(src: torch.Tensor, indices: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
    """
    Scatters src into a new tensor of size `out_size` based on `indices`.
    Ignores any index equal to -1.
    
    Args:
        src: Source tensor (1D for this example)
        indices: Index tensor (same shape as src), contains indices or -1
        out_size: Size of the output tensor
    """
    assert src.is_cuda and indices.is_cuda
    assert src.shape == indices.shape
    
    B, N = src.shape
    
    # Grid configuration
    BLOCK_SIZE = 512
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), B)

    scatter_with_mask_kernel[grid](
        src,
        indices,
        output_tensor,
        src.stride(0),
        indices.stride(0),
        output_tensor.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_tensor