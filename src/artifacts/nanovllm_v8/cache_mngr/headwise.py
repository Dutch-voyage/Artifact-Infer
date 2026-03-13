from src.core.artifact_base import Artifact
from src.core.service_base import BaseService

from ..attention.flashinfer_attention_headflatten import (
    Attention,
    store_kvcache,
    read_kvcache,
    read_q_cache,
)
from src.services.nanovllm_v8.engine.sequence import Sequence, torch_rotl_uint8
import torch

import itertools

from src.services.nanovllm_v8.utils.context import get_context, set_context_replace
from src.services.nanovllm_v8.utils.logging import get_log, set_log, append_item_to_log, append_item_to_seq_log
# all implemntation here

import triton
import triton.language as tl

# Run optimized triton kernel
def grid(batch_size, extend_len, BLOCK_SIZE):
    num_token_blocks = triton.cdiv(extend_len, BLOCK_SIZE)
    return (batch_size, num_token_blocks)

@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


@triton.jit
def write_req_to_token_pool_triton_headwise(
    req_to_token_ptr,  # [max_batch * num_kv_heads, max_context_len * num_kv_heads] # allocated by head
    req_pool_indices,
    seq_lens,
    out_cache_loc, # allocated by token 
    num_kv_heads: tl.constexpr,
    req_to_token_ptr_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_token = tl.program_id(1)

    req_pool_index = tl.load(req_pool_indices + pid_batch)
    # pre_len = tl.load(pre_lens + pid_batch)
    seq_len = tl.load(seq_lens + pid_batch)
    # extend_len = seq_len - pre_len
    extend_len = seq_len
    
    cumsum_start = 0
    for i in range(pid_batch):
        # cumsum_start += tl.load(extend_lens + i)
        cumsum_start += tl.load(seq_lens + i)
    
    token_start = pid_token * BLOCK_SIZE

    offset = tl.arange(0, BLOCK_SIZE)
    actual_offset = token_start + offset
    mask = actual_offset < extend_len
        
    src_ptr = out_cache_loc + cumsum_start + actual_offset
    src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
    value = tl.load(src_ptr, mask=mask)
    
    value_store = value[None, :] * num_kv_heads + tl.arange(0, num_kv_heads)[:, None]
    
    dst_ptr = (
        req_to_token_ptr
        + req_pool_index * req_to_token_ptr_stride * num_kv_heads
        # + actual_offset_store
        + actual_offset
        # + pre_len
    )

    dst_ptr = dst_ptr[None, :] + tl.arange(0, num_kv_heads)[:, None] * req_to_token_ptr_stride

    tl.store(dst_ptr, value_store, mask=mask[None, :])

def write_req_to_token_pool_headwise(
    req_to_token,  # [max_batch, max_context_len]
    req_pool_indices,
    # pre_lens,
    seq_lens,
    # extend_lens,
    out_cache_loc,
    num_kv_heads, 
    batch_size,
    req_pool_stride,
):
    # Run optimized triton kernel
    # max_extend_len = extend_lens.max().item()
    # NOTE: simplify
    max_extend_len = seq_lens.max().item()
    write_req_to_token_pool_triton_headwise[grid(batch_size, max_extend_len, 512 // num_kv_heads)](
        req_to_token,
        req_pool_indices,
        # pre_lens,
        seq_lens,
        # extend_lens,
        out_cache_loc,
        num_kv_heads, 
        req_pool_stride,
        BLOCK_SIZE=512 // num_kv_heads, 
    )

@triton.jit
def read_req_to_token_pool_triton_headwise(
    req_to_token_ptr,  # [max_batch * num_kv_heads, max_context
    req_pool_indices,
    seq_lens, 
    seq_lens_cumsum, 
    out_loc, # allocated by token
    req_to_token_ptr_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_token = tl.program_id(1)

    req_pool_index = tl.load(req_pool_indices + pid_batch)
    # pre_len = tl.load(pre_lens + pid_batch)
    seq_len = tl.load(seq_lens + pid_batch)

    seq_offset = pid_token * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    seq_mask = seq_offset < seq_len
    
    offsets = req_pool_index * req_to_token_ptr_stride + seq_offset
    
    read_indices = tl.load(req_to_token_ptr + offsets, mask=seq_mask)
    
    dst_ptr = out_loc + tl.load(seq_lens_cumsum + pid_batch) + seq_offset

    tl.store(dst_ptr, read_indices, mask=seq_mask)

def read_req_to_token_pool_headwise(
    req_to_token,  # [max_batch, max_context_len]
    req_pool_indices,
    seq_lens,
    batch_size,
    req_pool_stride,
):
    # Run optimized triton kernel
    # max_extend_len = extend_lens.max().item()
    # NOTE: simplify
    max_extend_len = seq_lens.max().item()

    packed_indices = torch.empty(
        seq_lens.sum().item(), dtype=torch.int32, device="cuda"
    )
    read_req_to_token_pool_triton_headwise[grid(batch_size, max_extend_len, 32)](
        req_to_token,
        req_pool_indices,
        seq_lens, 
        torch.cumsum(torch.cat([torch.zeros(1, dtype=seq_lens.dtype, device=seq_lens.device), seq_lens[:-1]]), dim=0),
        packed_indices,
        req_pool_stride,
        BLOCK_SIZE=32, 
    )
    
    return packed_indices

@triton.jit
def gather_req_to_token_pool_triton_headwise(
    req_to_token_ptr,              # [max_batch * num_kv_heads, max_context]
    req_pool_indices_ptr, 
    src_seq_pool_indices_ptr,      # Flat buffer of source indices
    seq_lens_ptr, 
    seq_lens_cumsum_ptr,           # Start index in src_seq_pool_indices_ptr for each batch
    req_to_token_ptr_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_token = tl.program_id(1)

    # 1. Load Batch Info
    req_pool_index = tl.load(req_pool_indices_ptr + pid_batch)
    seq_len = tl.load(seq_lens_ptr + pid_batch)

    # 2. Calculate Offsets
    seq_offset = pid_token * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    seq_mask = seq_offset < seq_len
    
    # 3. Load Source Indices (Where to read from)
    # Ensure seq_lens_cumsum_ptr points to the START of the current batch's indices in the flat buffer
    seq_start_index = tl.load(seq_lens_cumsum_ptr + pid_batch)
    src_seq_pool_indices = tl.load(src_seq_pool_indices_ptr + seq_start_index + seq_offset, mask=seq_mask)
    
    # 4. Calculate Read Pointers
    # Row: req_pool_index, Col: src_seq_pool_indices
    read_offsets = req_pool_index * req_to_token_ptr_stride + src_seq_pool_indices
    read_ptr = req_to_token_ptr + read_offsets # Base + Offset
    
    read_values = tl.load(read_ptr, mask=seq_mask)
    
    # 5. Calculate Write Pointers (Gather to Front)
    # Row: req_pool_index, Col: seq_offset (0, 1, 2...)
    dst_offset = req_pool_index * req_to_token_ptr_stride + seq_offset
    dst_ptr = req_to_token_ptr + dst_offset # [FIX] Add base pointer here
    
    tl.store(dst_ptr, read_values, mask=seq_mask)

def gather_req_to_token_pool_headwise(
    seq_to_token_pool, 
    seq_pool_indices, 
    src_indices, 
    seq_lens, 
    seq_lens_cumsum, 
):  
    seq_lens_cumsum = seq_lens_cumsum.to(torch.int32)
    seq_lens = seq_lens.to(torch.int32)
    for i in range(seq_lens.shape[0]):
        src_index = src_indices[seq_lens_cumsum[i]: seq_lens_cumsum[i] + seq_lens[i]]
        dst_index = torch.arange(0, seq_lens[i], device="cuda")
        seq_to_token_pool[seq_pool_indices[i], dst_index] = seq_to_token_pool[seq_pool_indices[i], src_index]
    # Using a Block Size of 32
    # BLOCK_SIZE = 32s
    
    # gather_req_to_token_pool_triton_headwise[grid(batch_size, max_len, BLOCK_SIZE)](
    #     seq_to_token_pool,
    #     seq_pool_indices,
    #     src_indices, 
    #     seq_lens, 
    #     seq_lens_cumsum.to(torch.int32),
    #     seq_to_token_pool.stride(0),
    #     BLOCK_SIZE=BLOCK_SIZE, 
    # )

def uint8_to_bits(tensor):
    """
    Converts a uint8 tensor to a tensor of bits (0s and 1s).
    The result will have one extra dimension of size 8 at the end.
    """
    # 1. Create a generic sequence of bit positions: [7, 6, 5, 4, 3, 2, 1, 0]
    # We move it to the same device as the input to support GPU usage.
    # bit_indices = torch.arange(7, -1, -1, device=tensor.device, dtype=torch.uint8)
    bit_indices = torch.arange(0, 8, device=tensor.device, dtype=torch.uint8)
    # 2. Expand the input tensor to prepare for broadcasting
    # tensor.unsqueeze(-1) adds a dimension at the end
    # >> bit_indices performs right-shifts for all 8 positions simultaneously
    # & 1 isolates the least significant bit after the shift
    return ((tensor.unsqueeze(-1) >> bit_indices) & 1).view(tensor.shape[0], tensor.shape[1], -1)

@triton.jit
def packed_mask_kernel(
    len_ptr,            # Pointer to input lengths (N,)
    out_ptr,            # Pointer to output mask (N, max_bytes)
    stride_len,         # Stride for lengths
    stride_out_row,     # Stride for output rows
    stride_out_col,     # Stride for output columns (usually 1)
    max_bytes,          # The maximum number of bytes (columns)
    BLOCK_SIZE: tl.constexpr
):
    # 1. Handle Grid Indices
    # We parallelize across rows (pid_row) and chunks of columns (pid_col)
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # 2. Load the length for this specific row
    # Note: We assume len_ptr is 1D here. If 2D, adjust strides accordingly.
    current_len = tl.load(len_ptr + pid_row * stride_len)

    # 3. Compute Column Offsets
    # Which bytes (columns) is this program block responsible for?
    col_offs = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't write past the declared tensor width
    col_mask = col_offs < max_bytes

    # 4. Calculate the Bitmask
    # Each output byte represents 8 bits. 
    # Byte 0: bits 0-7, Byte 1: bits 8-15, etc.
    start_bit_idx = col_offs * 8
    
    # Calculate how many bits in this byte are valid.
    # Logic: 
    # If len=20, start_bit=0  -> diff=20 -> clamp(0, 8) -> 8 bits set (255)
    # If len=20, start_bit=16 -> diff=4  -> clamp(0, 8) -> 4 bits set (15)
    # If len=20, start_bit=24 -> diff=-4 -> clamp(0, 8) -> 0 bits set (0)
    bits_in_byte = current_len - start_bit_idx
    bits_in_byte = tl.maximum(bits_in_byte, 0) # Clamp below 0
    bits_in_byte = tl.minimum(bits_in_byte, 8) # Clamp above 8
    
    # Convert count of bits to an integer mask.
    # Example: 3 bits -> (1 << 3) - 1 = 7 (binary 00000111)
    # Note: We must cast to int32 before shift to avoid overflow issues with smaller types
    val = (1 << bits_in_byte) - 1

    # 5. Store Result
    out_offset = pid_row * stride_out_row + col_offs * stride_out_col
    
    # Cast to uint8 before storing
    tl.store(out_ptr + out_offset, val.to(tl.uint8), mask=col_mask)

def generate_packed_mask(num_block_head: torch.Tensor, max_blocks: int = None):
    """
    Args:
        num_block_head: Tensor of shape (N,) or (Batch, Heads) containing lengths.
        max_blocks: Total blocks to pad to. If None, derived from max value.
    Returns:
        Tensor of shape (..., ceil(max_blocks/8)) with dtype uint8.
    """
    # Flatten input to behave like a list of rows
    original_shape = num_block_head.shape
    num_flat = num_block_head.view(-1).to("cuda")
    n_rows = num_flat.shape[0]

    if max_blocks is None:
        max_blocks = num_block_head.max().item()

    # Calculate required width in bytes (ceil(max_blocks / 8))
    max_bytes = (max_blocks + 7) // 8
    
    # Allocate output
    packed_mask = torch.empty((n_rows, max_bytes), dtype=torch.uint8, device=num_flat.device)
    
    # Kernel Launch Configuration
    BLOCK_SIZE = 128
    
    # Grid: (Rows, Chunks_of_Cols)
    grid = (n_rows, triton.cdiv(max_bytes, BLOCK_SIZE))
    
    packed_mask_kernel[grid](
        num_flat,
        packed_mask,
        num_flat.stride(0),
        packed_mask.stride(0),
        packed_mask.stride(1),
        max_bytes,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape back to (Batch, Heads, Bytes)
    return packed_mask.view(*original_shape, max_bytes)

class CacheManager(BaseService):
    @property
    def name(self):
        return "CacheManagerHeadwise"

    """
    This version of implementation only 
    """

    def __init__(self, attention_backend: Artifact, config, compressor=None):
        super().__init__()

        attention_backend.register(self)
        
        self.enforce_eager = config.enforce_eager
        
        self.num_kv_heads = config.hf_config.num_key_value_heads 
        
        self.head_dim = config.hf_config.head_dim
        
        self.num_layers = config.hf_config.num_hidden_layers
        
        self.max_context_len = config.max_model_len 
        
        self.max_num_seqs = config.max_num_seqs
        
        self.layer_budget = config.layer_budget

        self.cu_seq_pool_id = 0 
        
        self.seq_to_pool_id = {} 

        self.max_num_seqs = config.max_num_seqs
        
        self.max_context_len = config.max_model_len

        self.seq_to_slot_pool = torch.zeros((self.max_num_seqs * self.num_kv_heads, self.max_context_len), dtype=torch.int32, device="cuda")
        
        self.head_indices = torch.arange(0, self.num_kv_heads, dtype=torch.int32, device="cuda")
        
        self.cu_page_indices = self.cu_seq_lens = None
        
        # self.full_headwise_mask_per_token = torch.tensor([2 ** i for i in range(self.num_kv_heads)], device="cuda", dtype=torch.uint8)

        self.compressor = compressor
        
        self.if_fake_compress = config.if_fake_compress
        
        self.mask_arr = {layer_id: [] for layer_id in range(self.num_layers)}
        self.mask_indptr = {layer_id: None for layer_id in range(self.num_layers)}
        self.cu_packed_custom_mask = {layer_id: None for layer_id in range(self.num_layers)}
    
    def reset(self):
        self.seq_to_slot_pool = torch.zeros((self.max_num_seqs * self.num_kv_heads, self.max_context_len), dtype=torch.int32, device="cuda")
        self.mask_arr = {layer_id: [] for layer_id in range(self.num_layers)}
        self.mask_indptr = {layer_id: None for layer_id in range(self.num_layers)}
        self.cu_packed_custom_mask = {layer_id: None for layer_id in range(self.num_layers)}
    
    def _organize(self, seq: Sequence):
        cu_seq_pool_indices = self.seq_to_pool_id[seq.seq_id] * self.num_kv_heads + self.head_indices
        
        cu_seq_pool = self.seq_to_slot_pool[cu_seq_pool_indices, :seq.num_blocks_max_heads]   
        
        prod_pruned_mask = torch.prod(1 - uint8_to_bits(seq.headwise_mask_layer_transpose)[..., :seq.num_blocks_max_heads].to(torch.int32), dim=0)
        
        for head_id in range(self.num_kv_heads):
            prod_pruned_mask[head_id, seq.num_blocks_head[head_id]:] = 0
        
        pruned_indices = torch.nonzero(prod_pruned_mask, as_tuple=True)
        
        pruned_indices_token = cu_seq_pool[pruned_indices] 
        
        if pruned_indices_token.numel() == 0:
            return
        
        # seq.block_id_to_count[(pruned_indices_token // 8).tolist()] += 1
        
        # for block_id in (pruned_indices_token // 8).unique().tolist():
        #     count = seq.block_id_to_count[block_id.item()]
        #     if count == self.num_kv_heads:
        #         self._deallocate_block(block_id)
        #         seq.block_table.remove(block_id)
        #         seq.block_id_to_count.pop(block_id)
        
        kept_mask = torch.isin(cu_seq_pool, pruned_indices_token, invert=True)
        
        for head_id in range(self.num_kv_heads):
            kept_mask[head_id, seq.num_blocks_head[head_id]:] = 0
        
        kept_indices = torch.nonzero(kept_mask).squeeze().to("cuda")
        
        cu_kept_indices = kept_indices[:,0]
        
        cu_num_blocks_shifted_left = torch.cat([torch.zeros((1,), device="cuda", dtype=torch.int32), cu_kept_indices], dim=0)
        
        cu_num_blocks_shifted_right = torch.cat([cu_kept_indices, torch.zeros((1,), device="cuda", dtype=torch.int32)], dim=0)
        
        cu_num_blocks_diffs = cu_num_blocks_shifted_left - cu_num_blocks_shifted_right
        
        cu_num_block_head_cumsum = torch.nonzero(cu_num_blocks_diffs).squeeze(-1)
        
        cu_num_block_head = cu_num_block_head_cumsum - torch.cat([torch.zeros((1,), device="cuda", dtype=torch.int32), cu_num_block_head_cumsum[:-1]], dim=0)
        
        append_item_to_seq_log("num_blocks_head", seq.seq_id, seq.num_blocks_head.int().detach().cpu())
        
        # print(self.seq_to_slot_pool[cu_seq_pool_indices[0], :seq.num_blocks_head[0]])
        
        seq.num_blocks_head = cu_num_block_head

        gather_req_to_token_pool_headwise(
            self.seq_to_slot_pool, 
            self.seq_to_pool_id[seq.seq_id] * self.num_kv_heads + self.head_indices, 
            kept_indices[:, 1], 
            cu_num_block_head, 
            torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32), cu_num_block_head_cumsum[:-1]])
        )
        # print(self.seq_to_slot_pool[cu_seq_pool_indices[0], :seq.num_blocks_head[0]])
        # print('-' * 100)

        cu_headwise_mask_layer_transpose = generate_packed_mask(seq.num_blocks_head)[None, ...].repeat(self.num_layers, 1, 1)
        
        # print(seq.headwise_mask_layer_transpose[0, 0, :])
        
        seq.headwise_mask_layer_transpose = cu_headwise_mask_layer_transpose.contiguous()
        
        # print(seq.headwise_mask_layer_transpose[0, 0, :])
        
        # print("-" * 100)
        
        seq.next_mask = (torch.ones((self.num_kv_heads,), device="cpu", dtype=torch.uint8)) << (seq.num_blocks_head % 8).to("cpu")

        seq.next_mask = seq.next_mask.to(torch.uint8)

    def _rewrite_placeholder(self, seq):
        seq.num_blocks_head = seq.num_blocks_head - 1
        seq.next_mask = (torch.ones((self.num_kv_heads,), device="cpu", dtype=torch.uint8)) << (seq.num_blocks_head % 8)

        seq.next_mask = seq.next_mask.to(torch.uint8)
    
    def _rewrite_organize(self, seq: Sequence):
        # print(seq.num_blocks_head)
        cu_num_block_head = uint8_to_bits(seq.headwise_mask_layer_transpose.to("cuda"))[:, :seq.num_blocks_max_heads].to(torch.int32).sum(dim=-1).max(0).values

        # print(cu_num_block_head)
        # print("-" * 100)
        seq.num_blocks_head = cu_num_block_head 
        seq.headwise_mask_layer_transpose = seq.headwise_mask_layer_transpose[..., :(seq.num_blocks_max_heads - 1) // 8 + 1].contiguous()
        seq.next_mask = ((torch.ones((self.num_kv_heads,), device="cpu", dtype=torch.uint8)) << (seq.num_blocks_max_heads % 8))
        # seq.next_mask = (torch.ones((self.num_kv_heads,), device="cpu", dtype=torch.uint8)) << (seq.num_blocks_head % 8)
        seq.next_mask = seq.next_mask.to(torch.uint8)
        self.update_blocks_post_compression(seq, seq.num_blocks_max_heads)
        return 
    
    def organize(self):
        for seq in self.cu_seqs:
            if self.if_fake_compress:
                self._organize(seq)
            else:
                self._rewrite_organize(seq)
    
    def allocate_prefill_page_indices(self, seqs: list[Sequence]):
        self.cu_seqs = seqs
        for seq_id in seqs:
            self.seq_to_pool_id[seq_id.seq_id] = self.cu_seq_pool_id
            self.cu_seq_pool_id += 1
            self.cu_seq_pool_id %= self.max_num_seqs
        
        self.cu_seqs_to_slot_pool_indices = torch.tensor([self.seq_to_pool_id[seq_id.seq_id] for seq_id in seqs], device="cuda", dtype=torch.int32)
        
        self.len_seqs = len(seqs)
        
        self.seq_lens = torch.tensor(
            [seq.num_blocks_max_heads for seq in self.cu_seqs], #  * self.num_kv_heads, 
            device="cuda"
        ).to(torch.int32)
        
        context = get_context()
        
        cu_slot_mapping = context.slot_mapping

        write_req_to_token_pool_headwise(
            self.seq_to_slot_pool,
            self.cu_seqs_to_slot_pool_indices,
            self.seq_lens, 
            cu_slot_mapping, # 
            self.num_kv_heads, 
            self.len_seqs,
            self.seq_to_slot_pool.stride(0)
        ) 
            
    def update_masks_optimized(self, seqs): 
        self.cu_packed_custom_mask_optimized = torch.cat([seq.headwise_mask_layer_transpose.to(device="cuda", dtype=torch.uint8).view(self.num_layers, -1) for seq in seqs], dim=-1)
        
        context = get_context()
        context.packed_headwise_mask = self.cu_packed_custom_mask_optimized
        # set_context_replace(context)
        
    def allocate_decode_page_indices(self, seqs: list[Sequence]):
        # sglang says when using overlap mode, should not in-place operation, need to investigate, here is non-overlap mode 
        self.cu_seqs = seqs
        
        self.cu_seqs_to_slot_pool_indices = torch.tensor([self.seq_to_pool_id[seq_id.seq_id] for seq_id in seqs], device="cuda", dtype=torch.int32)
        
        self.len_seqs = len(seqs)
        
        self.seq_lens = torch.tensor(
            [seq.num_blocks_max_heads for seq in self.cu_seqs], #  * self.num_kv_heads, 
            device="cuda"
        ).to(torch.int32)

        context = get_context()
        
        cu_slot_mapping = context.slot_mapping
        
        cu_slot_mapping_headwise = ((cu_slot_mapping * self.num_kv_heads)[:, None] + self.head_indices[None, :])
        
        seq_to_pool_indices_headwise = (self.cu_seqs_to_slot_pool_indices[:, None] * self.num_kv_heads) + self.head_indices[None, :]
        
        seq_lens_headwise = ((self.seq_lens - 1)[:, None]) # + head_indices[None, :]
        
        self.seq_to_slot_pool[seq_to_pool_indices_headwise, seq_lens_headwise] = cu_slot_mapping_headwise
        
        self.cu_qo_indptr = torch.arange(self.len_seqs * self.num_kv_heads + 1, device="cuda", dtype=torch.int32) 
        
        self.cu_kv_page_lengths = (self.seq_lens).repeat_interleave(self.num_kv_heads)
        
        self.cu_kv_indptr = torch.zeros(self.len_seqs * self.num_kv_heads + 1, device="cuda", dtype=torch.int32)
        
        self.cu_kv_indptr[1:] = torch.cumsum(self.cu_kv_page_lengths, dim=0)
        
        self.cu_kv_page_indices = torch.empty(
            self.cu_kv_indptr[-1], dtype=torch.int32, device="cuda"
        )
        
        create_flashinfer_kv_indices_triton[(self.len_seqs * self.num_kv_heads,)](
            self.seq_to_slot_pool,
            torch.arange(0, self.len_seqs * self.num_kv_heads, device="cuda", dtype=torch.int32), 
            self.cu_kv_page_lengths, 
            self.cu_kv_indptr,
            None,
            self.cu_kv_page_indices,
            self.seq_to_slot_pool.stride(0),
        )
        
        # self.update_masks(seqs)
        self.update_masks_optimized(seqs)

    def allocate_page_indices_cudagraph(self, seqs: list[Sequence]):
        context = get_context()
        if context.is_prefill:
            self.allocate_prefill_page_indices(seqs)
        else:
            self.allocate_decode_page_indices(seqs)
            
            self.log_occupied_pages(self.cu_kv_page_indices.shape[0])
    
    def allocate_page_indices(self, seqs: list[Sequence]):
        # move to model runner before capturing cuda graph
        self.cu_seqs = seqs
        self.len_seqs = len(seqs)
            
        self.seq_lens = torch.tensor(
            [seq.num_blocks_max_heads for seq in self.cu_seqs], #  * self.num_kv_heads, 
        ).to(torch.int32).to("cuda")
        context = get_context()
        cu_slot_mapping = context.slot_mapping
        
        if context.is_prefill:
            for seq in seqs:
                if seq.seq_id not in self.seq_to_pool_id:
                    self.seq_to_pool_id[seq.seq_id] = self.cu_seq_pool_id
                    self.cu_seq_pool_id += 1
                    self.cu_seq_pool_id %= self.max_num_seqs
            
            self.cu_seqs_to_slot_pool_indices = torch.tensor([self.seq_to_pool_id[seq.seq_id] for seq in seqs], device="cuda", dtype=torch.int32)

            write_req_to_token_pool_headwise(
                self.seq_to_slot_pool,
                self.cu_seqs_to_slot_pool_indices,
                self.seq_lens, 
                cu_slot_mapping, # 
                self.num_kv_heads, 
                self.len_seqs,
                self.seq_to_slot_pool.stride(0)
            ) 
            
        else:
            self.cu_seqs_to_slot_pool_indices = torch.tensor([self.seq_to_pool_id[seq.seq_id] for seq in seqs], device="cuda", dtype=torch.int32)

            cu_slot_mapping_headwise = ((cu_slot_mapping * self.num_kv_heads)[:, None] + self.head_indices[None, :])
            
            seq_to_pool_indices_headwise = (self.cu_seqs_to_slot_pool_indices[:, None] * self.num_kv_heads) + self.head_indices[None, :]
            
            seq_lens_headwise = torch.cat([seq.num_blocks_head.unsqueeze(0) for seq in seqs]) - 1
            
            self.seq_to_slot_pool[seq_to_pool_indices_headwise, seq_lens_headwise] = cu_slot_mapping_headwise
            
            headwise_seq_pool_id = torch.tensor([self.seq_to_pool_id[seq.seq_id] for seq in seqs], device="cuda", dtype=torch.int32)[:, None] * self.num_kv_heads + self.head_indices[None, :]
            
            headwise_seq_lens = torch.cat([seq.num_blocks_head for seq in seqs]).to("cuda")
            
            self.cu_kv_page_indices = read_req_to_token_pool_headwise(
                self.seq_to_slot_pool,
                headwise_seq_pool_id.reshape(-1),
                headwise_seq_lens, 
                len(seqs) * self.num_kv_heads,
                self.seq_to_slot_pool.stride(0)
            )
                        
            self.cu_kv_indptr = torch.cumsum(
                torch.cat([torch.tensor([0], device="cuda"), headwise_seq_lens], dim=0),
                dim=0, dtype=torch.int32
            )
            
            self.cu_kv_last_page_lens = torch.tensor(
                [1] * (len(seqs) * self.num_kv_heads), device="cuda", dtype=torch.int32
            )

            self.cu_qo_indptr = torch.arange(len(seqs) * self.num_kv_heads + 1, device="cuda", dtype=torch.int32) # .to(torch.int32)
            
            self.cu_packed_custom_mask_optimized = torch.cat([seq.headwise_mask_layer_transpose.reshape(self.num_layers, -1).to(device="cuda", dtype=torch.uint8) for seq in seqs], dim=-1)
                        
            context = get_context()

            if self.enforce_eager:
                context.packed_headwise_mask = self.cu_packed_custom_mask_optimized
            else:
                context.packed_headwise_mask[:, :self.cu_packed_custom_mask_optimized.shape[1]].copy_(self.cu_packed_custom_mask_optimized)
            
            self.log_occupied_pages(self.cu_kv_page_indices.shape[0])

    def log_occupied_pages(self, occupied_pages):
        log = get_log()
        log.occupied_pages = occupied_pages
        set_log(log)
        
    def update_indices(self):        
        if get_context().is_prefill:
            self.prepare_metadata_for_attn_prefill(
                self.cu_seqs
            )
        else:
            self.prepare_metadata_for_attn_decode(
                self.cu_qo_indptr, 
                self.cu_kv_indptr, 
                self.cu_kv_page_indices, 
                self.cu_packed_custom_mask_optimized[0] 
            )

    def update_indices_capture(self, bs: int):
        self.init_forward_metadata_capture_cuda_graph(
            bs,
            self.cu_qo_indptr, 
            self.cu_kv_indptr, 
            self.cu_kv_page_indices,
            self.cu_packed_custom_mask_optimized[0]
        )

    def update_indices_replay(self, bs: int):
        self.init_forward_metadata_replay_cuda_graph(
            bs,
            self.cu_qo_indptr, 
            self.cu_kv_indptr, 
            self.cu_kv_page_indices,
            self.cu_packed_custom_mask_optimized[0]
        )
    
    def _read_and_store_cache_fakecompress(self, q_cache, k_cache, v_cache, layer_id):
        for seq in self.cu_seqs:
            
            cu_seq_pool_indices = self.seq_to_pool_id[seq.seq_id] * self.num_kv_heads + self.head_indices
            slot_mappings = (self.seq_to_slot_pool[cu_seq_pool_indices, :seq.num_blocks_max_heads] // self.num_kv_heads).view(-1)
            
            query_slot_mapping = torch.tensor([seq.query_block_id], device="cuda").to(torch.int32)
            
            query = read_q_cache(
                q_cache=q_cache,
                query_slot_mapping=query_slot_mapping,
            )
            key, value = read_kvcache(
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=slot_mappings,
                # num_kv_heads=self.num_kv_heads,
                num_kv_heads=1, 
                head_dim=self.head_dim
            )
            
            key = key.view(-1, self.num_kv_heads, 1, self.head_dim).squeeze(-2)
            value = value.view(-1, self.num_kv_heads, 1, self.head_dim).squeeze(-2)
            
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            
            ret = self.compressor.update_kv(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                # effective_mask=uint8_to_bits(seq.headwise_mask_layer_transpose[layer_id]).to("cuda").view(self.num_kv_heads, -1)[:, :len(seq.block_table)].to(torch.bool), 
                effective_mask=uint8_to_bits(seq.headwise_mask_layer_transpose[layer_id]).to("cuda").view(self.num_kv_heads, -1)[:, :seq.num_blocks_max_heads].to(torch.bool), 
                seq_id=seq.seq_id,
                layer_id=layer_id,
            )
            
            updated_k = ret["key_states"]
            updated_v = ret["value_states"]
            
            if "packed_selected_mask" in ret:
                if self.if_fake_compress:
                    seq.headwise_mask_layer_transpose[layer_id] = (seq.headwise_mask_layer_transpose[layer_id].clone().to("cuda") & ret["packed_selected_mask"])
                else:
                    packed_selected_mask_full = torch.zeros_like(seq.headwise_mask_layer_transpose[layer_id], device="cuda")
                    packed_selected_mask_full[:, :ret["packed_selected_mask"].shape[-1]] = ret["packed_selected_mask"].clone()
                    seq.headwise_mask_layer_transpose[layer_id] = packed_selected_mask_full
                
                            
            if not self.if_fake_compress:
                if "num_blocks_this_layer" not in ret:
                    return 
                assert ret["num_blocks_this_layer"] == updated_k.shape[0], f"num_blocks_this_layer {ret['num_blocks_this_layer']} vs updated_k {updated_k.shape}"
                slot_mappings_packed_store = torch.tensor(seq.block_table[:ret["num_blocks_this_layer"]], device="cuda").to(torch.int32)

                store_kvcache(
                    key=updated_k.contiguous(),
                    value=updated_v.contiguous(),
                    k_cache=k_cache.contiguous(),
                    v_cache=v_cache.contiguous(),
                    slot_mapping=slot_mappings_packed_store,
                )
            
        return 
    
    
    def _read_and_store_cache(self, q_cache, k_cache, v_cache, layer_id):
        for seq in self.cu_seqs:
            slot_mappings = torch.tensor(seq.block_table, device="cuda").to(torch.int32)
                        
            query_slot_mapping = torch.tensor([seq.query_block_id], device="cuda").to(torch.int32)
            
            query = read_q_cache(
                q_cache=q_cache,
                query_slot_mapping=query_slot_mapping,
            )
            
            key, value = read_kvcache(
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=slot_mappings,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim
            )
            
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            
            ret = self.compressor.update_kv(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                effective_mask=uint8_to_bits(seq.headwise_mask_layer_transpose[layer_id]).to("cuda").view(self.num_kv_heads, -1)[:, :len(seq.block_table)].to(torch.bool), 
                seq_id=seq.seq_id,
                layer_id=layer_id,
            )
            
            updated_k = ret["key_states"]
            updated_v = ret["value_states"]
            
            if "packed_selected_mask" in ret:
                if self.if_fake_compress:
                    seq.headwise_mask_layer_transpose[layer_id] = (seq.headwise_mask_layer_transpose[layer_id].clone().to("cuda") & ret["packed_selected_mask"])
                else:
                    packed_selected_mask_full = torch.zeros_like(seq.headwise_mask_layer_transpose[layer_id], device="cuda")
                    packed_selected_mask_full[:, :ret["packed_selected_mask"].shape[-1]] = ret["packed_selected_mask"].clone()
                    seq.headwise_mask_layer_transpose[layer_id] = packed_selected_mask_full
                
                            
            if not self.if_fake_compress:
                if "num_blocks_this_layer" not in ret:
                    return 
                assert ret["num_blocks_this_layer"] == updated_k.shape[0], f"num_blocks_this_layer {ret['num_blocks_this_layer']} vs updated_k {updated_k.shape}"
                slot_mappings_packed_store = torch.tensor(seq.block_table[:ret["num_blocks_this_layer"]], device="cuda").to(torch.int32)

                store_kvcache(
                    key=updated_k.contiguous(),
                    value=updated_v.contiguous(),
                    k_cache=k_cache.contiguous(),
                    v_cache=v_cache.contiguous(),
                    slot_mapping=slot_mappings_packed_store,
                )
            
        return 
                 
    def _packed_read_and_store_cache(self, q_cache, k_cache, v_cache, layer_id):
        slot_mappings_packed = torch.cat([self.seq_to_slot_pool[self.seq_to_pool_id[seq.seq_id], :seq.num_blocks_max_heads] for seq in self.cu_seqs], dim=0).to(torch.int32)
        
        slot_mappings_packed_store = torch.cat([self.seq_to_slot_pool[self.seq_to_pool_id[seq.seq_id], :self.layer_budget] for seq in self.cu_seqs], dim=0).to(torch.int32)
        
        seq_lens = torch.tensor([0] + [seq.num_blocks_max_heads for seq in self.cu_seqs], device="cuda").to(
            torch.int32
        )
        seq_lens_cumsum = torch.cumsum(seq_lens, dim=0)
        
        query_slot_mapping = [seq.query_block_id for seq in self.cu_seqs]
        query_slot_mapping_tensor = torch.tensor(query_slot_mapping, device="cuda").to(
            torch.int32
        )
        
        query = read_q_cache(
            q_cache=q_cache,
            query_slot_mapping=query_slot_mapping_tensor,
        )
        
        key, value = read_kvcache(
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=slot_mappings_packed,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim
        )
        
        updated_k_list = []
        updated_v_list = []
        
        for i, seq in enumerate(self.cu_seqs):
            query_i = query[i: i + 1, :, :, :]
            key_i = key[seq_lens_cumsum[i]: seq_lens_cumsum[i + 1], :, :].unsqueeze(0)
            value_i = value[ seq_lens_cumsum[i]: seq_lens_cumsum[i + 1], :, :].unsqueeze(0)
            
            ret = self.compressor.update_kv(
                query_i.transpose(1, 2),
                key_i.transpose(1, 2),
                value_i.transpose(1, 2),
                layer_id,
            )
            
            updated_k = ret["key_states"]
            updated_v = ret["value_states"]
            
            if "packed_selected_mask" in ret:
                seq.headwise_mask_layer_transpose[layer_id] = (seq.headwise_mask_layer_transpose[layer_id].clone().to("cuda") & ret["packed_selected_mask"]).contiguous()
            
            updated_k_list.append(updated_k.transpose(1, 2).squeeze(0).contiguous())
            updated_v_list.append(updated_v.transpose(1, 2).squeeze(0).contiguous())
        
        if self.if_fake_compress:
            return 
        
        key = torch.cat(updated_k_list, dim=0)
        value = torch.cat(updated_v_list, dim=0)
        
        store_kvcache(
            key=key,
            value=value,
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=slot_mappings_packed_store,
        )
    
    def read_and_store_cache(self, q_cache, k_cache, v_cache, layer_id):
        # return self._packed_read_and_store_cache(q_cache, k_cache, v_cache, layer_id)
        if self.if_fake_compress:
            return self._read_and_store_cache_fakecompress(q_cache, k_cache, v_cache, layer_id)
        else:
            return self._read_and_store_cache(q_cache, k_cache, v_cache, layer_id)
