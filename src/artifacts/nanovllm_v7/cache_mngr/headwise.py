from src.core.artifact_base import Artifact
from src.core.service_base import BaseService

from ..attention.flashinfer_attention_headflatten import (
    Attention,
    store_kvcache,
    read_kvcache,
    read_q_cache,
)
from src.services.nanovllm_v7.engine.sequence import Sequence
import torch

import itertools

from src.services.nanovllm_v7.utils.context import get_context, set_context_replace
from src.services.nanovllm_v7.utils.logging import get_log, set_log
# all implemntation here

import triton
import triton.language as tl

def reverse_bits_uint8(x: torch.Tensor) -> torch.Tensor:
        """Helper: Reverses bits (00000001 -> 10000000) for a uint8 tensor."""
        x = ((x & 0xF0) >> 4) | ((x & 0x0F) << 4)
        x = ((x & 0xCC) >> 2) | ((x & 0x33) << 2)
        x = ((x & 0xAA) >> 1) | ((x & 0x55) << 1)
        return x

def transpose_uint8_bits(x: torch.Tensor, original_length=None) -> torch.Tensor:
    """
    Transposes a uint8 array of shape (N,) into shape (8, ceil(N/8)).
    
    Conceptually:
    - Input: N bytes (N rows, 8 columns of bits)
    - Output: 8 rows, ceil(N/8) columns (packed as bytes)
    """
    N = x.numel()
    
    # 1. Pad input to be a multiple of 8 bytes (required for int64 view)
    pad_amt = (8 - (N % 8)) % 8
    if pad_amt > 0:
        x = torch.cat([x, torch.zeros(pad_amt, dtype=torch.uint8, device=x.device)])
        
    # 2. View as int64 blocks to process 8x8 bit blocks in parallel
    #    We use int64 because it allows us to SWAR transpose 64 bits at once.
    blocks = x.view(torch.int64)
    
    # 3. Perform 8x8 Bit Matrix Transpose (SWAR Algorithm)
    #    This transposes the bits within every 64-bit block.
    #    Note: We use signed int64 masks, but the logic holds for bit patterns.
    
    # Swap 1x1 adjacent bits
    # Mask selects bits 1, 3, 5... of every byte
    m1 = 0x00AA00AA00AA00AA 
    t = (blocks ^ (blocks >> 7)) & m1
    blocks = blocks ^ t ^ (t << 7)

    # Swap 2x2 adjacent bit-pairs
    # Mask selects bits 2,3, 6,7... of every byte
    m2 = 0x0000CCCC0000CCCC
    t = (blocks ^ (blocks >> 14)) & m2
    blocks = blocks ^ t ^ (t << 14)

    # Swap 4x4 adjacent bit-nibbles
    # Mask selects upper nibble of alternating bytes
    m4 = 0x00000000F0F0F0F0
    t = (blocks ^ (blocks >> 28)) & m4
    blocks = blocks ^ t ^ (t << 28)

    # 4. Unpack and Reshape
    #    At this point, 'blocks' contains transposed 8x8 grids.
    #    However, in memory, they are interleaved.
    #    We view back as uint8 -> shape (N_padded,)
    #    Reshape to (num_blocks, 8) -> Transpose to (8, num_blocks)
    
    #    Output Row 0 corresponds to Byte 0 of every block.
    #    Output Row 1 corresponds to Byte 1 of every block.
    #    etc.
    if original_length is None:
        output = blocks.view(torch.uint8).reshape(-1, 8).t()
    else:
        output = blocks.view(torch.uint8).flatten()[:original_length]
        
    return output

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
def write_req_to_token_pool_triton_optimize_headwise(
    req_to_token_ptr,  # [max_batch * num_kv_heads, max_context_len * num_kv_heads] # allocated by head
    req_pool_indices,
    # pre_lens,
    seq_lens,
    # extend_lens,
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
    
    # offset_store = offset[None, :] # + tl.arange(0, num_kv_heads)[:, None]
    # actual_offset_store = token_start + offset_store
    # mask_store = actual_offset_store < extend_len
    
    # actual_offset_store should be (num_kv_heads, BLOCK_SIZE)
    
    # actual_offset_store = tl.reshape(actual_offset_store, (BLOCK_SIZE), can_reorder=True)
    
    # mask_store = tl.broadcast_to(tl.reshape(mask_store, (BLOCK_SIZE * num_kv_heads), can_reorder=True)[None, :], (num_kv_heads, BLOCK_SIZE * num_kv_heads))
    
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
    # dst_ptr = tl.max_contiguous(tl.multiple_of(dst_ptr, BLOCK_SIZE), num_kv_heads * BLOCK_SIZE)

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
    write_req_to_token_pool_triton_optimize_headwise[grid(batch_size, max_extend_len, 512 // num_kv_heads)](
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
        
        self.num_kv_heads = config.hf_config.num_key_value_heads 
        
        self.head_dim = config.hf_config.head_dim
        
        self.num_layers = config.hf_config.num_hidden_layers
        
        self.max_context_len = config.max_model_len 
        
        self.max_num_seqs = config.max_num_seqs
        
        self.layer_budget = config.layer_budget

        self.cu_seq_pool_id = 0 
        
        self.seq_to_pool_id = {}

        self.seq_to_slot_pool = torch.zeros((config.max_num_seqs * self.num_kv_heads, config.max_model_len * self.num_kv_heads), dtype=torch.int32, device="cuda")
        
        self.cu_page_indices = self.cu_seq_lens = None
        
        self.full_headwise_mask_per_token = torch.tensor([2 ** i for i in range(self.num_kv_heads)], device="cuda", dtype=torch.uint8)

        self.compressor = compressor
        
        self.if_fake_compress = config.if_fake_compress
        
        self.mask_arr = {layer_id: [] for layer_id in range(self.num_layers)}
        self.mask_indptr = {layer_id: None for layer_id in range(self.num_layers)}
        self.cu_packed_custom_mask = {layer_id: None for layer_id in range(self.num_layers)}
        
    def allocate_prefill_page_indices(self, seqs: list[Sequence]):
        self.cu_seqs = seqs
        # print(self.cu_seqs[0].block_table)
        for seq_id in seqs:
            self.seq_to_pool_id[seq_id.seq_id] = self.cu_seq_pool_id
            self.cu_seq_pool_id += 1
            self.cu_seq_pool_id %= self.max_num_seqs
        
        self.cu_seqs_to_slot_pool_indices = torch.tensor([self.seq_to_pool_id[seq_id.seq_id] for seq_id in seqs], device="cuda", dtype=torch.int32)
        
        self.len_seqs = len(seqs)
        
        self.seq_lens = torch.tensor(
            [len(seq.block_table) for seq in self.cu_seqs], #  * self.num_kv_heads, 
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
            self.max_context_len * self.num_kv_heads,
        ) 
    
    def update_masks(self, seqs):
        for layer_id in range(self.num_layers):
            seq_bit_mask = [torch.tensor(seq.headwise_mask_layer[layer_id], device="cuda", dtype=torch.uint8).transpose(0, 1).contiguous().view(-1) for seq in seqs]

            self.mask_arr[layer_id] = [transpose_uint8_bits(m) for m in seq_bit_mask]
            
            self.mask_indptr[layer_id] = torch.cumsum(torch.tensor([0] + list(itertools.chain(*[[m.shape[-1]] * self.num_kv_heads for m in self.mask_arr[layer_id]])), device="cuda"), dim=0)
                        
            self.cu_packed_custom_mask[layer_id] = torch.cat([m.contiguous().view(-1) for m in self.mask_arr[layer_id]], dim=0)
        
    def _optimized_transpose_pad_batch(self, inputs: list[list[int]]):
        """
        Optimized helper to handle:
        1. CPU Tensor creation
        2. Padding to 8-byte boundary (required for SWAR)
        3. Flattening and Batching
        """
        # 1. Pre-calculate sizes to allocate memory once
        # Assuming input m is shaped (heads, seq_len) or similar linear list
        # We need to replicate: torch.tensor(m).transpose(0,1).contiguous().view(-1)
        # We assume the input list `m` is row-major on CPU. 
        # Note: If inputs are pure python lists, we flatten them efficiently.
        
        flat_tensors = []
        padded_lengths = []
        original_lengths = []

        # We stick to CPU for list processing to avoid CUDA sync overhead
        for m in inputs:
            # Create tensor on CPU. 
            # Dtype uint8 is crucial. 
            # Note: We must match the original .transpose(0,1) logic.
            # Assuming m is a list of lists:
            t = torch.tensor(m, dtype=torch.uint8, device="cuda")
            if t.ndim == 2:
                t = t.transpose(0, 1)
            t = t.contiguous().view(-1)
            
            original_len = t.numel()
            original_lengths.append(original_len)
            
            # Pad to 8 bytes for SWAR algo
            pad_amt = (8 - (original_len % 8)) % 8
            if pad_amt > 0:
                t = torch.cat([t, torch.zeros(pad_amt, dtype=torch.uint8, device="cuda")])
            
            flat_tensors.append(t)
            padded_lengths.append(t.numel())

        # 2. Create one large pinned/contiguous buffer
        batch_input = torch.cat(flat_tensors)
        
        return batch_input, padded_lengths, original_lengths

    def _swar_transpose_batch(self, blocks: torch.Tensor):
        """
        Applies SWAR bit-transpose in-place on a large Int64 view.
        """
        # View as int64 to process 8x8 bit blocks in parallel
        blocks = blocks.view(torch.int64)

        # Mask selects bits 1, 3, 5... of every byte
        m1 = 0x00AA00AA00AA00AA
        t = (blocks ^ (blocks >> 7)) & m1
        blocks = blocks ^ t ^ (t << 7)

        # Mask selects bits 2,3, 6,7... of every byte
        m2 = 0x0000CCCC0000CCCC
        t = (blocks ^ (blocks >> 14)) & m2
        blocks = blocks ^ t ^ (t << 14)

        # Mask selects upper nibble of alternating bytes
        m4 = 0x00000000F0F0F0F0
        t = (blocks ^ (blocks >> 28)) & m4
        blocks = blocks ^ t ^ (t << 28)
        
        return blocks

    def update_masks_optimized(self, seqs):
        for layer_id in range(self.num_layers):
            # 1. Gather raw data on CPU
            raw_masks = [seq.headwise_mask_layer[layer_id] for seq in seqs]
            
            # 2. Prepare batched input (CPU -> GPU)
            batch_input_gpu, padded_lens, original_lens = self._optimized_transpose_pad_batch(raw_masks)
            
            # Move to GPU once per layer
            # batch_input_gpu = batch_input_cpu.to(device="cuda", non_blocking=True)
            
            # 3. Run Vectorized SWAR (Bit Transpose)
            transposed_blocks = self._swar_transpose_batch(batch_input_gpu)
            
            # 4. Unpack and Slice
            # The SWAR output is packed 64-bit blocks. We view as uint8.
            # Shape: (Total_Blocks, 8)
            unpacked_bytes = transposed_blocks.view(torch.uint8).view(-1, 8)
            
            final_masks = []
            
            cursor = 0
            for p_len in padded_lens:
                num_blocks = p_len // 8
                
                # Extract the blocks belonging to this sequence
                seq_blocks = unpacked_bytes[cursor : cursor + num_blocks]
                
                # Transpose to get (8, num_blocks)
                # CHANGE: Removed .view(-1). We keep it 2D (8, N) to match original semantics.
                transposed = seq_blocks.t().contiguous() 
                
                final_masks.append(transposed)
                cursor += num_blocks
            
            self.mask_arr[layer_id] = final_masks
            
            # 5. Indptr and Packing
            # Since m is now (8, N), m.shape[-1] is N (bytes per head). 
            # This is correct. The previous flattened version made this 8*N (wrong).
            
            sizes = []
            for m in final_masks:
                # m.shape[-1] correctly grabs the width (seq_len_packed)
                sizes.extend([m.shape[-1]] * self.num_kv_heads)
                
            self.mask_indptr[layer_id] = torch.cumsum(
                torch.tensor([0] + sizes, device="cuda"), dim=0
            )

            # Final Concatenation
            # We flatten here just for the packed mask, which is safe.
            self.cu_packed_custom_mask[layer_id] = torch.cat([m.view(-1) for m in final_masks], dim=0)

    def allocate_decode_page_indices(self, seqs: list[Sequence]):
        # sglang says when using overlap mode, should not in-place operation, need to investigate, here is non-overlap mode 
        self.cu_seqs = seqs
        
        # print(f"in allocate {id(self.cu_seqs[0])}")
        # print(f"in allocate len {self.cu_seqs[0].block_table}")

        self.cu_seqs_to_slot_pool_indices = torch.tensor([self.seq_to_pool_id[seq_id.seq_id] for seq_id in seqs], device="cuda", dtype=torch.int32)
        
        self.len_seqs = len(seqs)
        
        self.seq_lens = torch.tensor(
            [len(seq.block_table) for seq in self.cu_seqs], #  * self.num_kv_heads, 
            device="cuda"
        ).to(torch.int32)

        context = get_context()
        
        cu_slot_mapping = context.slot_mapping
        
        head_indices = torch.arange(0, self.num_kv_heads, dtype=torch.int32, device="cuda")
        
        cu_slot_mapping_headwise = ((cu_slot_mapping * self.num_kv_heads)[:, None] + head_indices[None, :])
        
        seq_to_pool_indices_headwise = (self.cu_seqs_to_slot_pool_indices[:, None] * self.num_kv_heads) + head_indices[None, :]
        
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
            self.seq_to_slot_pool.shape[1],
        )
        
        # self.update_masks(seqs)
        self.update_masks_optimized(seqs)

        context = get_context()
        context.packed_headwise_mask = self.cu_packed_custom_mask
        context.mask_indptr = self.mask_indptr
        set_context_replace(context)

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
        
        self.seq_lens = torch.tensor(
            [len(seq.block_table) for seq in self.cu_seqs]
        ).to(torch.int32)
        
        headwise_seq_table = list(itertools.chain(*[seq.get_headwise_block_table() for seq in seqs]))
        
        self.cu_kv_indptr = torch.cumsum(
            torch.tensor([0] + [len(table) for table in headwise_seq_table], device="cuda"),
            dim=0, dtype=torch.int32
        )
        
        self.cu_kv_page_indices = torch.tensor(
            list(itertools.chain(*headwise_seq_table)), device="cuda", dtype=torch.int32
        )
        
        self.cu_kv_last_page_lens = torch.tensor(
            [1] * (len(seqs) * self.num_kv_heads), device="cuda", dtype=torch.int32
        )

        self.cu_qo_indptr = torch.arange(len(seqs) * self.num_kv_heads + 1, device="cuda", dtype=torch.int32)# .to(torch.int32)

        
        
        for layer_id in range(self.num_layers):
            seq_bit_mask = [torch.tensor(seq.headwise_mask_layer[layer_id], device="cuda", dtype=torch.uint8).transpose(0, 1).contiguous().view(-1) for seq in seqs]
            
            self.mask_arr[layer_id] = [transpose_uint8_bits(m) for m in seq_bit_mask]
            
            self.mask_indptr[layer_id] = torch.cumsum(torch.tensor([0] + list(itertools.chain(*[[m.shape[-1]] * self.num_kv_heads for m in self.mask_arr[layer_id]])), device="cuda"), dim=0)
                        
            self.cu_packed_custom_mask[layer_id] = torch.cat([m.contiguous().view(-1) for m in self.mask_arr[layer_id]], dim=0)
        
        context = get_context()
        context.packed_headwise_mask = self.cu_packed_custom_mask
        context.mask_indptr = self.mask_indptr
        set_context_replace(context)
        
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
                self.cu_seqs
            )

    def update_indices_capture(self, bs: int):
        self.init_forward_metadata_capture_cuda_graph(
            bs,
            self.cu_qo_indptr, 
            self.cu_kv_indptr, 
            self.cu_kv_page_indices,
            self.cu_packed_custom_mask[0]
        )

    def update_indices_replay(self, bs: int):
        self.init_forward_metadata_replay_cuda_graph(
            bs,
            self.cu_qo_indptr, 
            self.cu_kv_indptr, 
            self.cu_kv_page_indices,
            self.cu_packed_custom_mask[0]
        )
    
    def read_and_store_cache_iterative(self, q_cache, k_cache, v_cache, layer_id):
        total_steps = (len(self.cu_seqs[0].block_table) - self.compressor.budget) // self.config.steps_between_cache_compressions + 1
        for i in range(total_steps):
            start = i * self.config.steps_between_cache_compressions + self.compressor.budget
            end = min(start + self.config.steps_between_cache_compressions, len(self.cu_seqs[0].block_table))
            slot_mappings = self.cu_seqs[0].block_table[:self.compressor.budget] + self.cu_seqs[0].block_table[start:end]
            assert len(self.cu_seqs) == 1, "Currently only support single request"

            slot_mappings_tensor = torch.tensor(slot_mappings, device="cuda").to(
                torch.int32
            )
            
            query_slot_mapping = [self.cu_seqs[0].query_block_id]

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
                slot_mapping=slot_mappings_tensor,
                num_kv_heads=self.num_kv_heads, 
                head_dim=self.head_dim
            )
            
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

            updated_k, updated_v = self.compressor.update_kv(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                self.cu_seqs[0],
                layer_id, 
            )

            key = updated_k.transpose(1, 2).squeeze(0).contiguous()
            value = updated_v.transpose(1, 2).squeeze(0).contiguous()

            slot_mappings_tensor = slot_mappings_tensor[: key.shape[0]]
            
            store_kvcache(
                key=key,
                value=value,
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=slot_mappings_tensor,
            )
    
    def _packed_read_and_store_cache(self, q_cache, k_cache, v_cache, layer_id):        
        # print(f"in compress {id(self.cu_seqs[0])}")
        # print(f"in compress len {self.cu_seqs[0].block_table}")
        slot_mappings_packed = list(
            itertools.chain(
                *[seq.block_table for seq in self.cu_seqs]
            )
        )
                
        slot_mappings_packed_store = list(
            itertools.chain(
                *[seq.block_table[:self.layer_budget] for seq in self.cu_seqs]
            )
        )
        
        slot_mappings_packed = torch.tensor(slot_mappings_packed, device="cuda").to(
            torch.int32
        )
        
        slot_mappings_packed_store = torch.tensor(slot_mappings_packed_store, device="cuda").to(
            torch.int32
        )
        
        seq_lens = torch.tensor([0] + [len(seq.block_table) for seq in self.cu_seqs], device="cuda").to(
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
        
        for i in range(len(self.cu_seqs)):
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
                self.mask_arr[layer_id][i] = self.mask_arr[layer_id][i].clone() & ret["packed_selected_mask"]
                # print(ret["packed_selected_mask"])
                headwise_mask_i = transpose_uint8_bits(self.mask_arr[layer_id][i].transpose(0, 1).reshape(-1), original_length=len(self.cu_seqs[i].headwise_mask_layer[layer_id]))          
                # print(headwise_mask_i)
                self.cu_seqs[i].headwise_mask_layer[layer_id] = headwise_mask_i.unsqueeze(-1).tolist()
                            
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
        return self._read_and_store_cache(q_cache, k_cache, v_cache, layer_id)
    
    def _read_and_store_cache(self, q_cache, k_cache, v_cache, layer_id):
        """
        option 1: per-sequence handling

        option 2: like flashinfer's layout, handling with packed indices,
        """
        slot_mappings = self.cu_seqs[0].block_table

        assert len(self.cu_seqs) == 1, "Currently only support single request"

        slot_mappings_tensor = torch.tensor(slot_mappings, device="cuda").to(
            torch.int32
        )
        
        query_slot_mapping = [self.cu_seqs[0].query_block_id]

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
            slot_mapping=slot_mappings_tensor,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim
        )
        
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        
        ret = self.compressor.update_kv(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            self.cu_seqs[0],
            layer_id, 
        )
        
        if self.if_fake_compress:
            return 
        
        updated_k = ret["key_states"]
        updated_v = ret["value_states"]

        key = updated_k.transpose(1, 2).squeeze(0).contiguous()
        value = updated_v.transpose(1, 2).squeeze(0).contiguous()
        
        slot_mappings_tensor = slot_mappings_tensor[: key.shape[0]]
        
        store_kvcache(
            key=key,
            value=value,
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=slot_mappings_tensor,
        )
        
        self.update_blocks_post_compression(self.cu_seqs[0], self.layer_budget)
