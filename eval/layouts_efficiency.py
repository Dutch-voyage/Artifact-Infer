from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    BatchDecodeWithPagedKVCacheWrapper,
)

from flashinfer.sparse import BlockSparseAttentionWrapper
from flashinfer.quantization import segment_packbits
import itertools
import torch
import numpy as np
import triton
import triton.language as tl 

# LS: layerwise selection
# LA: layerwise allocation
# HS: headwise selection
# HA: headwise allocation

torch.set_printoptions(profile="full")

device = "cuda:0"

num_blocks = 2048 * 128

num_kv_heads = 8
num_heads = 32
num_layers = 36
head_dim = 128

def page_partial_update_mask(self, packed_custom_mask):
    # self._custom_mask_buf = packed_custom_mask
    self._custom_mask_buf = packed_custom_mask.to(
        self.device, non_blocking=True
    )

def bsr_partial_update_mask(self, packed_mask):
    # self._packed_mask_buf = packed_mask
    self._packed_mask_buf = packed_mask.to(
        self.device, non_blocking=True
    )

BatchPrefillWithPagedKVCacheWrapper.partial_update_custom_mask = page_partial_update_mask
BlockSparseAttentionWrapper.partial_update_custom_mask = bsr_partial_update_mask

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
def _inplace_rewrite_flattened(
    k_cache_ptr,
    v_cache_ptr,
    src_page_indptr,  # The actual indices of the pages
    dst_page_indptr,
    total_elements,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # This PID now represents a chunk of the GLOBAL workload
    pid = tl.program_id(0)
    
    # Identify the range of elements this block is responsible for
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # 1. We need to find which page each 'offset' belongs to.
    # For simplicity in Triton, we can pass a 'page_id_map' 
    # which maps every BLOCK_SIZE chunk to a specific page and intra-page offset.
    # Alternatively, we calculate it here if the mapping is simple.
    
    # Let's assume we pass a pre-computed mapping for maximum speed:
    # chunk_to_page_map: [num_chunks] -> which page_id
    # chunk_to_offset_map: [num_chunks] -> which offset inside that page
    
    # (Simplified approach for logic):
    # Every element in this block belongs to a global index. 
    # We load the source/destination index for this specific global element.
    src_idx = tl.load(src_page_indptr + offsets, mask=mask)
    dst_idx = tl.load(dst_page_indptr + offsets, mask=mask)

    # 2. Standard KV movement logic
    d_cols = tl.arange(0, D)[None, :]
    
    # Load K
    k_src_ptrs = k_cache_ptr + src_idx[:, None] * D + d_cols
    k_data = tl.load(k_src_ptrs, mask=mask[:, None])
    
    # Store K
    k_dst_ptrs = k_cache_ptr + dst_idx[:, None] * D + d_cols
    tl.store(k_dst_ptrs, k_data, mask=mask[:, None])

    # Load V
    v_src_ptrs = v_cache_ptr + src_idx[:, None] * D + d_cols
    v_data = tl.load(v_src_ptrs, mask=mask[:, None])
    
    # Store V
    v_dst_ptrs = v_cache_ptr + dst_idx[:, None] * D + d_cols
    tl.store(v_dst_ptrs, v_data, mask=mask[:, None])        

def inplace_rewrite_flattened(k_cache, v_cache, list_of_src_indices, list_of_dst_indices):
    """
    Args:
        list_of_src_indices: List of tensors, e.g., [tensor([1, 2, 3]), tensor([10, 11])]
    """
    # Flatten everything into 1D arrays
    # Now the GPU sees one long list of "moves" to make
    flat_src_indices = torch.cat(list_of_src_indices)
    flat_dst_indices = torch.cat(list_of_dst_indices)

    total_elements = flat_src_indices.numel()
    BLOCK_SIZE = 8
    D = k_cache.stride(0)
    
    # Grid is based on total work, not number of pages
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    _inplace_rewrite_flattened[grid](
        k_cache, v_cache,
        flat_src_indices, flat_dst_indices,
        total_elements,
        D,
        BLOCK_SIZE
    )

class BenchMark:
    def __init__(self):
        self.workspace_buffer = torch.empty(
            512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        
        self.packed_custom_mask = {}        
        
        self.global_block_table = torch.zeros((256, 2048 * 10 * 64), dtype=torch.int32, device="cuda")
        self.rewrite_block_table = torch.zeros((32 * 8 * 36, 2048 * 10), dtype=torch.int32, device="cuda")
        self.kv_last_page_lens = torch.ones(32 * 36, dtype=torch.int32, device="cuda")   

        self.paged_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend="auto",
        )

        self.block_sparse_wrapper = BlockSparseAttentionWrapper(
            self.workspace_buffer,
            backend="auto",
        )
        
        self.ragged_prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, 
            "NHD", 
            backend="auto"
        )
                    
    def LS_generate_layouts(self, num_seqs, seq_length, num_selected):        
        selected_indices_all_layers = torch.cumsum(
            torch.tensor([0] + [num_selected] * num_layers, dtype=torch.int32, device="cuda"), dim=0
        )
        
        seq_length_sum_layers = selected_indices_all_layers[-1]
        
        masks_sparse = {layer_id: [] for layer_id in range(num_layers)}
        mask_indptr = [0] + [seq_length] * num_seqs
        mask_indptr = torch.cumsum(torch.tensor(mask_indptr, dtype=torch.int32, device="cuda"), dim=0)
        
        masks_sum = {layer_id: [] for layer_id in range(num_layers)}
        mask_indptr_sum = [0] + [seq_length_sum_layers] * num_seqs
        mask_indptr_sum = torch.cumsum(torch.tensor(mask_indptr_sum, dtype=torch.int32, device="cuda"), dim=0)
        
        self.src_page_indptr_list = []
        self.dst_page_indptr_list = []
        
        for seq_id in range(num_seqs):
            block_table_ids = torch.randint(
                0, num_blocks, (seq_length_sum_layers,), device="cuda"
            )
            
            if self.mode == "LS_rewrite_prefill":
                self.global_block_table[seq_id, :num_selected] = block_table_ids[:num_selected].to(torch.int32)
            if self.mode == "LS_sparse_prefill":
                self.global_block_table[seq_id, :seq_length] = block_table_ids[:seq_length].to(torch.int32)
            if self.mode == "LS_flatten_sum_prefill":
                self.global_block_table[seq_id, :seq_length_sum_layers] = block_table_ids.to(torch.int32)
            
            for layer_id in range(num_layers):
                selected_indices_sparse = torch.randint(
                    0, seq_length, (num_selected,), device="cuda"
                )
            
                cu_mask_sum = torch.zeros((seq_length_sum_layers,), dtype=torch.bool, device="cuda")
                cu_mask_sum[selected_indices_all_layers[layer_id]:selected_indices_all_layers[layer_id + 1]] = True
                masks_sum[layer_id].append(cu_mask_sum)
            
                cu_mask_sparse = torch.zeros((seq_length,), dtype=torch.bool, device="cuda")
                cu_mask_sparse[selected_indices_sparse] = True
                masks_sparse[layer_id].append(cu_mask_sparse)
                
                if self.mode == "LS_rewrite_prefill":
                    self.src_page_indptr_list.append(block_table_ids[selected_indices_sparse].to(torch.int32))
                    self.dst_page_indptr_list.append(self.global_block_table[seq_id, :num_selected].to(torch.int32))
        
        if self.mode == "LS_sparse_prefill":    
            for layer_id in range(num_layers):
                cu_mask_sparse = torch.cat(masks_sparse[layer_id], dim=0).contiguous().view(-1)
                cu_packed_mask, _ = segment_packbits(cu_mask_sparse, mask_indptr, bitorder="little")
                self.packed_custom_mask[layer_id] = cu_packed_mask
            self.cu_page_lengths = torch.ones(
                (num_seqs, ), dtype=torch.int32, device="cuda"  
            ) * seq_length
        
        if self.mode == "LS_flatten_sum_prefill":
            for layer_id in range(num_layers):
                cu_mask_sum = torch.cat(masks_sum[layer_id], dim=0).contiguous().view(-1)
                cu_packed_mask, _ = segment_packbits(cu_mask_sum, mask_indptr_sum, bitorder="little")
                self.packed_custom_mask[layer_id] = cu_packed_mask
            self.cu_page_lengths = torch.ones(
                (num_seqs, ), dtype=torch.int32, device="cuda"
            ) * seq_length_sum_layers
        if self.mode == "LS_rewrite_prefill":
            self.cu_page_lengths = torch.ones(
                (num_seqs, ), dtype=torch.int32, device="cuda"
            ) * num_selected

    def LA_generate_layouts(self, num_seqs, seq_length, num_selected):
        nums_selected = [int(seq_length * (1 / 2) * (0.890533 ** l)) for l in range(num_layers)]
        # for flatten sum indexing 
        nums_selected_indptr = torch.cumsum(
            torch.tensor([0] + nums_selected, dtype=torch.int32, device="cuda"), dim=0
        )
        
        seq_length_sum_layers = nums_selected_indptr[-1]
        seq_length_max_layers = nums_selected[0]
        
        masks_sparse = {layer_id: [] for layer_id in range(num_layers)}
        
        masks_max = {layer_id: [] for layer_id in range(num_layers)}
        
        mask_indptr_sparse = [0] + [seq_length] * num_seqs
        mask_indptr_sparse = torch.cumsum(torch.tensor(mask_indptr_sparse, dtype=torch.int32, device="cuda"), dim=0)
        
        # masks_sum = []
        # mask_indptr_sum = [0] + [seq_length_sum_layers] * num_seqs
        # mask_indptr_sum = torch.cumsum(torch.tensor(mask_indptr_sum, dtype=torch.int32, device="cuda"), dim=0)
        # block_indices_all_layers = torch.arange(0, seq_length_sum_layers, device="cuda", dtype=torch.int32)

        mask_indptr_max = [0] + [seq_length_max_layers] * num_seqs
        
        mask_indptr_max = torch.cumsum(torch.tensor(mask_indptr_max, dtype=torch.int32, device="cuda"), dim=0)
                
        self.src_page_indptr_list = []
        self.dst_page_indptr_list = []
        
        for seq_id in range(num_seqs):
            block_table_ids = torch.randint(
                0, num_blocks, (seq_length_sum_layers,), device="cuda"
            )

            selected_indices_all_layers = torch.randint(
                0, seq_length, (seq_length_sum_layers, ), device="cuda"
            )

            # # sum is probably useless here
            # selected_indices_sum = block_indices_all_layers[selected_indices_all_layers[self.layer_id]:selected_indices_all_layers[self.layer_id + 1]]
            # cu_mask_sum = torch.zeros((seq_length_sum_layers,), dtype=torch.bool, device="cuda")
            # cu_mask_sum[selected_indices_sum] = True
            # masks_sum.append(cu_mask_sum)   
            
            # if self.mode == "LA_flatten_sum_prefill":
            #     self.global_block_table[seq_id, :selected_indices_sum.shape[0]] = block_table_ids[selected_indices_sum].to(torch.int32)
            if self.mode == "LA_flatten_max_prefill":
                self.global_block_table[seq_id, :seq_length_max_layers] = block_table_ids[:seq_length_max_layers].to(torch.int32)
            if self.mode == "LA_sparse_prefill":
                self.global_block_table[seq_id, :seq_length] = block_table_ids[:seq_length].to(torch.int32)

            for layer_id in range(num_layers):
                selected_indices_sparse = selected_indices_all_layers[nums_selected_indptr[layer_id]: nums_selected_indptr[layer_id + 1]]
                
                cu_mask_sparse = torch.zeros((seq_length,), dtype=torch.bool, device="cuda")
                cu_mask_sparse[selected_indices_sparse] = True
                masks_sparse[layer_id].append(cu_mask_sparse)
                
                cu_mask_max = torch.zeros((seq_length_max_layers,), dtype=torch.bool, device="cuda")
                cu_mask_max[:nums_selected[layer_id]] = True
                masks_max[layer_id].append(cu_mask_max)
                
                if self.mode == "LA_flatten_max_prefill":
                    self.src_page_indptr_list.append(block_table_ids[selected_indices_sparse].to(torch.int32))
                    self.dst_page_indptr_list.append(self.global_block_table[seq_id, :seq_length_max_layers].to(torch.int32))
        
        # if self.mode == "LA_flatten_sum_prefill":
        #     cu_mask_sum = torch.cat(masks_sum, dim=0).contiguous().view(-1)
        #     cu_packed_mask_sum, _ = segment_packbits(cu_mask_sum, mask_indptr_sum, bitorder="little")
        #     self.packed_custom_mask = cu_packed_mask_sum
        #     self.cu_page_lengths = torch.ones(
        #         (num_seqs, ), dtype=torch.int32, device="cuda"
        #     ) * seq_length_sum_layers
        
        if self.mode == "LA_flatten_max_prefill":
            for layer_id in range(num_layers):
                cu_mask_max = torch.cat(masks_max[layer_id], dim=0).contiguous().view(-1)
                cu_packed_mask_max, _ = segment_packbits(cu_mask_max, mask_indptr_max, bitorder="little")
                self.packed_custom_mask[layer_id] = cu_packed_mask_max
            self.cu_page_lengths = torch.ones(
                (num_seqs, ), dtype=torch.int32, device="cuda"
            ) * seq_length_max_layers
        
        if self.mode == "LA_sparse_prefill":
            for layer_id in range(num_layers):
                cu_mask_sparse = torch.cat(masks_sparse[layer_id], dim=0).contiguous().view(-1)
                cu_packed_mask_sparse, _ = segment_packbits(cu_mask_sparse, mask_indptr_sparse, bitorder="little")
                self.packed_custom_mask[layer_id] = cu_packed_mask_sparse
            self.cu_page_lengths = torch.ones(
                (num_seqs, ), dtype=torch.int32, device="cuda"
            ) * seq_length
    
    def HS_generate_layouts(self, num_seqs, seq_length, num_selected):
        nums_selected = [int(seq_length * (1 / 2) * (0.502017 ** l)) for l in range(num_kv_heads)]        
        nums_selected_inptr = torch.cumsum(
            torch.tensor([0] + nums_selected, dtype=torch.int32, device="cuda"), dim=0
        )
        seq_length_sum_heads = nums_selected_inptr[-1]
        seq_length_max_heads = nums_selected[0]
    
        mask_sparse = {layer_id: [] for layer_id in range(num_layers)}
        mask_indptr_sparse = [0] + [seq_length] * num_seqs * num_kv_heads
        mask_indptr_sparse = torch.cumsum(torch.tensor(mask_indptr_sparse, dtype=torch.int32, device="cuda"), dim=0)
        
        mask_sum = {layer_id: [] for layer_id in range(num_layers)}
        mask_indptr_sum = [0] + [seq_length_sum_heads] * num_seqs * num_kv_heads
        mask_indptr_sum = torch.cumsum(torch.tensor(mask_indptr_sum, dtype=torch.int32, device="cuda"), dim=0)
        
        mask_max = {layer_id: [] for layer_id in range(num_layers)}
        mask_indptr_max = [0] + [seq_length_max_heads] * num_seqs * num_kv_heads
        mask_indptr_max = torch.cumsum(torch.tensor(mask_indptr_max, dtype=torch.int32, device="cuda"), dim=0)
                
        self.src_page_indptr_list = []
        self.dst_page_indptr_list = []
        
        for seq_id in range(num_seqs):
            block_table_ids = []
            block_table_ids_sum = []
            block_table_ids_max = []
            
            cu_mask_sparse = {layer_id: [] for layer_id in range(num_layers)}
            cu_mask_sum = {layer_id: [] for layer_id in range(num_layers)}
            cu_mask_max = {layer_id: [] for layer_id in range(num_layers)}
            
            for head_id in range(num_kv_heads):
                block_table_ids_per_head = torch.randint(
                    0, num_blocks, (seq_length,), device="cuda"
                )
                block_table_ids.append(block_table_ids_per_head + head_id * num_blocks)
                block_table_ids_sum.append(block_table_ids_per_head[:seq_length_sum_heads] + head_id * num_blocks)
                block_table_ids_max.append(block_table_ids_per_head[:seq_length_max_heads] + head_id * num_blocks)
            
            block_table_ids = torch.stack(block_table_ids, dim=0)
            block_table_ids_sum = torch.stack(block_table_ids_sum, dim=0)
            block_table_ids_max = torch.stack(block_table_ids_max, dim=0)
            
            if self.mode == "HS_sparse_prefill":
                self.global_block_table[seq_id * num_kv_heads: (seq_id + 1) * num_kv_heads, :block_table_ids.shape[-1]] = block_table_ids.to(torch.int32)
            if self.mode == "HS_flatten_sum_prefill":
                self.global_block_table[seq_id * num_kv_heads: (seq_id + 1) * num_kv_heads, :block_table_ids_sum.shape[-1]] = block_table_ids_sum.to(torch.int32)
            if self.mode == "HS_flatten_max_prefill":
                self.global_block_table[seq_id * num_kv_heads: (seq_id + 1) * num_kv_heads, :block_table_ids_max.shape[-1]] = block_table_ids_max.to(torch.int32)
            
            for layer_id in range(num_layers):
                selected_indices_sparse = []
                for head_id in range(num_kv_heads):
                    selected_indices_sparse_per_head = torch.randint(
                        0, seq_length, (nums_selected[head_id],), device="cuda"
                    )
                    
                    selected_indices_sparse.append(selected_indices_sparse_per_head + head_id * seq_length)
                    
                    mask_sparse_per_head = torch.zeros((seq_length,), dtype=torch.bool, device="cuda")
                    mask_sparse_per_head[selected_indices_sparse_per_head] = True
                    cu_mask_sparse[layer_id].append(mask_sparse_per_head)
                    
                    mask_sum_per_head = torch.zeros(seq_length_sum_heads, dtype=torch.bool, device="cuda")
                    mask_sum_per_head[nums_selected_inptr[head_id]:nums_selected_inptr[head_id + 1]] = True
                    cu_mask_sum[layer_id].append(mask_sum_per_head)
                    
                    mask_max_per_head = torch.zeros(seq_length_max_heads, dtype=torch.bool, device="cuda")
                    mask_max_per_head[:nums_selected[head_id]] = True
                    cu_mask_max[layer_id].append(mask_max_per_head)
                    
                    if self.mode == "HS_flatten_max_prefill":
                        self.src_page_indptr_list.append(block_table_ids.view(-1)[selected_indices_sparse[-1]].to(torch.int32))
                        self.dst_page_indptr_list.append(self.global_block_table[seq_id, seq_length * head_id: seq_length * head_id + nums_selected[head_id]].to(torch.int32))
                
                selected_indices_sparse = torch.cat(selected_indices_sparse, dim=0).contiguous().view(-1)
                
                if self.mode == "HS_flatten_sum_prefill":
                    self.src_page_indptr_list.append(block_table_ids.view(-1)[selected_indices_sparse].to(torch.int32))
                    self.dst_page_indptr_list.append(self.global_block_table[seq_id, :seq_length_sum_heads].to(torch.int32))
            
                cu_mask_sparse[layer_id] = torch.cat(cu_mask_sparse[layer_id], dim=0).contiguous().view(-1)
                mask_sparse[layer_id].append(cu_mask_sparse[layer_id])

                cu_mask_sum[layer_id] = torch.cat(cu_mask_sum[layer_id], dim=0).contiguous().view(-1)
                mask_sum[layer_id].append(cu_mask_sum[layer_id])
                
                cu_mask_max[layer_id] = torch.cat(cu_mask_max[layer_id], dim=0).contiguous().view(-1)
                mask_max[layer_id].append(cu_mask_max[layer_id])
        
        if self.mode == "HS_sparse_prefill":
            for layer_id in range(num_layers):
                cu_mask_sparse = torch.cat(mask_sparse[layer_id], dim=0).contiguous().view(-1)
                cu_packed_mask_sparse, _ = segment_packbits(cu_mask_sparse, mask_indptr_sparse, bitorder="little")
                self.packed_custom_mask[layer_id] = cu_packed_mask_sparse
            self.cu_page_lengths = torch.ones(
                (num_seqs * num_kv_heads,), dtype=torch.int32, device="cuda"
            ) * seq_length 
        
        if self.mode == "HS_flatten_sum_prefill":
            for layer_id in range(num_layers):
                cu_mask_sum = torch.cat(mask_sum[layer_id], dim=0).contiguous().view(-1)
                cu_packed_mask_sum, _ = segment_packbits(cu_mask_sum, mask_indptr_sum, bitorder="little")
                self.packed_custom_mask[layer_id] = cu_packed_mask_sum
            self.cu_page_lengths = torch.ones(
                (num_seqs * num_kv_heads,), dtype=torch.int32, device="cuda"
            ) * seq_length_sum_heads
        
        if self.mode == "HS_flatten_max_prefill":
            for layer_id in range(num_layers):
                cu_mask_max = torch.cat(mask_max[layer_id], dim=0).contiguous().view(-1)
                cu_packed_mask_max, _ = segment_packbits(cu_mask_max, mask_indptr_max, bitorder="little")
                self.packed_custom_mask[layer_id] = cu_packed_mask_max
            self.cu_page_lengths = torch.ones(
                (num_seqs * num_kv_heads,), dtype=torch.int32, device="cuda"
            ) * seq_length_max_heads
    
    def HA_generate_layouts(self, num_seqs, seq_length, num_selected):
        nums_selected = np.zeros((num_layers, num_kv_heads), dtype=np.int32)
        for i in range(num_layers):
            for j in range(num_kv_heads):
                nums_selected[i, j] = seq_length * 0.5 * (0.93389**i) * (0.87215**j)
        seq_length_max = nums_selected[0, 0]
        
        mask_sparse = {layer_id: [] for layer_id in range(num_layers)}
        mask_indptr_sparse = [0] + [seq_length] * num_seqs * num_kv_heads
        mask_indptr_sparse = torch.cumsum(torch.tensor(mask_indptr_sparse, dtype=torch.int32, device="cuda"), dim=0)
        
        mask_max = {layer_id: [] for layer_id in range(num_layers)}
        mask_indptr_max = [0] + [seq_length_max] * num_seqs * num_kv_heads
        mask_indptr_max = torch.cumsum(torch.tensor(mask_indptr_max, dtype=torch.int32, device="cuda"), dim=0)
        
        self.src_page_indptr_list = []
        self.dst_page_indptr_list = []
        
        for seq_id in range(num_seqs):
            block_table_ids = []
            block_table_ids_max = []
            
            cu_mask_sparse = {layer_id: [] for layer_id in range(num_layers)}
            cu_mask_max = {layer_id: [] for layer_id in range(num_layers)}
            
            for head_id in range(num_kv_heads):
                block_table_ids_per_head = torch.randint(
                    0, num_blocks * num_kv_heads, (seq_length,), device="cuda"
                )
                block_table_ids.append(block_table_ids_per_head + head_id * num_blocks)
                block_table_ids_max.append(block_table_ids_per_head[:seq_length_max] + head_id * num_blocks)
            
            block_table_ids = torch.stack(block_table_ids, dim=0)
            block_table_ids_max = torch.stack(block_table_ids_max, dim=0)
            
            if self.mode == "HA_sparse_prefill":
                self.global_block_table[seq_id * num_kv_heads: (seq_id + 1) * num_kv_heads, :block_table_ids.shape[-1]] = block_table_ids.to(torch.int32)
            if self.mode == "HA_flatten_max_prefill":
                self.global_block_table[seq_id * num_kv_heads: (seq_id + 1) * num_kv_heads, :block_table_ids_max.shape[-1]] = block_table_ids_max.to(torch.int32)
            
            for layer_id in range(num_layers):
                selected_indices_sparse = []
                for head_id in range(num_kv_heads):
                    selected_indices_sparse_per_head = torch.randint(
                        0, seq_length, (int(nums_selected[layer_id, head_id]),), device="cuda"
                    )
                    
                    selected_indices_sparse.append(selected_indices_sparse_per_head + head_id * seq_length)

                    mask_sparse_per_head = torch.zeros((seq_length,), dtype=torch.bool, device="cuda")
                    mask_sparse_per_head[selected_indices_sparse_per_head] = True
                    cu_mask_sparse[layer_id].append(mask_sparse_per_head)

                    mask_mask_per_head = torch.zeros(seq_length_max, dtype=torch.bool, device="cuda")
                    mask_mask_per_head[:int(nums_selected[layer_id, head_id])] = True
                    cu_mask_max[layer_id].append(mask_mask_per_head)
                    
                    if self.mode == "HA_flatten_max_prefill":
                        self.src_page_indptr_list.append(block_table_ids.view(-1)[selected_indices_sparse[-1]].to(torch.int32))
                        self.dst_page_indptr_list.append(self.global_block_table[seq_id, seq_length * head_id: seq_length * head_id + int(nums_selected[layer_id, head_id])].to(torch.int32))
                    
                selected_indices_sparse = torch.cat(selected_indices_sparse, dim=0).contiguous().view(-1)
                
                cu_mask_sparse[layer_id] = torch.cat(cu_mask_sparse[layer_id], dim=0).contiguous().view(-1)
                mask_sparse[layer_id].append(cu_mask_sparse[layer_id])
                
                cu_mask_max[layer_id] = torch.cat(cu_mask_max[layer_id], dim=0).contiguous().view(-1)
                mask_max[layer_id].append(cu_mask_max[layer_id])
        
        if self.mode == "HA_sparse_prefill":
            for layer_id in range(num_layers):
                cu_mask_sparse = torch.cat(mask_sparse[layer_id], dim=0).contiguous().view(-1)
                cu_packed_mask_sparse, _ = segment_packbits(cu_mask_sparse, mask_indptr_sparse, bitorder="little")
                self.packed_custom_mask[layer_id] = cu_packed_mask_sparse
            self.cu_page_lengths = torch.ones(
                (num_seqs * num_kv_heads,), dtype=torch.int32, device="cuda"
            ) * seq_length
        
        if self.mode == "HA_flatten_max_prefill":
            for layer_id in range(num_layers):
                cu_mask_max = torch.cat(mask_max[layer_id], dim=0).contiguous().view(-1)
                cu_packed_mask_max, _ = segment_packbits(cu_mask_max, mask_indptr_max, bitorder="little")
                self.packed_custom_mask[layer_id] = cu_packed_mask_max
            self.cu_page_lengths = torch.ones(
                (num_seqs * num_kv_heads,), dtype=torch.int32, device="cuda"
            ) * seq_length_max
            
    def headwise_prepare(self):
        num_reqs = self.cu_page_lengths.shape[0]
        kv_indptr = torch.zeros((self.cu_page_lengths.shape[0] + 1), device="cuda", dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(self.cu_page_lengths, dim=0)

        kv_page_indices = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device="cuda"
        )

        create_flashinfer_kv_indices_triton[(num_reqs,)](
            self.global_block_table,
            torch.arange(num_reqs, device="cuda", dtype=torch.int32),
            self.cu_page_lengths,
            kv_indptr,
            None,
            kv_page_indices,
            self.global_block_table.shape[1],
        )
                
        kv_last_page_lens = self.kv_last_page_lens[:num_reqs]   
        qo_indptr = torch.arange(num_reqs + 1, device="cuda").to(torch.int32)
        
        self.paged_prefill_wrapper.begin_forward(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_page_indices,
            paged_kv_last_page_len=kv_last_page_lens,   
            num_qo_heads=num_heads // num_kv_heads,
            num_kv_heads=1,
            packed_custom_mask=self.packed_custom_mask[0] if 0 in self.packed_custom_mask.keys() else None,
            head_dim_qk=head_dim,
            page_size=1,
            q_data_type=torch.bfloat16,
        )

        self.forward_wrapper = self.paged_prefill_wrapper
        
    def prepare(self):        
        num_reqs = self.cu_page_lengths.shape[0]
        kv_indptr = torch.zeros((self.cu_page_lengths.shape[0] + 1), device="cuda", dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(self.cu_page_lengths, dim=0)
        
        kv_page_indices = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device="cuda"
        )
        
        create_flashinfer_kv_indices_triton[(num_reqs,)](
            self.global_block_table,
            torch.arange(num_reqs, device="cuda", dtype=torch.int32),
            self.cu_page_lengths,
            kv_indptr,
            None,
            kv_page_indices,
            self.global_block_table.shape[1],
        )
        
        kv_last_page_lens = self.kv_last_page_lens[:num_reqs]   
        qo_indptr = torch.arange(num_reqs + 1, device="cuda").to(torch.int32)
        
        self.paged_prefill_wrapper.begin_forward(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_page_indices,
            paged_kv_last_page_len=kv_last_page_lens,   
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            packed_custom_mask=self.packed_custom_mask[0] if 0 in self.packed_custom_mask.keys() else None,
            head_dim_qk=head_dim,
            page_size=1,
            q_data_type=torch.bfloat16,
        )
        
        self.forward_wrapper = self.paged_prefill_wrapper
    
    def sparse_prepare(self):
        num_reqs = self.cu_page_lengths.shape[0]
        kv_indptr = torch.zeros((self.cu_page_lengths.shape[0] + 1), device="cuda", dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(self.cu_page_lengths, dim=0)

        kv_page_indices = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device="cuda"
        )

        create_flashinfer_kv_indices_triton[(num_reqs,)](
            self.global_block_table,
            torch.arange(num_reqs, device="cuda", dtype=torch.int32),
            self.cu_page_lengths,
            kv_indptr,
            None,
            kv_page_indices,
            self.global_block_table.shape[1],
        )
        
        qo_indptr = torch.arange(num_reqs + 1, device="cuda").to(torch.int32)
        self.block_sparse_wrapper.begin_forward(
            indptr = qo_indptr, 
            indices = kv_page_indices, 
            M = num_reqs, 
            N = num_blocks, 
            R = 1, 
            C = 1, 
            num_qo_heads = num_heads, 
            num_kv_heads = num_kv_heads,
            head_dim = head_dim, 
            packed_mask=self.packed_custom_mask if hasattr(self, "packed_custom_mask") else None,
            q_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16, 
        )
        
        self.forward_wrapper = self.block_sparse_wrapper

        
    def flatten_sparse_compute(self, q, k_cache, v_cache):
        q = q.view(-1, num_kv_heads, num_heads // num_kv_heads, head_dim).reshape(-1, num_heads // num_kv_heads, head_dim)
        self.forward_wrapper.partial_update_custom_mask(self.packed_custom_mask[self.layer_id])
        o = self.forward_wrapper.forward(q, (k_cache, v_cache))
        o = o.view(-1, num_kv_heads, num_heads // num_kv_heads, head_dim).view(-1, num_heads, head_dim)
        return o 
    
    def sparse_compute(self, q, k_cache, v_cache):
        self.forward_wrapper.partial_update_custom_mask(self.packed_custom_mask[self.layer_id])
        o = self.forward_wrapper.forward(q, (k_cache, v_cache))
        return o
    
    def placeholder_rewrite(self):
        return 
    
    def rewrite(self):
        inplace_rewrite_flattened(self.k_cache, self.v_cache, self.src_page_indptr_list, self.dst_page_indptr_list)  
        
    def compute(self, q, k_cache, v_cache):
        o = self.forward_wrapper.forward(q, (k_cache, v_cache))
        return o
    
    def run(self, method, num_blocks, num_seqs_list, seq_length_means):
        stats = []
        if method == "LS_rewrite_prefill":
            self.mode = "LS_rewrite_prefill"
            # generate_tables = self.generate_block_tables
            generate_tables = self.LS_generate_layouts
            prepare = self.prepare
            self.k_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            rewrite = self.rewrite
            # rewrite = self.placeholder_rewrite # for baseline computation
            compute = self.compute
        
        elif method == "LS_sparse_prefill":
            self.mode = "LS_sparse_prefill"
            generate_tables = self.LS_generate_layouts
            prepare = self.prepare
            self.k_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            compute = self.sparse_compute
            rewrite = self.placeholder_rewrite
        
        elif method == "LS_flatten_sum_prefill":
            self.mode = "LS_flatten_sum_prefill"
            generate_tables = self.LS_generate_layouts
            prepare = self.prepare 
            # prepare = self.sparse_prepare
            self.k_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            # compute = self.sparse_compute_bsr
            compute = self.sparse_compute
            rewrite = self.placeholder_rewrite
            # rewrite = self.rewrite
            
        elif method == "LA_flatten_sum_prefill":
            self.mode = "LA_flatten_sum_prefill"
            generate_tables = self.LA_generate_layouts
            prepare = self.prepare
            self.k_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            compute = self.sparse_compute
         
        elif method == "LA_flatten_max_prefill":
            self.mode = "LA_flatten_max_prefill"
            generate_tables = self.LA_generate_layouts
            prepare = self.prepare
            self.k_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            # compute = self.sparse_rewrite_compute
            compute = self.compute
            rewrite = self.rewrite
            
        elif method == "LA_sparse_prefill":
            self.mode = "LA_sparse_prefill"
            generate_tables = self.LA_generate_layouts
            prepare = self.prepare
            self.k_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            compute = self.sparse_compute
            rewrite = self.placeholder_rewrite
        
        elif method == "HS_sparse_prefill":
            self.mode = "HS_sparse_prefill"
            generate_tables = self.HS_generate_layouts
            prepare = self.headwise_prepare
            self.k_cache = torch.randn(
                [num_blocks * num_kv_heads, 1, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks * num_kv_heads, 1, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            compute = self.flatten_sparse_compute
            rewrite = self.placeholder_rewrite
        
        elif method == "HS_flatten_sum_prefill":
            self.mode = "HS_flatten_sum_prefill"
            generate_tables = self.HS_generate_layouts
            prepare = self.headwise_prepare
            self.k_cache = torch.randn(
                [num_blocks * num_kv_heads, 1, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks * num_kv_heads, 1, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            compute = self.flatten_sparse_compute
            rewrite = self.rewrite
        
        elif method == "HS_flatten_max_prefill":
            self.mode = "HS_flatten_max_prefill"
            generate_tables = self.HS_generate_layouts
            prepare = self.headwise_prepare
            self.k_cache = torch.randn(
                [num_blocks * num_kv_heads, 1, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks * num_kv_heads, 1, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            compute = self.flatten_sparse_compute
            rewrite = self.rewrite
        
        elif method == "HA_sparse_prefill":
            self.mode = "HA_sparse_prefill"
            generate_tables = self.HA_generate_layouts
            prepare = self.headwise_prepare
            self.k_cache = torch.randn(
                [num_blocks * num_kv_heads, 1, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks * num_kv_heads, 1, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            compute = self.flatten_sparse_compute
            rewrite = self.placeholder_rewrite
        
        elif method == "HA_flatten_max_prefill":
            self.mode = "HA_flatten_max_prefill"
            generate_tables = self.HA_generate_layouts
            prepare = self.headwise_prepare
            self.k_cache = torch.randn(
                [num_blocks * num_kv_heads, 1, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks * num_kv_heads, 1, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            compute = self.flatten_sparse_compute
            rewrite = self.rewrite
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        for num_seqs in num_seqs_list:
            q = torch.randn([num_seqs, 32, 128], dtype=torch.bfloat16).to(device)
            # Warmup runs
            self.layer_id = 0
            for _ in range(5):
                with torch.no_grad():
                    generate_tables(num_seqs, 2048, 512)
                    prepare()
                    rewrite()
                    _ = compute(q, self.k_cache, self.v_cache)

            torch.cuda.synchronize()

            # Timing runs
            
            # num_runs = 10
            num_runs = 1
            
            print("Starting benchmark...")
            with torch.no_grad():
                for seq_length_mean in seq_length_means:
                    prepare_times = []
                    rewrite_times = []
                    compute_times = []
                    for _ in range(num_runs):
                        generate_tables(num_seqs, seq_length_mean, seq_length_mean // 8)
                        prepare_start_event = torch.cuda.Event(enable_timing=True)
                        prepare_end_event = torch.cuda.Event(enable_timing=True)
                        rewrite_start_event = torch.cuda.Event(enable_timing=True)

                        prepare_start_event.record()
                        prepare()
                        rewrite_start_event.record()
                        rewrite()
                        prepare_end_event.record()
                        torch.cuda.synchronize()

                        rewrite_times.append(
                            rewrite_start_event.elapsed_time(prepare_end_event)
                        )
                        
                        prepare_times.append(
                            prepare_start_event.elapsed_time(prepare_end_event)
                        )
                        
                        self.layer_id = 0
                        for _ in range(num_layers):
                            compute_start_event = torch.cuda.Event(enable_timing=True)
                            compute_end_event = torch.cuda.Event(enable_timing=True)

                            compute_start_event.record()
                            o = compute(q, self.k_cache, self.v_cache)
                            compute_end_event.record()
                            torch.cuda.synchronize()

                            compute_times.append(
                                compute_start_event.elapsed_time(compute_end_event)
                            )
                            
                            self.layer_id += 1
                    rewrite_mean_ms = np.mean(rewrite_times)
                    prepare_mean_ms = np.mean(prepare_times)
                    compute_mean_ms = np.mean(compute_times)
                    stats.append({
                        "seq_length_mean": seq_length_mean,
                        "num_seqs": num_seqs,
                        "prepare_mean_ms": prepare_mean_ms,
                        "rewrite_mean_ms": rewrite_mean_ms,
                        "compute_mean_ms": compute_mean_ms,
                    })

        return stats


if __name__ == "__main__":
    # set random seed
    np.random.seed(1111)
    bench = BenchMark()
    num_seqs = [1, 2, 4, 8, 16, 32]
    # num_seqs = [32]
    seq_lengths = [i * 2048 for i in range(1, 11)]
    # seq_lengths = [i * 256 for i in range(1, 31)]
    # seq_lengths = [20480]
    
    # LS_rewrite_stats = bench.run("LS_rewrite_prefill", num_blocks, num_seqs, seq_lengths)
    # LS_sparse_stats = bench.run("LS_sparse_prefill", num_blocks, num_seqs, seq_lengths)
    # LS_flatten_sum_stats = bench.run("LS_flatten_sum_prefill", num_blocks, num_seqs, seq_lengths)
    
    # LA_sparse_stats = bench.run("LA_sparse_prefill", num_blocks, num_seqs, seq_lengths)
    # LA_flatten_max_stats = bench.run("LA_flatten_max_prefill", num_blocks, num_seqs, seq_lengths)
    # LA_flatten_sum_stats = bench.run("LA_flatten_sum_prefill", num_blocks, num_seqs, seq_lengths)
    
    # HS_sparse_stats = bench.run("HS_sparse_prefill", num_blocks, num_seqs, seq_lengths)
    # HS_flatten_sum_stats = bench.run("HS_flatten_sum_prefill", num_blocks, num_seqs, seq_lengths)
    # HS_flatten_max_stats = bench.run("HS_flatten_max_prefill", num_blocks, num_seqs, seq_lengths)
    
    HA_sparse_stats = bench.run("HA_sparse_prefill", num_blocks, num_seqs, seq_lengths)
    HA_flatten_max_stats = bench.run("HA_flatten_max_prefill", num_blocks, num_seqs, seq_lengths)
    
    # print("LA Sparse Prefill Stats:", LA_sparse_stats)
    # print("LA Flatten Max Prefill Stats:", LA_flatten_max_stats)
    # print("LA Flatten Sum Prefill Stats:", LA_flatten_sum_stats)
    
    # print("LS Rewrite Prefill Stats:", LS_rewrite_stats)
    # print("LS Sparse Prefill Stats:", LS_sparse_stats)
    # print("LS Flatten Sum Prefill Stats:", LS_flatten_sum_stats)
    
    # print("HS Sparse Prefill Stats:", HS_sparse_stats)
    # print("HS Flatten Sum Prefill Stats:", HS_flatten_sum_stats)
    # print("HS Flatten Max Prefill Stats:", HS_flatten_max_stats)
    
    print("HA Sparse Prefill Stats:", HA_sparse_stats)
    print("HA Flatten Max Prefill Stats:", HA_flatten_max_stats)

    # np.save("baseline_stats.npy", LS_rewrite_stats)

    # np.save("LS_backend_stats.npy", 
    #         {
    #          "LS_flatten_sum_prefill": LS_flatten_sum_stats,
    #          "LS_rewrite_prefill": LS_rewrite_stats,
    #          "LS_sparse_prefill": LS_sparse_stats,    
    #         })
    
    # np.save("LA_backend_stats.npy", 
    #         {
    #          "LA_flatten_max_prefill": LA_flatten_max_stats,
    #          "LA_sparse_prefill": LA_sparse_stats,   
    #         })
    
    # np.save("HS_backend_stats.npy",
    #         {
    #             "HS_flatten_max_prefill": HS_flatten_max_stats,
    #             "HS_flatten_sum_prefill": HS_flatten_sum_stats,
    #             "HS_sparse_prefill": HS_sparse_stats,
    #         }
    # )
    
    np.save("HA_backend_stats.npy",
            {
                "HA_flatten_max_prefill": HA_flatten_max_stats,
                "HA_sparse_prefill": HA_sparse_stats,
            }
    )
    