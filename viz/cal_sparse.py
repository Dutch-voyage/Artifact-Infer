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

start_val = None
sparse_ratio = None 


def generate_matrix(rows=36, cols=8):
    target_sum = rows * cols * sparse_ratio
    low, high = 0.0, 0.9999
    r1 = 0.5
    k = 2

    for _ in range(100):
        r1 = (low + high) / 2
        r2 = r1 ** k
        sum_r1 = (1 - r1**rows) / (1 - r1) if abs(1-r1) > 1e-9 else rows
        sum_r2 = (1 - r2**cols) / (1 - r2) if abs(1-r2) > 1e-9 else cols
        current_sum = start_val * sum_r1 * sum_r2

        if abs(current_sum - target_sum) < 1e-9:
            break
        elif current_sum < target_sum:
            low = r1
        else:
            high = r1

    r2 = r1 ** k
    # Optimized matrix generation using broadcasting
    matrix = start_val * np.outer(r1**np.arange(rows), r2**np.arange(cols))
    return matrix
device = "cuda:0"

num_blocks = 2048 * 128

num_kv_heads = 8
num_heads = 32
num_layers = 36
head_dim = 128

sparse_ratio = None
start_val = None

def page_partial_update_mask(self, packed_custom_mask):
    self._custom_mask_buf = packed_custom_mask.to(
        self.device, non_blocking=True
    )

def bsr_partial_update_mask(self, packed_mask):
    self._packed_mask_buf = packed_mask.to(
        self.device, non_blocking=True
    )

BatchPrefillWithPagedKVCacheWrapper.partial_update_custom_mask = page_partial_update_mask

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
    def HA_generate_layouts(self, num_seqs, seq_length, num_selected):
        # nums_selected = np.zeros((num_layers, num_kv_heads), dtype=np.int32)
        # for i in range(num_layers):
        #     for j in range(num_kv_heads):
        #         nums_selected[i, j] = seq_length * 0.5 * (0.93389**i) * (0.87215**j)
        nums_selected = (generate_matrix(num_layers, num_heads) * seq_length).astype(np.int32)
        
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
                    0, num_blocks, (seq_length,), device="cuda"
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
            
    def flatten_sparse_compute(self, q, k_cache, v_cache):
        q = q.view(-1, num_kv_heads, num_heads // num_kv_heads, head_dim).reshape(-1, num_heads // num_kv_heads, head_dim)
        self.forward_wrapper.partial_update_custom_mask(self.packed_custom_mask[self.layer_id])
        o = self.forward_wrapper.forward(q, (k_cache, v_cache))
        o = o.view(-1, num_kv_heads, num_heads // num_kv_heads, head_dim).view(-1, num_heads, head_dim)
        return o 
    
    def placeholder_rewrite(self):
        return 
    
    def rewrite(self):
        inplace_rewrite_flattened(self.k_cache, self.v_cache, self.src_page_indptr_list, self.dst_page_indptr_list)  
    
    def run(self, method, num_blocks, num_seqs_list, seq_length_means):
        stats = []
        
        if method == "HA_sparse_prefill":
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
                    compute_times = []
                    for _ in range(num_runs):
                        generate_tables(num_seqs, seq_length_mean, seq_length_mean // 8)
                        prepare_start_event = torch.cuda.Event(enable_timing=True)
                        prepare_end_event = torch.cuda.Event(enable_timing=True)

                        prepare_start_event.record()
                        prepare()
                        rewrite()
                        prepare_end_event.record()
                        torch.cuda.synchronize()

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

                    prepare_mean_ms = np.mean(prepare_times)
                    compute_mean_ms = np.mean(compute_times)
                    stats.append({
                        "seq_length_mean": seq_length_mean,
                        "num_seqs": num_seqs,
                        "prepare_mean_ms": prepare_mean_ms,
                        "compute_mean_ms": compute_mean_ms,
                    })

        return stats


if __name__ == "__main__":
    # set random seed
    np.random.seed(1111)
    bench = BenchMark()
    # num_seqs = [1, 2, 4, 8, 16, 32]
    num_seqs = [32]
    # seq_lengths = [i * 2048 for i in range(1, 11)]
    seq_lengths = [20480]
    
    start_val = 0.5
    sparse_ratio = 0.125
    
    # sparse_stats = bench.run("HA_sparse_prefill", num_blocks, num_seqs, seq_lengths)
    # max_stats =  bench.run("HA_flatten_max_prefill", num_blocks, num_seqs, seq_lengths)
    
    
    # sparse_levels = np.arange(0.2, 0.8, 0.1)
    # max_sparse_evvels = np.arange(0.2, 0.8, 0.1)
    
    # parameter_configs = list(itertools.product(sparse_levels, max_sparse_evvels))
    # parameter_configs = [[d1, d2] for d1, d2 in parameter_configs if d1 >= d2]
    
    sparse_stats = []
    max_stats = []
    
    parameter_configs = [[0.5, 0.125]]
    
    for d1, d2 in parameter_configs:
        start_val = d1
        sparse_ratio = d2
        HA_sparse_stats = bench.run("HA_sparse_prefill", num_blocks, num_seqs, seq_lengths)
        sparse_stats.append({"starts_val": d1, "sprase_ratio": d2, "methods": "HS_sparse_prefill", "stats": HA_sparse_stats})
        HA_flatten_max_prefill = bench.run("HA_flatten_max_prefill", num_blocks, num_seqs, seq_lengths)
        max_stats.append({"stats_val": d1, "sparse_ratio": d2, "method": "HA_flatten_max_prefill", "stats": HA_flatten_max_prefill})
    
    # np.save(
    #     "sparse_pattern_to_latency", 
    #     {
    #         "HA_flatten_max_prefill": max_stats 
    #     }
    # )
    
    np.save(
        "sparse_compare", 
        {
            "HA_sparse_prefill": sparse_stats, 
            "HA_flatten_max_prefill": max_stats 
        }
    )