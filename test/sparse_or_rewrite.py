from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    BatchDecodeWithPagedKVCacheWrapper,
)
from flashinfer.quantization import segment_packbits
import itertools
import torch
import numpy as np
import triton
import triton.language as tl 

torch.set_printoptions(profile="full")

device = "cuda:0"

num_blocks = 2048 * 128

num_kv_heads = 8
num_heads = 32
num_layers = 8
head_dim = 128

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

class BenchMark:
    def __init__(self):
        self.workspace_buffer = torch.empty(
            512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        
        self.global_block_table = torch.zeros((32, 2048 * 10 * 64), dtype=torch.int32, device="cuda")
        
        self.kv_last_page_lens = torch.ones(32 * 36, dtype=torch.int32, device="cuda")   

        self.paged_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            # use_tensor_cores=True,
        )

        self.paged_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend="fa2",
        )

        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, 
            "NHD", 
            backend="auto"
        )
        
        self.sparse_decode_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend="fa2",
        )
        
        self.layerwise_sparse_decode_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend="fa2",
        )
    
    def ragged_prefill_prepare(self, block_tables: list[list[int]], num_seqs):
        # Create cu_page_lengths from block tables
        cu_page_lengths = torch.tensor([len(block_table) for block_table in block_tables], dtype=torch.int32, device="cuda")

        kv_indptr = torch.zeros((cu_page_lengths.shape[0] + 1), device="cuda", dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(cu_page_lengths, dim=0)

        kv_page_indices = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device="cuda"
        )

        create_flashinfer_kv_indices_triton[(len(block_tables),)](
            self.global_block_table,
            torch.arange(len(block_tables), device="cuda", dtype=torch.int32),
            cu_page_lengths,
            kv_indptr,
            None,
            kv_page_indices,
            self.global_block_table.shape[1],
        )

        cu_kv_page_indices = torch.arange(0, kv_page_indices.shape[0], device="cuda").to(torch.int32)

        self.k_cache[cu_kv_page_indices] = self.k_cache[kv_page_indices]

        self.cu_kv_page_indices = cu_kv_page_indices

        qo_indptr = torch.arange(len(block_tables) + 1, device="cuda").to(torch.int32)

        self.prefill_wrapper_ragged.begin_forward(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            causal=True,
            q_data_type=torch.bfloat16,
        )
        self.forward_wrapper = self.prefill_wrapper_ragged

    def paged_prefill_prepare(self, block_tables: list[list[int]], num_seqs):
        kv_indptr = torch.zeros((self.cu_page_lengths .shape[0] + 1), device="cuda", dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(self.cu_page_lengths, dim=0)
        
        kv_page_indices = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device="cuda"
        )
        
        create_flashinfer_kv_indices_triton[(len(block_tables),)](
            self.global_block_table,
            torch.arange(len(block_tables), device="cuda", dtype=torch.int32),
            self.cu_page_lengths,
            kv_indptr,
            None,
            kv_page_indices,
            self.global_block_table.shape[1],
        )
        
        kv_last_page_lens = self.kv_last_page_lens[:len(block_tables)]

        qo_indptr = torch.arange(len(block_tables) + 1, device="cuda").to(torch.int32)

        self.paged_prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_page_indices,
            paged_kv_last_page_len=kv_last_page_lens,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            causal=True,
            page_size=1,
            q_data_type=torch.bfloat16,
        )
        self.forward_wrapper = self.paged_prefill_wrapper

    def paged_decode_prepare(self, block_tables: list[list[int]], num_seqs):
        kv_indptr = torch.zeros((self.cu_page_lengths.shape[0] + 1), device="cuda", dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(self.cu_page_lengths, dim=0)

        kv_page_indices = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device="cuda"
        )

        create_flashinfer_kv_indices_triton[(len(block_tables),)](
            self.global_block_table,
            torch.arange(len(block_tables), device="cuda", dtype=torch.int32),
            self.cu_page_lengths,
            kv_indptr,
            None,
            kv_page_indices,
            self.global_block_table.shape[1],
        )

        kv_last_page_lens = self.kv_last_page_lens[:len(block_tables)]

        self.paged_decode_wrapper.begin_forward(
            indptr=kv_indptr,
            indices=kv_page_indices,
            last_page_len=kv_last_page_lens,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
        )

        self.forward_wrapper = self.paged_decode_wrapper
    
    def layerwise_sparse_decode_prepare(self, seq_metadata, num_seqs):
        block_tables = seq_metadata[0]
        packed_custom_mask = seq_metadata[1]

        # Use the cu_page_lengths already computed in layerwise_generate_sparse_block_table
        kv_indptr = torch.zeros((self.cu_page_lengths.shape[0] + 1), device="cuda", dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(self.cu_page_lengths, dim=0)

        kv_page_indices = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device="cuda"
        )
        
        create_flashinfer_kv_indices_triton[(len(block_tables),)](
            self.global_block_table,
            torch.arange(len(block_tables), device="cuda", dtype=torch.int32),
            self.cu_page_lengths,
            kv_indptr,
            None,
            kv_page_indices,
            self.global_block_table.shape[1],
        )
                
        kv_last_page_lens = self.kv_last_page_lens[:len(block_tables)]

        qo_indptr = torch.arange(num_seqs + 1, device="cuda").to(torch.int32)

        self.layerwise_sparse_decode_prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_page_indices,
            paged_kv_last_page_len=kv_last_page_lens,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            # causal=True,
            # custom_mask=mask,
            packed_custom_mask=packed_custom_mask,
            page_size=1,
            q_data_type=torch.bfloat16,
        )
        self.forward_wrapper = self.layerwise_sparse_decode_prefill_wrapper

    def sparse_decode_prepare(self, seq_metadata, num_seqs):
        block_tables = seq_metadata[0]
        packed_custom_mask = seq_metadata[1]

        # Use the cu_page_lengths already computed in generate_sparse_block_table
        kv_indptr = torch.zeros((self.cu_page_lengths.shape[0] + 1), device="cuda", dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(self.cu_page_lengths, dim=0)
        
        kv_page_indices = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device="cuda"
        )

        create_flashinfer_kv_indices_triton[(len(block_tables),)](
            self.global_block_table,
            torch.arange(len(block_tables), device="cuda", dtype=torch.int32),
            self.cu_page_lengths,
            kv_indptr,
            None,
            kv_page_indices,
            self.global_block_table.shape[1],
        )        
        
        kv_last_page_lens = self.kv_last_page_lens[:len(block_tables)]

        qo_indptr = torch.arange(num_seqs * num_kv_heads + 1, device="cuda").to(torch.int32)

        self.sparse_decode_prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_page_indices,
            paged_kv_last_page_len=kv_last_page_lens,
            num_qo_heads=num_heads // num_kv_heads,
            num_kv_heads=1,
            head_dim_qk=head_dim,
            # causal=True,
            # custom_mask=mask,
            packed_custom_mask=packed_custom_mask,
            page_size=1,
            q_data_type=torch.bfloat16,
        )

        self.forward_wrapper = self.sparse_decode_prefill_wrapper

    def flatten_compute(self, q, k_cache, v_cache):
        q = q.view(-1, num_kv_heads, num_heads // num_kv_heads, head_dim).reshape(-1, num_heads // num_kv_heads, head_dim)
        o = self.forward_wrapper.forward(q, (k_cache, v_cache))
        o = o.view(-1, num_kv_heads, num_heads // num_kv_heads, head_dim).view(-1, num_heads, head_dim)
        return o 
    
    def ragged_compute(self, q, k_cache, v_cache ):
        key, value = k_cache[:self.cu_kv_page_indices[-1]], v_cache[:self.cu_kv_page_indices[-1]]
        o = self.forward_wrapper.forward(q, key, value, causal=True)
        return o
    
    def compute(self, q, k_cache, v_cache):
        o = self.forward_wrapper.forward(q, (k_cache, v_cache))
        return o
    
    def layerwise_generate_sparse_block_table(self, num_seqs, seq_length_mean):
        block_tables = []
        layerwise_masks = []

        for seq_id in range(num_seqs):
            block_table_ids = torch.randint(
                0, num_blocks, (seq_length_mean,), device="cuda"
            ).tolist()
            layerwise_mask = torch.full((num_layers,), False, device="cuda", dtype=torch.bool)
            layerwise_mask[0] = True
            layerwise_masks.append(layerwise_mask.unsqueeze(0).repeat(len(block_table_ids), 1).view(-1))
            cu_block_table = list(itertools.chain(*[[table_id * num_layers + i for i in range(num_layers)] for table_id in block_table_ids]))
            assert np.max(block_table_ids) * num_layers < num_blocks * num_layers, "Block ID exceeds number of blocks"
            block_tables.append(cu_block_table)

            # Write to global block table: [seq_id, seq_length] = block_table_id using tensor indexing
            block_table_tensor = torch.tensor(cu_block_table, device="cuda")
            self.global_block_table[seq_id, :len(cu_block_table)] = block_table_tensor
        
        mask_indptr = torch.cumsum(
            torch.tensor(
                [0] + [len(block_table) for block_table in block_tables], device="cuda"
            ),
            dim=0,
        ).to(torch.int32)
        
        custom_mask = torch.cat(layerwise_masks, dim=0).contiguous().view(-1)
        # print(custom_mask.shape)
        packed_custom_mask, _ = segment_packbits(custom_mask, mask_indptr, bitorder="little")
        
        self.cu_page_lengths = torch.tensor(
            [len(block_table) for block_table in block_tables], device="cuda"
        ).to(torch.int32)
        
        return (block_tables, packed_custom_mask)
        
    def generate_sparse_block_table(self, num_seqs, seq_length_mean):
        used_block_table = []
        block_tables = []
        headwise_masks = []

        for seq_id in range(num_seqs):
            block_table_ids = torch.randint(
                0, num_blocks, (seq_length_mean,), device="cuda"
            ).tolist()
            headwise_masks.append([[2 ** i for i in range(num_kv_heads)] for _ in range(len(block_table_ids))])
            cu_block_table = list(itertools.chain(*[[table_id * num_kv_heads + i for i in range(num_kv_heads)] for table_id in block_table_ids]))
            for i in range(num_kv_heads):
                block_tables.append(cu_block_table)
                block_table_tensor = torch.tensor(cu_block_table, dtype=torch.int32, device="cuda")
                self.global_block_table[seq_id * num_kv_heads + i, :len(cu_block_table)] = block_table_tensor
            assert np.max(block_table_ids) < num_blocks * num_kv_heads, "Block ID exceeds number of blocks"
            used_block_table.append(block_table_ids)
            
        mask_arr = [torch.tensor(headwise_mask, device="cuda", dtype=torch.uint8).transpose(0, 1).contiguous().view(-1) for headwise_mask in headwise_masks]
        packed_custom_mask = torch.cat(mask_arr, dim=0).contiguous().view(-1)
        
        self.cu_page_lengths = torch.tensor(
            [len(block_table) for block_table in block_tables], device="cuda"
        ).to(torch.int32)
        
        return (block_tables, packed_custom_mask)

    def generate_block_tables(self, num_seqs, seq_length_mean):
        used_block_table = []
        block_tables = []

        for seq_id in range(num_seqs):
            block_table_ids = torch.randint(
                0, num_blocks, (seq_length_mean,), device="cuda"
            ).tolist()
            block_tables.append(block_table_ids)
            used_block_table.append(block_table_ids)

            # Write to global block table: [seq_id, seq_length] = block_table_id using tensor indexing
            block_table_tensor = torch.tensor(block_table_ids, dtype=torch.int32, device="cuda")
            self.global_block_table[seq_id, :len(block_table_ids)] = block_table_tensor

        self.cu_page_lengths = torch.tensor(
            [len(block_table) for block_table in block_tables], device="cuda"
        ).to(torch.int32)
        
        return block_tables

    def run(self, method, num_blocks, num_seqs_list, seq_length_means):
        stats = []
        
        if method == "paged_decode":
            generate_tables = self.generate_block_tables
            prepare = self.paged_decode_prepare
            self.k_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )  # [num_blocks, layer_idx, num_kv_heads, head_dim]
            self.v_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            compute = self.compute
            
        elif method == "paged_prefill":
            generate_tables = self.generate_block_tables
            prepare = self.paged_prefill_prepare
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
            compute = self.compute
            
        elif method == "sparse_decode_prefill":
            generate_tables = self.generate_sparse_block_table
            prepare = self.sparse_decode_prepare
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
            compute = self.flatten_compute
            
        elif method == "layerwise_sparse_decode_prefill":
            generate_tables = self.layerwise_generate_sparse_block_table
            prepare = self.layerwise_sparse_decode_prepare
            self.k_cache = torch.randn(
                [num_blocks * num_layers, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks * num_layers, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            compute = self.compute

        elif method == "ragged_prefill":
            generate_tables = self.generate_block_tables
            prepare = self.ragged_prefill_prepare
            self.k_cache = torch.randn(
                [num_blocks * 4, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks * 4, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            
            compute = self.ragged_compute
        else:
            raise ValueError(f"Unknown method: {method}")
            
        for num_seqs in num_seqs_list:
            q = torch.randn([num_seqs, 32, 128], dtype=torch.bfloat16).to(device)
            # Warmup runs
            for _ in range(5):
                with torch.no_grad():
                    block_tables = generate_tables(num_seqs, 2048)
                    prepare(block_tables, num_seqs)
                    _ = compute(q, self.k_cache, self.v_cache)

            torch.cuda.synchronize()

            # Timing runs
            prepare_times = []
            compute_times = []
            num_runs = 5
            
            print("Starting benchmark...")
            with torch.no_grad():
                for seq_length_mean in seq_length_means:
                    for _ in range(num_runs):
                        block_tables = generate_tables(num_seqs, seq_length_mean)
                        prepare_start_event = torch.cuda.Event(enable_timing=True)
                        prepare_end_event = torch.cuda.Event(enable_timing=True)

                        prepare_start_event.record()
                        prepare(block_tables, num_seqs)
                        prepare_end_event.record()
                        torch.cuda.synchronize()

                        prepare_times.append(
                            prepare_start_event.elapsed_time(prepare_end_event)
                        )

                        compute_start_event = torch.cuda.Event(enable_timing=True)
                        compute_end_event = torch.cuda.Event(enable_timing=True)

                        compute_start_event.record()
                        output = compute(q, self.k_cache, self.v_cache)
                        compute_end_event.record()
                        torch.cuda.synchronize()

                        compute_times.append(
                            compute_start_event.elapsed_time(compute_end_event)
                        )

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
    num_seqs = [1]
    seq_lengths = [i * 2048 for i in range(1, 11)]
    # seq_lengths = [128]
    # paged_decode_stats = bench.run("paged_decode", num_blocks, num_seqs, seq_lengths)
    # paged_prefill_stats = bench.run("paged_prefill", num_blocks, num_seqs, seq_lengths)
    sparse_prefill_stats = bench.run("sparse_decode_prefill", num_blocks, num_seqs, seq_lengths)
    # ragged_prefill_stats = bench.run("ragged_prefill", num_blocks, num_seqs, seq_lengths)
    layerwise_sparse_decode_prefill_stats = bench.run("layerwise_sparse_decode_prefill", num_blocks, num_seqs, seq_lengths)

    # print("Paged Decode Stats:", paged_decode_stats)
    # print("Paged Prefill Stats:", paged_prefill_stats)
    print("Sparse Prefill Stats:", sparse_prefill_stats)
    # print("Ragged Prefill Stats:", ragged_prefill_stats)
    print("Layerwise Sparse Prefill Stats:", layerwise_sparse_decode_prefill_stats)
    
    # np.save("backend_stats.npy", 
    #         {
    #         #  "paged_decode": paged_decode_stats,
    #         #  "paged_prefill": paged_prefill_stats,
    #         #  "sparse_prefill": sparse_prefill_stats,
    #         "ragged_prefill": ragged_prefill_stats, 
    #         "layerwise_sparse_decode_prefill": layerwise_sparse_decode_prefill_stats
    #         })
