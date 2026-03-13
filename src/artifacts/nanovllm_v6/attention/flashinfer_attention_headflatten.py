import torch
from torch import nn
import triton
import triton.language as tl

from flashinfer import BatchPrefillWithPagedKVCacheWrapper, BatchPrefillWithRaggedKVCacheWrapper

from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.prefill import _compute_page_mask_indptr
from flashinfer.decode import _get_range_buf, get_seq_lens, fast_decode_plan
from flashinfer.cascade import merge_state
from flashinfer.quantization import segment_packbits, packbits, _get_indptr_for_packed_mask

import itertools
from typing import Optional, Union
from dataclasses import dataclass

from src.services.nanovllm_v6.utils.context import get_context
from src.services.nanovllm_v6.utils.logging import append_lse_log
from src.services.nanovllm_v6.engine.sequence import Sequence

from src.core.artifact_base import Artifact
from src.core.service_base import BaseService

from functools import partial 

@dataclass
class PrefillMetadata:
    use_ragged: bool
    no_prefix: bool


global_workspace_buffer = None

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


@triton.jit
def store_q_cache_kernel_prefill(
    q_cache_ptr, 
    query_slot_mapping_ptr, 
    query_window_pos_ptr, 
    query_ptr, 
    query_stride, 
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    query_slot = tl.load(query_slot_mapping_ptr + idx)
    query_window_pos = tl.load(query_window_pos_ptr + idx)
    cache_offsets = query_slot * D + tl.arange(0, D)
    query_offsets = query_window_pos * query_stride + tl.arange(0, D)
    query = tl.load(query_ptr + query_offsets)
    tl.store(q_cache_ptr + cache_offsets, query)


@triton.jit
def store_q_cache_kernel_decode(
    q_cache_ptr,
    query_slot_mapping_ptr, 
    query_ptr, 
    query_stride, 
    D: tl.constexpr
):
    idx = tl.program_id(0)
    query_slot = tl.load(query_slot_mapping_ptr + idx)
    cache_offsets = query_slot * D + tl.arange(0, D)
    query_offsets = idx * query_stride + tl.arange(0, D)
    query = tl.load(query_ptr + query_offsets)
    tl.store(q_cache_ptr + cache_offsets, query)


def store_q_cache(query: torch.Tensor, q_cache: torch.Tensor, query_slot_mapping: torch.Tensor, query_window_pos: torch.Tensor=None, is_prefill: bool = True):
    _, num_heads, head_dim = query.shape
    D = num_heads * head_dim
    N = query_slot_mapping.shape[0]
    
    if is_prefill:
        assert query_window_pos.numel() == N
        store_q_cache_kernel_prefill[(N,)](q_cache, query_slot_mapping, query_window_pos, query, query.stride(0), D)
    else:
        store_q_cache_kernel_decode[(N, )](q_cache, query_slot_mapping, query, query.stride(0), D)


@triton.jit
def read_q_cache_kernel(
    q_cache_ptr, 
    query_slot_mapping_ptr,
    query_ptr, 
    L: tl.constexpr, 
    D: tl.constexpr, 
):
    n_idx = tl.program_id(0)
    l_idx = tl.program_id(1)
    query_slot = tl.load(query_slot_mapping_ptr + n_idx)

    cache_offsets = query_slot * L * D + l_idx * D + tl.arange(0, D)
    query_offsets = n_idx * L * D + l_idx * D + tl.arange(0, D)
    query = tl.load(q_cache_ptr + cache_offsets)
    tl.store(query_ptr + query_offsets, query)

def read_q_cache(q_cache: torch.Tensor, query_slot_mapping: torch.Tensor):
    _, L, num_heads, head_dim = q_cache.shape
    D = num_heads * head_dim
    N = query_slot_mapping.shape[0]
    query = torch.empty((N, L, num_heads, head_dim), dtype=q_cache.dtype, device="cuda")
    read_q_cache_kernel[(N, L,)](q_cache, query_slot_mapping, query, L, D)
    return query

@triton.jit
def read_kvcache_kernel(
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    key = tl.load(k_cache_ptr + cache_offsets)
    value = tl.load(v_cache_ptr + cache_offsets)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    tl.store(key_ptr + key_offsets, key)
    tl.store(value_ptr + value_offsets, value)


def read_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor, num_kv_heads=None, head_dim=None):
    N = slot_mapping.numel()
    if num_kv_heads == None:
        num_kv_heads = k_cache.shape[-2]
    if head_dim == None:
        head_dim = k_cache.shape[-1]
    D = num_kv_heads * head_dim
    key = torch.empty((N, num_kv_heads, head_dim), dtype=k_cache.dtype, device=k_cache.device)
    value = torch.empty((N, num_kv_heads, head_dim), dtype=v_cache.dtype, device=v_cache.device)
    
    read_kvcache_kernel[(N,)](k_cache, v_cache, slot_mapping, key, key.stride(0), value, value.stride(0), D)
    
    return key, value

class Attention(nn.Module, Artifact):

    @property
    def name(self):
        return "VanillaAttention"
    
    def __init__(
        self,
        model_runner, 
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        Artifact.__init__(self)
        print("initializing attention")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.model_runner = model_runner
        self.if_log_lse = model_runner.config.if_log_lse_in_attn
        self.block_size = model_runner.config.kvcache_block_size
        
        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
        self.workspace_buffer = global_workspace_buffer
        
        max_bs = min(model_runner.config.max_num_seqs, 512)
        max_seq_len = model_runner.config.max_model_len
        
        self.qo_indptr = torch.zeros(
            (max_bs * num_kv_heads + 1, ), dtype=torch.int32, device=model_runner.device
        )
        
        self.kv_indptr = torch.zeros(
            (max_bs * num_kv_heads + 1,), dtype=torch.int32, device=model_runner.device
        )
        
        self.kv_last_page_len = torch.ones(
            (max_bs * num_kv_heads,), dtype=torch.int32, device=model_runner.device
        )
        
        # packed_custom_mask_buf when cudagraph is enabled
        self.custom_mask_buf = torch.zeros(
            (model_runner.config.hf_config.max_position_embeddings * max_bs,), dtype=torch.uint8, device=model_runner.device
        )
        
        self.mask_indptr_buf = torch.zeros(
            max_bs * num_kv_heads + 1, dtype=torch.int32, device=model_runner.device
        )
        
        self.cuda_graph_kv_indices = torch.zeros(
            model_runner.config.hf_config.max_position_embeddings * max_bs * num_kv_heads, 
            dtype=torch.int32,
            device=model_runner.device
        ) 
        
        self.prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, 
            "NHD", 
            backend="fa2"
        )
        
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, 
            "NHD", 
            backend="auto"
        )
        
        self.decode_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, 
            "NHD", 
            backend="auto",
        )
        self.decode_cuda_graph_metadata = {}
               
    def register_for_attn(self, service: BaseService):
        methods_to_register = ["attn"]
        for method in methods_to_register:
            self._register_method(method, service)
    
    def prepare_metadata_for_attn_prefill(self, seqs: list[Sequence]):
        context = get_context()
        cu_seqlens_q = context.cu_seqlens_q
        qo_indptr = cu_seqlens_q
        
        kv_indptr = torch.cumsum(
            torch.tensor([0] + [seq.num_cached_blocks for seq in seqs], device="cuda"),
            dim=0,
        ).to(torch.int32)
        kv_page_indices = torch.tensor(
            list(itertools.chain(*[seq.block_table[:seq.num_cached_blocks] for seq in seqs])), device="cuda"
        ).to(torch.int32)
        kv_last_page_lens = torch.tensor(
            [seq.last_block_num_tokens for seq in seqs], device="cuda"
        ).to(torch.int32)
        
        self.prefill_metadata = PrefillMetadata(use_ragged=True, no_prefix=context.no_prefix)
        self.prefill_wrapper_ragged.begin_forward(
            qo_indptr=qo_indptr,
            kv_indptr=qo_indptr, 
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim, 
            causal=True, 
            q_data_type=torch.bfloat16,
        )
        
        self.prefill_wrapper_paged.begin_forward(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_page_indices,
            paged_kv_last_page_len=kv_last_page_lens,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            causal=False, 
            page_size=self.block_size,
            q_data_type=torch.bfloat16,
        )
            
    def prepare_metadata_for_attn_decode(self, seqs: list[Sequence]):
        """See https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout for metadata required for flashinfer kernel"""
        headwise_seq_table = list(itertools.chain(*[seq.get_headwise_block_table() for seq in seqs]))
        
        kv_indptr = torch.cumsum(
            torch.tensor([0] + [len(table) for table in headwise_seq_table], device="cuda"),
            dim=0, dtype=torch.int32
        )
        
        kv_page_indices = torch.tensor(
            list(itertools.chain(*headwise_seq_table)), device="cuda", dtype=torch.int32
        )
        
        kv_last_page_lens = torch.tensor(
            [1] * (len(seqs) * self.num_kv_heads), device="cuda", dtype=torch.int32
        )

        qo_indptr = torch.arange(len(seqs) * self.num_kv_heads + 1, device="cuda", dtype=torch.int32) # .to(torch.int32)
        
        mask_arr = [torch.tensor(seq.headwise_mask, device="cuda", dtype=torch.uint8).transpose(0, 1).contiguous().view(-1) for seq in seqs]
        
        packed_custom_mask = torch.cat(mask_arr, dim=0).contiguous().view(-1)

        self.decode_prefill_wrapper.plan(
            qo_indptr=qo_indptr, 
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_page_indices,
            paged_kv_last_page_len=kv_last_page_lens,
            num_qo_heads=self.num_heads // self.num_kv_heads,
            num_kv_heads=1,
            head_dim_qk=self.head_dim,
            packed_custom_mask=packed_custom_mask, 
            page_size=self.block_size,
            q_data_type=torch.bfloat16,
        )
        
        self.forward_wrapper = self.decode_prefill_wrapper

    def _update_indices(self,
                        bs: int,
                        decode_wrapper: BatchDecodeWithPagedKVCacheWrapper,
                        cu_qo_indptr: torch.Tensor,
                        cu_kv_indptr: torch.Tensor,
                        cu_page_indices: torch.Tensor,
                        cu_packed_custom_mask: torch.Tensor,
    ):

        
        self.qo_indptr[: bs * self.num_kv_heads + 1] = cu_qo_indptr
        self.kv_indptr[: bs * self.num_kv_heads + 1] = cu_kv_indptr
        kv_indices_buf = decode_wrapper._paged_kv_indices_buf
        kv_indices_buf[: cu_page_indices.shape[0]] = cu_page_indices
                
        # packed_custom_mask_buf = decode_wrapper._custom_mask_buf
        # packed_custom_mask_buf[: cu_packed_custom_mask.shape[0]] = cu_packed_custom_mask
        
        decode_wrapper.begin_forward(
            qo_indptr=self.qo_indptr[:bs * self.num_kv_heads + 1], 
            paged_kv_indptr=self.kv_indptr[:bs * self.num_kv_heads + 1],
            paged_kv_indices=cu_page_indices,
            paged_kv_last_page_len=self.kv_last_page_len[:bs * self.num_kv_heads],
            packed_custom_mask=cu_packed_custom_mask,
            num_qo_heads=self.num_heads // self.num_kv_heads,
            num_kv_heads=1,
            head_dim_qk=self.head_dim,
            page_size=self.block_size,
            q_data_type=torch.bfloat16, 
            non_blocking=True,
        )

    def init_forward_metadata_capture_cuda_graph(
        self, 
        bs: int, 
        cu_qo_indptr: torch.Tensor,  
        cu_kv_indptr: torch.Tensor, 
        cu_page_indices: torch.Tensor,
        cu_packed_custom_mask: torch.Tensor, 
    ):
        decode_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_cuda_graph=True, 
            # use_tensor_cores=True, 
            qo_indptr_buf=self.qo_indptr[:bs * self.num_kv_heads + 1],
            paged_kv_indptr_buf=self.kv_indptr[:bs * self.num_kv_heads + 1],
            paged_kv_indices_buf=self.cuda_graph_kv_indices, 
            paged_kv_last_page_len_buf=self.kv_last_page_len[:bs * self.num_kv_heads], 
            custom_mask_buf=self.custom_mask_buf,
            mask_indptr_buf=self.mask_indptr_buf[:bs * self.num_kv_heads + 1],
        )
        
        self._update_indices(
            bs, 
            decode_wrapper, 
            cu_qo_indptr, 
            cu_kv_indptr,
            cu_page_indices, 
            cu_packed_custom_mask
        )

        self.decode_cuda_graph_metadata[bs] = decode_wrapper
        self.forward_wrapper = decode_wrapper
    
    def init_forward_metadata_replay_cuda_graph(
        self, 
        bs: int, 
        cu_qo_indptr: torch.Tensor,  
        cu_kv_indptr: torch.Tensor,
        cu_page_indices: torch.Tensor,
        cu_packed_custom_mask: torch.Tensor,    
    ):
        self._update_indices(
            bs, 
            self.decode_cuda_graph_metadata[bs], 
            cu_qo_indptr, 
            cu_kv_indptr,
            cu_page_indices, 
            cu_packed_custom_mask
        )

    def attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor): 
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()

        k_cache, v_cache = self.k_cache.contiguous(), self.v_cache.contiguous()

        if k_cache.numel() and v_cache.numel():
            # store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
            
        if context.is_prefill:
            store_q_cache(q, self.q_cache, context.query_slot_mapping, context.query_window_pos, is_prefill=True)
            if self.prefill_metadata.no_prefix:
                o = self.prefill_wrapper_ragged.forward(
                    q=q, 
                    k=k, 
                    v=v, 
                    causal=True,
                    sm_scale=self.scale, 
                )
            else:
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q, k, v, causal=True, sm_scale=self.scale,
                )
                o2, s2 = self.prefill_wrapper_paged.forward_return_lse(
                    q, (self.k_cache, self.v_cache), causal=False, sm_scale=self.scale,
                )
                o, _ = merge_state(o1, s1, o2, s2)
        else:    # decode
            store_q_cache(q, self.q_cache, context.query_slot_mapping, is_prefill=False)
            q = q.view(-1, self.num_kv_heads, self.num_heads // self.num_kv_heads, self.head_dim).view(-1, self.num_heads // self.num_kv_heads, self.head_dim)
            k_cache = self.k_cache.view(-1, 1, self.head_dim)
            v_cache = self.v_cache.view(-1, 1, self.head_dim)
            
            if self.if_log_lse:
                o, lse = self.forward_wrapper.forward_return_lse(
                    q, (k_cache, v_cache)
                )
                # append_lse_log(lse)
            else:
                o = self.forward_wrapper.forward(q, (k_cache, v_cache))
                
            o = o.view(-1, self.num_kv_heads, self.num_heads // self.num_kv_heads, self.head_dim).view(-1, self.num_heads, self.head_dim)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o