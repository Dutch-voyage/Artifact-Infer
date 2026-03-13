import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
import itertools
from typing import Optional, Union


from src.services.nanovllm_v1.utils.context import get_context
from src.services.nanovllm_v1.engine.sequence import Sequence


from src.core.artifact_base import Artifact
from src.core.service_base import BaseService

from functools import partial 

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
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


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
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        
        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
        self.workspace_buffer = global_workspace_buffer
        
        max_bs = min(model_runner.config.max_num_seqs, 512)
        
        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        
        self.cuda_graph_kv_indices = torch.zeros(
            model_runner.config.hf_config.max_position_embeddings * max_bs, 
            dtype=torch.int32,
            device=model_runner.device
        ) 
        
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=True, 
        )
               
    def register_for_attn(self, service: BaseService):
        methods_to_register = ["attn"]
        for method in methods_to_register:
            self._register_method(method, service)
        objs_to_register = ["k_cache", "v_cache"]
        for obj in objs_to_register:
            self._register_obj(obj, service)
    
    def register_for_runner(self, service: BaseService):
        methods_to_regsiter = ["prepare_metadata_for_attn"]
        for method in methods_to_regsiter:
            self._register_method(method, service)
    
    def prepare_metadata_for_attn(self, seqs: list[Sequence]):
        """See https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout for metadata required for flashinfer kernel"""
        kv_indptr = torch.cumsum(
            torch.tensor([0] + [len(seq.block_table) for seq in seqs], device="cuda"),
            dim=0,
        ).to(torch.int32)
        kv_page_indices = torch.tensor(
            list(itertools.chain(*[seq.block_table for seq in seqs])), device="cuda"
        ).to(torch.int32)
        kv_last_page_lens = torch.tensor(
            [seq.last_block_num_tokens for seq in seqs], device="cuda"
        ).to(torch.int32)
    
        self.decode_wrapper.plan(
            indptr=kv_indptr,
            indices=kv_page_indices,
            last_page_len=kv_last_page_lens,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=256,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
        )
    
    def attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor): 
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = self.forward_wrapper.forward(q, (self.k_cache, self.v_cache))
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
