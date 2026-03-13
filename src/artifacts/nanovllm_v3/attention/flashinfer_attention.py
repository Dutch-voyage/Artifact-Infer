import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.decode import _get_range_buf, get_seq_lens
import itertools
from typing import Optional, Union


from src.services.nanovllm_v3.utils.context import get_context
from src.services.nanovllm_v3.engine.sequence import Sequence

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


def read_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N = slot_mapping.numel()
    num_heads = k_cache.shape[-2]
    head_dim = k_cache.shape[-1]
    D = num_heads * head_dim
    key = torch.empty((N, num_heads, head_dim), dtype=k_cache.dtype, device=k_cache.device)
    value = torch.empty((N, num_heads, head_dim), dtype=v_cache.dtype, device=v_cache.device)
    
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
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # self.k_cache = self.v_cache = torch.tensor([])

        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
        self.workspace_buffer = global_workspace_buffer
        
        self.num_layers = model_runner.config.hf_config.num_hidden_layers
        
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
        
        self.decode_wrappers = [BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=True, 
        )] * self.num_layers
        
        self.decode_cuda_graph_metadata = {layer_id: {} for layer_id in range(self.num_layers)}
        self.forward_wrapper = {layer_id: None for layer_id in range(self.num_layers)}
        
    def register_for_attn(self, service: BaseService):
        methods_to_register = ["attn"]
        for method in methods_to_register:
            self._register_method(method, service)
                
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
            page_size=1,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
        )
    
    def update_indices(self, 
                       bs: int, 
                       decode_wrapper: BatchDecodeWithPagedKVCacheWrapper, 
                       cu_page_indices: torch.Tensor, 
                       seq_lens: torch.Tensor, 
                       ):
        self.kv_indptr[1: bs + 1] = torch.cumsum(seq_lens, dim=0)
        self.kv_indptr = self.kv_indptr[: bs + 1]
        
        kv_indices = decode_wrapper._paged_kv_indices_buf
        kv_indices[: cu_page_indices.shape[0]] = cu_page_indices
        
        decode_wrapper.begin_forward(
            indptr=self.kv_indptr,
            indices=kv_indices,
            last_page_len=self.kv_last_page_len[:bs],
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            q_data_type=torch.bfloat16, 
            non_blocking=True,
        )
        
    def init_forward_metadata_capture_cuda_graph(
        self, 
        bs: int, 
        seq_lens: torch.Tensor, 
        cu_page_indices: torch.Tensor, 
        layer_id: int, 
    ):
        decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_cuda_graph=True, 
            use_tensor_cores=True, 
            paged_kv_indptr_buffer=self.kv_indptr[:bs + 1],
            paged_kv_indices_buffer=self.cuda_graph_kv_indices, 
            paged_kv_last_page_len_buffer=self.kv_last_page_len[:bs] 
        )
        
        self.update_indices(
            bs, 
            decode_wrapper, 
            cu_page_indices, 
            seq_lens
        )
        # TODO look into sglang's patch to find why there is an performance gain in flashinfer plan
        # decode_wrapper.begin_forward = partial(
        #     fast_decode_plan, decode_wrapper
        # )
        self.decode_cuda_graph_metadata[layer_id][bs] = decode_wrapper
        self.forward_wrapper[layer_id] = decode_wrapper
    
    def init_forward_metadata_replay_cuda_graph(
        self, 
        bs: int, 
        seq_lens: torch.Tensor,  
        cu_page_indices: torch.Tensor, 
        layer_id: int, 
    ):
        self.update_indices(
            bs, 
            self.decode_cuda_graph_metadata[layer_id][bs], 
            cu_page_indices, 
            seq_lens[:bs]
        )

    def attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int): 
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
            # self.prepare_metadata(seqs)
            o = self.forward_wrapper[layer_id].forward(q, (self.k_cache, self.v_cache))
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
