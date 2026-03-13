from dataclasses import dataclass
import torch

CUDA_GRAPH_ENABLED = False


def set_cuda_graph_flag():
    global CUDA_GRAPH_ENABLED
    CUDA_GRAPH_ENABLED = True


def get_cuda_graph_flag():
    global CUDA_GRAPH_ENABLED
    return CUDA_GRAPH_ENABLED


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    query_slot_mapping: torch.Tensor | None = None
    query_window_pos: torch.Tensor | None = None
    no_prefix: bool | None = None
    packed_headwise_mask: dict[int, torch.Tensor] | None = None
    mask_indptr: dict[int, torch.Tensor] | None = None


_CONTEXT = Context()

def init_packed_wise_mask_for_cudagraph(num_layers, max_num_seqs, max_context_len):
    packed_headwise_mask = torch.zeros((num_layers, max_num_seqs * max_context_len), device="cuda", dtype=torch.uint8)
    global _CONTEXT
    _CONTEXT.packed_headwise_mask = packed_headwise_mask

def get_context():
    global _CONTEXT
    return _CONTEXT

def set_context_replace(context):
    global _CONTEXT
    _CONTEXT = context

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    query_slot_mapping=None,
    query_window_pos=None,
    no_prefix=None, 
):
    global _CONTEXT
    if cu_seqlens_k is not None and cu_seqlens_q is not None:
        no_prefix = not torch.any(cu_seqlens_q < cu_seqlens_k).item()
    else:
        no_prefix = None

    _CONTEXT.is_prefill = is_prefill
    _CONTEXT.cu_seqlens_q = cu_seqlens_q
    _CONTEXT.cu_seqlens_k = cu_seqlens_k
    _CONTEXT.max_seqlen_q = max_seqlen_q
    _CONTEXT.max_seqlen_k = max_seqlen_k
    _CONTEXT.slot_mapping = slot_mapping
    _CONTEXT.context_lens = context_lens
    _CONTEXT.query_slot_mapping = query_slot_mapping
    _CONTEXT.query_window_pos = query_window_pos
    _CONTEXT.no_prefix = no_prefix



def reset_context():
    global _CONTEXT
    _CONTEXT.is_prefill = False
    _CONTEXT.cu_seqlens_q = None
    _CONTEXT.cu_seqlens_k = None
    _CONTEXT.max_seqlen_q = 0
    _CONTEXT.max_seqlen_k = 0
    _CONTEXT.slot_mapping = None
    _CONTEXT.context_lens = None
    _CONTEXT.query_slot_mapping = None
    _CONTEXT.query_window_pos = None
    _CONTEXT.no_prefix = None
    _CONTEXT.packed_headwise_mask = None
    _CONTEXT.mask_indptr = None
