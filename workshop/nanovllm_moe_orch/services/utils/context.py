from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    no_prefix: bool | None = None

    # MoE-relevant (see docs/moe/design.md §5.1)
    num_tokens: int = 0
    moe_t_cap: int = 0
    moe_block_size_m: int = 0
    moe_capture_buf_id: int | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    num_tokens=0,
):
    global _CONTEXT

    if cu_seqlens_k is not None and cu_seqlens_q is not None:
        no_prefix = not torch.any(cu_seqlens_q < cu_seqlens_k).item()
    else:
        no_prefix = None

    # Preserve MoE static fields (set once by MoeBackend at init), only overwrite num_tokens.
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        no_prefix=no_prefix,
        num_tokens=num_tokens,
        moe_t_cap=_CONTEXT.moe_t_cap,
        moe_block_size_m=_CONTEXT.moe_block_size_m,
        moe_capture_buf_id=_CONTEXT.moe_capture_buf_id,
    )


def set_moe_capacity(t_cap: int, block_size_m: int):
    """Called once by MoeBackend.__post_init__ to publish workspace dims."""
    global _CONTEXT
    _CONTEXT.moe_t_cap = t_cap
    _CONTEXT.moe_block_size_m = block_size_m


def reset_context():
    """Clear per-batch fields but keep MoE static capacity (set once at init)."""
    global _CONTEXT
    _CONTEXT = Context(
        moe_t_cap=_CONTEXT.moe_t_cap,
        moe_block_size_m=_CONTEXT.moe_block_size_m,
        moe_capture_buf_id=_CONTEXT.moe_capture_buf_id,
    )
