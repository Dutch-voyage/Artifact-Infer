import torch
import torch.nn as nn 
import torch.distributed as dist
from .linear import QKVParallelLinear, RowParallelLinear
from .rotary_embedding import apply_rotary_emb_native
from einops import rearrange
from flash_attn import flash_attn_varlen_func

def divide(n: int, k: int) -> int:
    """Helper function to divide n by k and round up."""
    assert n % k == 0, f"{n} is not divisible by {k}"
    return n // k

class VisionFlash3Attention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        bsz: int,
        seq_len: int,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """

        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )

        return output

class VisionAttention(nn.Module):
    r"""
        Multi-headed attention without any cache, mostly used for multimodal transformers.


    Args:
        use_qkv_parallel (bool, optional): If True, use QKV-parallel attention.
        softmax_in_single_precision (bool, default to False):
            if ``True``, the softmax will be performed in single-precision
            Otherwise, it will be performed in half-precision

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        proj_bias: bool = True,
        qkv_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        tp_size = dist.get_world_size()
        self.tp_size = tp_size
        self.head_size = embed_dim // num_heads
        self.hidden_size_per_attention_head = divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = divide(
            num_heads, self.tp_size
        )
        self.num_attention_kv_heads_per_partition = divide(
            num_heads, self.tp_size
        )

        self.q_size = self.num_attention_heads_per_partition * self.head_size
        self.kv_size = self.num_attention_kv_heads_per_partition * self.head_size

        # Additional dummy heads are used to enable TP for common GPU counts.
        self.dummy_dim = (num_heads) * self.head_size

        self.qkv_backend = VisionFlash3Attention()

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_size,
            num_heads,
            num_heads,
            bias=qkv_bias,
        )
        self.proj = RowParallelLinear(
            self.dummy_dim,
            embed_dim,
            bias=proj_bias,
        )


    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens = None,
        position_embeddings  = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim]
            cu_seqlens: [b]
        Returns:
             [s, b, head * head_size]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, x.shape
        x_shape = x.shape
        bsz, s, _ = x_shape
        head = self.num_attention_heads_per_partition
        kv_head = self.num_attention_kv_heads_per_partition

        # [b, s, embed_dim] --> [b, s, embed_dim]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # [b, s, embed_dim] --> [b * s, head, head_size]
        q = q.reshape(bsz * s, head, -1).contiguous()
        k = k.reshape(bsz * s, kv_head, -1).contiguous()
        v = v.reshape(bsz * s, kv_head, -1).contiguous()
        
        original_shape = q.shape
        cos, sin = position_embeddings

        # [total_tokens, head, head_size]
        q = q.view(-1, head, self.head_size)
        k = k.view(-1, head, self.head_size)

        q, k = apply_rotary_emb_native(q, k, cos, sin)

        q = q.view(original_shape)
        k = k.view(original_shape)

        if q.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            q = rearrange(q, "b s ... -> (b s) ...")
        if k.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            k = rearrange(k, "b s ... -> (b s) ...")
        if v.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            v = rearrange(v, "b s ... -> (b s) ...")

        assert q.dim() == 3, q.dim()
        assert k.dim() == 3, k.dim()
        assert v.dim() == 3, v.dim()

        output = self.qkv_backend.forward(
            q=q,
            k=k,
            v=v,
            bsz=bsz,
            seq_len=s,
            cu_seqlens=cu_seqlens,
        )

        assert output.dim() == 3, output.shape

        # [b * s, h, head_size] --> [b, s, h * head_size]
        output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)

        # [b, s, h * head_size] --> [b, s, h * head_size]
        output = self.proj(output)

        return output
