"""Qwen3-MoE model — `Qwen3MLP` swapped for `FusedMoE`.

Diff from `qwen3.py`:
- `Qwen3MoeDecoderLayer.mlp` is a `FusedMoE` instead of `Qwen3MLP`.
- `packed_modules_mapping` drops `gate_proj`/`up_proj` (the MoE loader branch
  in `services/utils/loader.py` handles per-expert gate/up/down via regex).
- The MoE MLP is `[hidden_size → moe_intermediate_size × 2 → hidden_size]`
  with 128 experts and top-8 routing for Qwen3-30B-A3B.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3MoeConfig

from ..layers.layernorm import RMSNorm
from ..layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from ..layers.rotary_embedding import get_rope
from ..layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from ..layers.moe import FusedMoE
from workshop.nanovllm_moe_orch.services.utils.parallel import get_tp_world_size

from src.core.artifact import Artifact


class Qwen3MoeAttention(Artifact, nn.Module):
    """Identical to Qwen3Attention in qwen3.py — repeated here to avoid an
    import-cycle between the two model files (qwen3.py imports MergedColumn-
    ParallelLinear which is only used by Qwen3MLP)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()

        self.k_cache = self.v_cache = torch.tensor([])

        tp_size = get_tp_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


def _resolve_rope_theta(config) -> float:
    """Qwen3-MoE may put rope_theta directly on the config, *or* nested inside
    `rope_scaling = {"rope_type": "default", "rope_theta": ...}`. Handle both
    so we can pass `rope_scaling=None` to the simple `RotaryEmbedding`."""
    direct = getattr(config, "rope_theta", None)
    if direct is not None:
        return float(direct)
    rs = getattr(config, "rope_scaling", None) or {}
    if isinstance(rs, dict) and rs.get("rope_type", "default") == "default":
        return float(rs.get("rope_theta", 10000.0))
    return 10000.0


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        moe_block_size_m: int,
        moe_mode: str = "single",
        m_max: int = 0,
        ep_ll_dispatch_kernel: str = "triton",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=_resolve_rope_theta(config),
            rope_scaling=None,
        )
        self.mlp = FusedMoE(
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            block_size_m=moe_block_size_m,
            norm_topk_prob=getattr(config, "norm_topk_prob", True),
            moe_mode=moe_mode,
            m_max=m_max,
            ep_ll_dispatch_kernel=ep_ll_dispatch_kernel,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        moe_block_size_m: int,
        moe_mode: str = "single",
        m_max: int = 0,
        ep_ll_dispatch_kernel: str = "triton",
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        mlp_only = getattr(config, "mlp_only_layers", []) or []
        assert not mlp_only, (
            f"This model file only handles all-MoE Qwen3MoE checkpoints; "
            f"got mlp_only_layers={mlp_only}"
        )
        assert getattr(config, "shared_expert_intermediate_size", None) in (None, 0), (
            "Shared experts are not supported in MVP (see docs/moe/design.md §11)."
        )
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(
                config,
                moe_block_size_m=moe_block_size_m,
                moe_mode=moe_mode,
                m_max=m_max,
                ep_ll_dispatch_kernel=ep_ll_dispatch_kernel,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(Artifact, nn.Module):
    # Note: gate_proj/up_proj are intentionally NOT here — the MoE loader
    # branch in services/utils/loader.py routes them per-expert via a regex.
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(
        self,
        config: Qwen3MoeConfig,
        moe_block_size_m: int = 64,
        moe_mode: str = "single",
        m_max: int = 0,
        ep_ll_dispatch_kernel: str = "triton",
    ) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(
            config,
            moe_block_size_m=moe_block_size_m,
            moe_mode=moe_mode,
            m_max=m_max,
            ep_ll_dispatch_kernel=ep_ll_dispatch_kernel,
        )
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
