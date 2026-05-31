"""Dispatch — first of the three MoE sub-modules.

Owns the routing gate (`nn.Linear`-equivalent) and produces the per-batch
metadata that `Experts` and `Combine` consume.

Buffers consumed (registered onto this module by ModelRunner via the
orchestrator; see docs/moe/design.md §9):
  - sorted_token_ids_buf, expert_ids_buf, num_tokens_post_padded, cumsum_buffer
  - topk_weights_buf, topk_ids_buf

Output: a `TokMeta` NamedTuple consumed downstream by Experts and Combine.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size
from sgl_kernel import topk_softmax as sgl_topk_softmax

from src.core.artifact import Artifact


class TokMeta(NamedTuple):
    """Per-batch token metadata flowing Dispatch → Experts → Combine."""
    topk_weights: torch.Tensor          # [T, K]   fp32
    topk_ids: torch.Tensor              # [T, K]   int32
    sorted_token_ids: torch.Tensor      # [T*K + pad]    int32
    expert_ids: torch.Tensor            # [(T*K + pad) // BLOCK_M]  int32
    num_tokens_post_padded: torch.Tensor  # [1]    int32 (device scalar)


class Dispatch(Artifact, nn.Module):
    """topk_softmax + moe_align_block_size.

    The routing gate lives on `FusedMoE` (so HF's `mlp.gate.weight` key matches
    directly) — this module receives the precomputed `router_logits` as input.

    Reads (registered):
      - sorted_token_ids_buf, expert_ids_buf, num_tokens_post_padded,
        cumsum_buffer, topk_weights_buf, topk_ids_buf.
    """

    @property
    def name(self) -> str:
        return "Dispatch"

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        block_size_m: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.block_size_m = block_size_m
        self.norm_topk_prob = norm_topk_prob

        # The buffers below are populated at orchestrator-finalize time.
        self.sorted_token_ids_buf: torch.Tensor   # int32
        self.expert_ids_buf: torch.Tensor          # int32
        self.num_tokens_post_padded: torch.Tensor  # int32 [1]
        self.cumsum_buffer: torch.Tensor           # int32 [E + 2]
        self.topk_weights_buf: torch.Tensor        # fp32 [T_cap, K]
        self.topk_ids_buf: torch.Tensor            # int32 [T_cap, K]

    def forward(self, router_logits: torch.Tensor) -> TokMeta:
        """router_logits: [T, E] — bf16 or fp32; sgl_kernel handles all 3 types."""
        T = router_logits.size(0)
        K = self.top_k
        E = self.num_experts
        BLOCK_M = self.block_size_m

        # sgl_kernel.topk_softmax requires fp32 weights and int32 ids.
        topk_weights = self.topk_weights_buf[:T]   # [T, K]  fp32
        topk_ids = self.topk_ids_buf[:T]           # [T, K]  int32
        sgl_topk_softmax(
            topk_weights,
            topk_ids,
            router_logits,
            self.norm_topk_prob,
            0.0,    # moe_softcapping disabled
            None,   # no correction_bias
        )

        # 2) Block alignment over the (T, K) routing fan-out.
        # NB sgl_kernel uses a 1-indexed expert space internally where slot 0
        # is the "filtered/padding" expert; we pass `num_experts + 1` to leave
        # room for all E real experts plus the filter slot. Blocks that map
        # to the filter slot get expert_id = -1 — the consumer Triton kernel
        # is responsible for skipping those.
        # `pad_sorted_token_ids=True` ensures padding positions are filled with
        # `topk_ids.numel()` so the consumer's `offs_token < num_valid_tokens`
        # mask works without relying on stale buffer contents.
        max_padded = T * K + (E + 1) * (BLOCK_M - 1)
        max_blocks = (max_padded + BLOCK_M - 1) // BLOCK_M
        sorted_token_ids = self.sorted_token_ids_buf[:max_padded]
        expert_ids = self.expert_ids_buf[:max_blocks]
        sgl_moe_align_block_size(
            topk_ids,
            E + 1,
            BLOCK_M,
            sorted_token_ids,
            expert_ids,
            self.num_tokens_post_padded,
            self.cumsum_buffer,
            True,  # pad_sorted_token_ids: write `numel` sentinel into padded slots
        )

        return TokMeta(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=self.num_tokens_post_padded,
        )
