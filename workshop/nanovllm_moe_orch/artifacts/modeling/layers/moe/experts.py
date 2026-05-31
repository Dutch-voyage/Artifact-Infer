"""Experts — second of the three MoE sub-modules.

Holds the stacked expert weights `w1: [E, 2N, H]` and `w2: [E, H, N]`. Forward
delegates to the `run_experts` MethodCell registered onto it by the orchestrator
(origin = `MoeBackend`, host = this module). The MethodCell resolves the
intermediate caches via the MethodProxy.

Weight loading (see docs/moe/design.md §5.6):
- `Experts.w1.weight_loader(param, loaded, expert_id, shard_id)` accepts shard_id
  in `{"gate", "up"}` and writes into the [0:N) or [N:2N) half of `w1[expert_id]`.
- `Experts.w2.weight_loader(param, loaded, expert_id, shard_id=None)` writes into
  `w2[expert_id]`.
"""

from __future__ import annotations

import torch
from torch import nn

from src.core.artifact import Artifact
from workshop.nanovllm_moe_orch.artifacts.modeling.layers.moe.dispatch import TokMeta


class Experts(Artifact, nn.Module):
    @property
    def name(self) -> str:
        return "Experts"

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size

        # Stacked across experts — vllm/sglang layout.
        # w1 packs gate then up: [E, 2N, H], shard halves [0:N) = gate, [N:2N) = up.
        self.w1 = nn.Parameter(
            torch.empty((num_experts, 2 * moe_intermediate_size, hidden_size))
        )
        self.w2 = nn.Parameter(
            torch.empty((num_experts, hidden_size, moe_intermediate_size))
        )
        # Bind the per-expert weight loaders that loader.py calls.
        self.w1.weight_loader = self._w1_loader
        self.w2.weight_loader = self._w2_loader

        # Buffers + method registered by the orchestrator at finalize() time.
        self.intermediate_cache1: torch.Tensor
        self.intermediate_cache2: torch.Tensor
        self.intermediate_cache3: torch.Tensor
        # `run_experts` is a MethodCell wrapped by the orchestrator and surfaced
        # here as `self.run_experts(...)`.

    def _w1_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                   expert_id: int, shard_id: str) -> None:
        N = self.moe_intermediate_size
        if shard_id == "gate":
            param.data[expert_id, 0:N, :].copy_(loaded_weight)
        elif shard_id == "up":
            param.data[expert_id, N:2 * N, :].copy_(loaded_weight)
        else:
            raise ValueError(f"w1 expects shard_id in {{'gate', 'up'}}, got {shard_id!r}")

    def _w2_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                   expert_id: int, shard_id: str | None) -> None:
        param.data[expert_id].copy_(loaded_weight)

    def forward(self, hidden_states: torch.Tensor, tok_meta: TokMeta) -> torch.Tensor:
        # `run_experts` is a MethodProxy — when called it resolves
        # `self.intermediate_cache{1,2,3}` against the origin (MoeBackend)
        # and `self.w1/w2` against the host (this module).
        return self.run_experts(
            hidden_states=hidden_states,
            w1=self.w1,
            w2=self.w2,
            topk_weights=tok_meta.topk_weights,
            topk_ids=tok_meta.topk_ids,
            sorted_token_ids=tok_meta.sorted_token_ids,
            expert_ids=tok_meta.expert_ids,
            num_tokens_post_padded=tok_meta.num_tokens_post_padded,
        )
