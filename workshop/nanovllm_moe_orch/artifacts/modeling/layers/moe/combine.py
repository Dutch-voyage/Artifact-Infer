"""Combine — third of the three MoE sub-modules.

K-axis reduction over the [T, K, H] expert output. Uses `sgl_kernel.moe_sum`
(C++/CUDA, in-place) when available; falls back to `torch.sum`.

`Combine` is a Module (not a free function) so it can:
  1. Read its `intermediate_cache3` buffer through the orchestrator (Q4).
  2. Be uniformly addressed by the orchestrator's `model.modules()` walk.
  3. Be swapped for an EP-HT version that owns a reverse all-to-all workspace
     (see docs/moe/design.md §6, §7.2) without changing the call site.
"""

from __future__ import annotations

import torch
from torch import nn

from sgl_kernel import moe_sum as sgl_moe_sum

from src.core.artifact import Artifact
from workshop.nanovllm_moe_orch.services.utils.context import get_context


class Combine(Artifact, nn.Module):
    @property
    def name(self) -> str:
        return "Combine"

    def __init__(self, hidden_size: int, top_k: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k
        # Registered by the orchestrator.
        self.intermediate_cache3: torch.Tensor

    def forward(self, expert_out_TKH: torch.Tensor) -> torch.Tensor:
        """expert_out_TKH: a view of intermediate_cache3 of shape [T, K, H].

        We could derive T from the input tensor itself, but reading it from
        Context is symmetric with how Dispatch reads `num_tokens_post_padded`
        — i.e., the layer is shape-agnostic and never queries `.size(0)`.
        """
        T = expert_out_TKH.size(0)
        H = self.hidden_size
        out = torch.empty((T, H), dtype=expert_out_TKH.dtype, device=expert_out_TKH.device)
        # sgl_moe_sum: input [T, K, H] -> output [T, H]
        sgl_moe_sum(expert_out_TKH, out)
        return out
