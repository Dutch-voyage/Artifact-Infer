"""ExpertsEPLL — per-rank expert holder for EP-LL.

Holds only this rank's slice of the global expert weights:
  w1 : [E_local, 2N, H]    bf16
  w2 : [E_local, H, N]     bf16

Weight loading (called by services/utils/loader.py with global expert_id):
  Each call with `expert_id` outside this rank's range is a NO-OP — we just
  drop the tensor on the floor. The loader is oblivious to EP; the slicing
  decision lives entirely here, where the parameter knows its sharding plan.

Forward:
  Calls the MethodCell `self.run_experts_ll` registered by MoeBackend (step 7).
  For now (before orchestrator wiring) the user passes `inner_kernel` directly.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from src.core.artifact import Artifact
from workshop.nanovllm_moe_orch.artifacts.modeling.layers.moe.dispatch_ep_ll import TokMetaEPLL
from workshop.nanovllm_moe_orch.services.utils.parallel import get_ep_rank, get_ep_world_size


class ExpertsEPLL(Artifact, nn.Module):
    @property
    def name(self) -> str:
        return "ExpertsEPLL"

    def __init__(
        self,
        num_experts_global: int,
        hidden_size: int,
        moe_intermediate_size: int,
    ) -> None:
        super().__init__()
        self.E_global = int(num_experts_global)
        self.H = int(hidden_size)
        self.N = int(moe_intermediate_size)

        self.world_size = get_ep_world_size() if dist.is_initialized() else 1
        self.rank = get_ep_rank() if dist.is_initialized() else 0
        assert self.E_global % self.world_size == 0, (
            f"num_experts={self.E_global} must be divisible by ep_size={self.world_size}"
        )
        self.E_local = self.E_global // self.world_size
        self.expert_id_lo = self.rank * self.E_local        # inclusive
        self.expert_id_hi = (self.rank + 1) * self.E_local  # exclusive

        self.w1 = nn.Parameter(
            torch.empty((self.E_local, 2 * self.N, self.H))
        )
        self.w2 = nn.Parameter(
            torch.empty((self.E_local, self.H, self.N))
        )
        self.w1.weight_loader = self._w1_loader
        self.w2.weight_loader = self._w2_loader

        # Set by orchestrator at finalize time (step 7).
        # `run_experts_ll(hidden_recv, w1, w2, masked_m) -> [E_local, N*M_max, H]`
        # For step 6 the test path passes `inner_kernel` directly via forward().

    def _expert_is_local(self, expert_id_global: int) -> bool:
        return self.expert_id_lo <= expert_id_global < self.expert_id_hi

    def _w1_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                   expert_id: int, shard_id: str) -> None:
        if not self._expert_is_local(expert_id):
            return  # belongs to another rank; drop on the floor
        local_id = expert_id - self.expert_id_lo
        if shard_id == "gate":
            param.data[local_id, 0:self.N, :].copy_(loaded_weight)
        elif shard_id == "up":
            param.data[local_id, self.N:2 * self.N, :].copy_(loaded_weight)
        else:
            raise ValueError(f"w1 expects shard_id in {{'gate', 'up'}}, got {shard_id!r}")

    def _w2_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                   expert_id: int, shard_id: str | None) -> None:
        if not self._expert_is_local(expert_id):
            return
        local_id = expert_id - self.expert_id_lo
        param.data[local_id].copy_(loaded_weight)

    def forward(
        self,
        tok_meta: TokMetaEPLL,
        inner_kernel=None,
    ) -> torch.Tensor:
        """Returns expert_out: [E_local, N_ranks * M_max, H].

        `inner_kernel` is provided directly during testing (before orchestrator
        wiring is in place). The wired engine path will set
        `self.run_experts_ll` and ignore this argument.
        """
        if hasattr(self, "run_experts_ll"):
            return self.run_experts_ll(
                hidden_recv=tok_meta.hidden_recv,
                w1=self.w1,
                w2=self.w2,
                masked_m=tok_meta.masked_m,
            )
        assert inner_kernel is not None, (
            "ExpertsEPLL.forward needs either `run_experts_ll` registered "
            "or an `inner_kernel` argument"
        )
        return inner_kernel(tok_meta.hidden_recv, self.w1, self.w2, tok_meta.masked_m)
