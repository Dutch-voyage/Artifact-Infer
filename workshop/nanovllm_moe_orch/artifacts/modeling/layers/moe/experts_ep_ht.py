"""ExpertsEPHT — per-rank expert holder for EP-HT.

Same weight-sharding plan as ExpertsEPLL:
  w1 : [E_local, 2N, H]    bf16
  w2 : [E_local, H, N]     bf16
loaded with the global expert_id, dropping non-local experts.

Forward delegates to MoeBackend.run_experts (registered by the orchestrator)
which dispatches to triton_fused_moe — the same inner kernel used by the
single-rank path. EP-HT achieves "expert parallelism" purely by replacing
the surrounding Dispatch/Combine; the kernel itself doesn't know it's running
EP.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from src.core.artifact import Artifact
from workshop.nanovllm_moe_orch.artifacts.modeling.layers.moe.dispatch_ep_ht import TokMetaEPHT
from workshop.nanovllm_moe_orch.services.utils.parallel import get_ep_rank, get_ep_world_size


class ExpertsEPHT(Artifact, nn.Module):
    @property
    def name(self) -> str:
        return "ExpertsEPHT"

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
        assert self.E_global % self.world_size == 0
        self.E_local = self.E_global // self.world_size
        self.expert_id_lo = self.rank * self.E_local
        self.expert_id_hi = (self.rank + 1) * self.E_local

        self.w1 = nn.Parameter(
            torch.empty((self.E_local, 2 * self.N, self.H))
        )
        self.w2 = nn.Parameter(
            torch.empty((self.E_local, self.H, self.N))
        )
        self.w1.weight_loader = self._w1_loader
        self.w2.weight_loader = self._w2_loader

    def _expert_is_local(self, expert_id_global: int) -> bool:
        return self.expert_id_lo <= expert_id_global < self.expert_id_hi

    def _w1_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                   expert_id: int, shard_id: str) -> None:
        if not self._expert_is_local(expert_id):
            return
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

    def forward(self, tok_meta: TokMetaEPHT) -> torch.Tensor:
        """Returns expert_out: [total_recv, 1, H]   bf16.

        Calls MoeBackend.run_experts (set by orchestrator) which itself dispatches
        to triton_fused_moe with the receive-side metadata DispatchEPHT prepared.
        """
        # Empty-recv shortcut: if no tokens routed here, return zero shaped
        # to (0, 1, H). CombineEPHT handles total_recv=0 fine.
        if tok_meta.recv_hidden.shape[0] == 0:
            return torch.zeros(
                (0, 1, self.H),
                dtype=tok_meta.recv_hidden.dtype,
                device=tok_meta.recv_hidden.device,
            )

        assert hasattr(self, "run_experts"), (
            "ExpertsEPHT requires MoeBackend.run_experts registered via the orchestrator"
        )
        # run_experts returns intermediate_cache3 view of shape [T_in, K, H].
        # Here T_in = total_recv, K=1.
        return self.run_experts(
            hidden_states=tok_meta.recv_hidden,
            w1=self.w1,
            w2=self.w2,
            topk_weights=tok_meta.recv_topk_weights,    # [total_recv, 1]   = 1.0
            topk_ids=tok_meta.recv_topk_ids,            # [total_recv, 1]
            sorted_token_ids=tok_meta.sorted_token_ids,
            expert_ids=tok_meta.expert_ids,
            num_tokens_post_padded=tok_meta.num_tokens_post_padded,
        )
