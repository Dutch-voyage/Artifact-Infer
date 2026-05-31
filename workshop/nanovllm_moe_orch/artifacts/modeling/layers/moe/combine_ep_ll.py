"""CombineEPLL — Expert-Parallel Low-Latency combine (torch reference).

Contract:
  IN:  expert_out: [E_local, N * M_max, H]   bf16   (output of inner masked GEMM)
       tok_meta : TokMetaEPLL from DispatchEPLL on the same rank
  OUT: out: [T_local, H]   bf16

Steps:
  1. Reshape expert_out [E_local, N*M_max, H] back to [N, E_local, M_max, H].
     The rows `[s*M_max:(s+1)*M_max]` for expert e are destined for source rank s.
  2. Reverse all_to_all_single: produces `output_recv: [N, E_local, M_max, H]` on the
     ORIGINAL sender rank, where slot [r, e, m] holds the output for whatever this
     rank originally sent to (r, e, m).
  3. Scatter-reduce: for each slot, look up `original_indices[r, e, m] = (t, k)`.
     If t == -1, skip. Else accumulate `output_recv[r, e, m] * topk_weights[t, k]`
     into `final[t]`.

The reverse a2a workspace is ALSO the same shape as send_buf — for now we
allocate it lazily; orchestrator wiring (step 7) will move ownership to MoeBackend.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from src.core.artifact import Artifact
from workshop.nanovllm_moe_orch.artifacts.modeling.layers.moe.dispatch_ep_ll import TokMetaEPLL
from workshop.nanovllm_moe_orch.services.utils.parallel import get_ep_group, get_ep_world_size


class CombineEPLL(Artifact, nn.Module):
    @property
    def name(self) -> str:
        return "CombineEPLL"

    def __init__(self, hidden_size: int, top_k: int, num_experts_local: int, m_max: int) -> None:
        super().__init__()
        self.H = int(hidden_size)
        self.K = int(top_k)
        self.E_local = int(num_experts_local)
        self.M_max = int(m_max)
        self.world_size = get_ep_world_size() if dist.is_initialized() else 1

        # Buffers populated by the orchestrator (registered from MoeBackend) in the
        # wired engine path. Standalone tests fall back to lazy allocation below.
        self.rev_send: torch.Tensor                     # [N, E_local, M_max, H]
        self.rev_recv: torch.Tensor                     # [N, E_local, M_max, H]

    def _ensure_buffers(self, dtype: torch.dtype, device: torch.device):
        if hasattr(self, "rev_send") and isinstance(self.rev_send, torch.Tensor):
            return
        N, E, M, H = self.world_size, self.E_local, self.M_max, self.H
        self.rev_send = torch.empty((N, E, M, H), dtype=dtype, device=device)
        self.rev_recv = torch.empty((N, E, M, H), dtype=dtype, device=device)

    def forward(
        self,
        expert_out: torch.Tensor,   # [E_local, N * M_max, H]
        tok_meta: TokMetaEPLL,
    ) -> torch.Tensor:
        N, E, M, H = self.world_size, self.E_local, self.M_max, self.H
        device = expert_out.device
        dtype = expert_out.dtype
        self._ensure_buffers(dtype, device)

        # 1. Write expert_out into rev_send, undoing the dispatch's permute.
        #    expert_out is contiguous in [E, N*M, H] layout (alias of [E, N, M, H]).
        #    rev_send is contiguous in [N, E, M, H] layout. We write through a
        #    transposed VIEW of rev_send so no transient buffer is allocated.
        #    This replaces:
        #        rev_send_view = expert_out.view(E, N, M, H).permute(1,0,2,3).contiguous()
        #        self.rev_send.copy_(rev_send_view)
        #    which materialized a fresh ~1 GB tensor per layer.
        self.rev_send.permute(1, 0, 2, 3).copy_(expert_out.view(E, N, M, H))

        # 2. Reverse a2a: each rank's slice rev_send[r, :, :, :] is sent to rank r,
        #    where it lands as rev_recv[my_rank-as-seen-by-r, :, :, :] on rank r's side.
        #    Conveniently, all_to_all_single with the same shape on both sides handles this.
        if N > 1:
            dist.all_to_all_single(self.rev_recv, self.rev_send, group=get_ep_group())
        else:
            self.rev_recv.copy_(self.rev_send)

        # 3. Scatter-reduce by (t, k) with topk_weights — DENSE (cuda-graph compatible).
        # original_indices[r,e,m,0]=t, ...[r,e,m,1]=k. -1 means "this slot was unused".
        # The boolean-mask version (rec[valid]) produces dynamic shapes that break
        # cuda-graph capture, so we instead:
        #   - clamp t_idx, k_idx to 0 for invalid slots (so .index_add_ has valid indices)
        #   - scale rev_recv IN-PLACE by a (w * valid_mask) factor — invalid rows
        #     become zero, no transient `contrib` tensor (which would be ~1 GB
        #     and OOMs cuda-graph capture on tight budgets).
        # All shapes are static -> graph-friendly.
        out = torch.zeros((tok_meta.T_local, H), dtype=dtype, device=device)
        oi = tok_meta.original_indices  # [N, E, M, 2]   int32
        t_idx = oi[..., 0].view(-1)  # [N*E*M]
        k_idx = oi[..., 1].view(-1)
        rec = self.rev_recv.view(N * E * M, H)

        valid_mask = (t_idx >= 0).to(dtype).unsqueeze(-1)  # [N*E*M, 1]
        t_safe = t_idx.clamp_min(0).long()
        k_safe = k_idx.clamp_min(0).long()
        w = tok_meta.topk_weights[t_safe, k_safe].to(dtype).unsqueeze(-1)  # [N*E*M, 1]
        # In-place scale on the persistent rev_recv buffer (about to be
        # overwritten by the next layer's all_to_all anyway).
        rec.mul_(w * valid_mask)
        out.index_add_(0, t_safe, rec)

        return out
