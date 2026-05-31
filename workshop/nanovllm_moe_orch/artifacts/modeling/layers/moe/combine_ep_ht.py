"""CombineEPHT — Expert-Parallel High-Throughput combine (eager-only).

Inverse of DispatchEPHT.

Inputs:
  expert_out: [total_recv, 1, H]  (or [total_recv, H])  bf16
  tok_meta : TokMetaEPHT (kept on this rank since DispatchEPHT)

Steps:
  1. Reverse all_to_all_single on expert_out using REVERSED counts:
     output_split_sizes=send_counts, input_split_sizes=recv_counts.
     -> rev_perm[T*K, H] in the SAME sorted-by-target-rank order we had on send.
  2. Un-permute: gather by sort_perm to get back [T, K, H] layout via index.
  3. Weight-and-reduce: out[t] = sum_k topk_weights[t, k] * unperm[t, k]
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from src.core.artifact import Artifact
from workshop.nanovllm_moe_orch.artifacts.modeling.layers.moe.dispatch_ep_ht import TokMetaEPHT
from workshop.nanovllm_moe_orch.services.utils.parallel import get_ep_group, get_ep_world_size


class CombineEPHT(Artifact, nn.Module):
    @property
    def name(self) -> str:
        return "CombineEPHT"

    def __init__(self, hidden_size: int, top_k: int) -> None:
        super().__init__()
        self.H = int(hidden_size)
        self.K = int(top_k)
        self.world_size = get_ep_world_size() if dist.is_initialized() else 1

    def forward(
        self,
        expert_out: torch.Tensor,    # [total_recv, 1, H]   or   [total_recv, H]
        tok_meta: TokMetaEPHT,
    ) -> torch.Tensor:
        N = self.world_size
        H = self.H
        K = self.K
        T = tok_meta.T_local
        device = expert_out.device
        dtype = expert_out.dtype

        # Squeeze the K=1 dim if present.
        if expert_out.ndim == 3:
            assert expert_out.shape[1] == 1, (
                f"EP-HT expert_out should be [total_recv, 1, H], got {expert_out.shape}"
            )
            expert_out = expert_out.squeeze(1)   # [total_recv, H]

        total_recv = expert_out.shape[0]
        send_total = sum(tok_meta.send_counts)

        # 1. Reverse a2a: send back per-recv-rank slices.
        rev = torch.empty(send_total, H, dtype=dtype, device=device)
        if N > 1:
            dist.all_to_all_single(
                rev, expert_out.contiguous(),
                output_split_sizes=tok_meta.send_counts,
                input_split_sizes=tok_meta.recv_counts,
                group=get_ep_group(),
            )
        else:
            rev.copy_(expert_out)

        # 2. Un-permute. `sort_perm` maps sorted-position -> original (t*K + k).
        # rev[i] corresponds to sorted-position i. Scatter into [T*K, H] at sort_perm[i].
        unperm = torch.empty(T * K, H, dtype=dtype, device=device)
        unperm[tok_meta.sort_perm] = rev

        # 3. Weight-and-reduce: out[t] = sum_k topk_weights[t, k] * unperm[t*K + k].
        unperm_TKH = unperm.view(T, K, H)
        weights = tok_meta.topk_weights.to(dtype).unsqueeze(-1)  # [T, K, 1]
        out = (unperm_TKH * weights).sum(dim=1)                   # [T, H]
        return out
