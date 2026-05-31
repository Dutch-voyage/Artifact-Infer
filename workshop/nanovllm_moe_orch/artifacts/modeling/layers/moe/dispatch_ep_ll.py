"""DispatchEPLL — Expert-Parallel Low-Latency dispatcher (torch reference).

Contract (compared with the single-rank `Dispatch`):
  IN:  router_logits: [T, E_global]    (E_global = total experts across all ranks)
  OUT: TokMetaEPLL with:
       - hidden_recv : [E_local, N_ranks * M_max, H]   bf16   (zero-padded)
       - masked_m    : [E_local]                       int32  (= N_ranks * M_max,
                                                       a coarse upper bound; the
                                                       inner kernel processes
                                                       everything; padding rows
                                                       are zero so they produce
                                                       zero output, harmless)
       - topk_weights, topk_ids: kept locally, consumed by Combine
       - original_indices : [N_ranks, E_local, M_max, 2]   int32
                             (target_rank, local_expert, slot) -> (token_idx, k_idx);
                             -1 sentinel for unused slots. Stored on this rank's
                             SEND side so we can reverse-a2a outputs back and
                             scatter-reduce.

Buffer ownership: DispatchEPLL does NOT own buffers. They are allocated in
`MoeBackend` and registered onto this module via the orchestrator (Q4). For
this initial torch reference we let DispatchEPLL allocate them lazily on first
forward — the orchestrator path is wired up in step 7.

EP layout assumption (for now):
  Pure EP. Each rank holds E_global / N_ranks contiguous experts.
  Expert e routes to: target_rank = e // E_local, local_id = e % E_local.

Overflow policy:
  M_max is sized statically. If actual count > M_max, this implementation
  RAISES in eager mode (we do not silently drop). Production would use EPLB
  to keep per-bucket counts bounded.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.distributed as dist
from torch import nn

from src.core.artifact import Artifact
from workshop.nanovllm_moe_orch.services.utils.parallel import (
    get_ep_group,
    get_ep_rank,
    get_ep_world_size,
)


class TokMetaEPLL(NamedTuple):
    """EP-LL metadata flowing DispatchEPLL -> ExpertsEPLL -> CombineEPLL."""
    # Output of dispatch a2a, ready for the inner masked grouped GEMM.
    hidden_recv: torch.Tensor             # [E_local, N * M_max, H]   bf16
    masked_m: torch.Tensor                # [E_local]                 int32

    # Routing decisions (kept locally on the sender; consumed by CombineEPLL).
    topk_weights: torch.Tensor            # [T, K]   fp32
    topk_ids: torch.Tensor                # [T, K]   int32

    # Send-side bookkeeping for the reverse a2a in CombineEPLL.
    original_indices: torch.Tensor        # [N, E_local, M_max, 2]   int32
    T_local: int                          # rows in the local hidden_states


class DispatchEPLL(Artifact, nn.Module):
    @property
    def name(self) -> str:
        return "DispatchEPLL"

    def __init__(
        self,
        num_experts_global: int,
        top_k: int,
        m_max: int,
        norm_topk_prob: bool = True,
        dispatch_kernel: str = "triton",   # "triton" | "torch"
    ) -> None:
        super().__init__()
        self.E_global = int(num_experts_global)
        self.K = int(top_k)
        self.M_max = int(m_max)
        self.norm_topk_prob = norm_topk_prob
        self.dispatch_kernel = dispatch_kernel
        assert dispatch_kernel in ("triton", "torch"), (
            f"dispatch_kernel must be 'triton' or 'torch', got {dispatch_kernel!r}"
        )

        # Use EP subgroup, not world group — matters for TP × EP composition.
        # In pure-EP setups (tp_size=1), EP group == world group.
        self.world_size = get_ep_world_size() if dist.is_initialized() else 1
        self.rank = get_ep_rank() if dist.is_initialized() else 0
        assert self.E_global % self.world_size == 0, (
            f"num_experts={self.E_global} must be divisible by ep_size={self.world_size}"
        )
        self.E_local = self.E_global // self.world_size

        # Buffers are populated at orchestrator-finalize time (registered from MoeBackend).
        # In standalone tests we lazy-allocate via `_ensure_buffers`.
        self.send_buf: torch.Tensor                     # [N, E_local, M_max, H]
        self.recv_buf: torch.Tensor                     # [N, E_local, M_max, H]
        self.original_indices: torch.Tensor             # [N, E_local, M_max, 2]   int32
        self.local_counts: torch.Tensor                 # [N, E_local]             int32
        self.topk_weights_buf: torch.Tensor             # [T_cap, K]   fp32 — optional
        self.topk_ids_buf: torch.Tensor                 # [T_cap, K]   int32 — optional

    def _ensure_buffers(self, H: int, dtype: torch.dtype, device: torch.device):
        """Lazy allocation for standalone tests. The wired engine path skips this
        because the orchestrator has populated the attributes already."""
        if hasattr(self, "send_buf") and isinstance(self.send_buf, torch.Tensor):
            return
        N, E, M = self.world_size, self.E_local, self.M_max
        self.send_buf = torch.zeros((N, E, M, H), dtype=dtype, device=device)
        self.recv_buf = torch.empty((N, E, M, H), dtype=dtype, device=device)
        self.original_indices = torch.full(
            (N, E, M, 2), -1, dtype=torch.int32, device=device,
        )
        self.local_counts = torch.zeros((N, E), dtype=torch.int32, device=device)
        self.hidden_recv = torch.empty((E, N * M, H), dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [T, H]   bf16
        router_logits: torch.Tensor,   # [T, E_global]   bf16 or fp32
    ) -> TokMetaEPLL:
        T, H = hidden_states.shape
        K = self.K
        N = self.world_size
        E_local = self.E_local
        M_max = self.M_max

        self._ensure_buffers(H, hidden_states.dtype, hidden_states.device)

        # ---- topk + softmax (cuda-graph compatible — pure GPU ops) ----
        logits_fp32 = router_logits.float()
        topk_vals, topk_ids = torch.topk(logits_fp32, K, dim=-1)         # [T, K]
        topk_weights = torch.softmax(topk_vals, dim=-1)                  # [T, K]   fp32
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        topk_ids = topk_ids.to(torch.int32)

        # ---- bucket tokens into send_buf ----
        # We DO NOT zero send_buf: stale data in unused slots produces stale outputs
        # through the kernel and reverse a2a, which CombineEPLL.scatter_reduce ignores
        # because original_indices == -1 there. Saves 256 MB of writes per layer.
        self.original_indices.fill_(-1)
        self.local_counts.zero_()

        if self.dispatch_kernel == "triton":
            from workshop.nanovllm_moe_orch.artifacts.moe_backend.triton_ep_ll_dispatch import (
                triton_ep_ll_dispatch,
            )
            triton_ep_ll_dispatch(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                send_buf=self.send_buf,
                original_indices=self.original_indices,
                local_counts=self.local_counts,
                M_max=M_max,
                E_local=E_local,
            )
        else:
            # Torch reference: host-side bucketing loop. Slow, NOT cuda-graph compatible.
            local_counts_cpu = torch.zeros((N, E_local), dtype=torch.int32)
            topk_ids_cpu = topk_ids.cpu()
            for t in range(T):
                for k in range(K):
                    eid = int(topk_ids_cpu[t, k].item())
                    target_rank = eid // E_local
                    target_local = eid % E_local
                    slot = int(local_counts_cpu[target_rank, target_local].item())
                    if slot >= M_max:
                        raise RuntimeError(
                            f"DispatchEPLL overflow: bucket (rank={target_rank}, "
                            f"local_expert={target_local}) exceeded M_max={M_max} "
                            f"at token {t}, k={k}. Increase the recipe LL_M_MAX or apply EPLB."
                        )
                    self.send_buf[target_rank, target_local, slot] = hidden_states[t]
                    self.original_indices[target_rank, target_local, slot, 0] = t
                    self.original_indices[target_rank, target_local, slot, 1] = k
                    local_counts_cpu[target_rank, target_local] += 1
            self.local_counts.copy_(local_counts_cpu)

        # ---- forward all-to-all over the dense buffer ----
        # send_buf[r, e, m, :] is destined for EP-rank r, local expert e.
        # After a2a, recv_buf[s, e, m, :] is what source EP-rank s sent here.
        if N > 1:
            dist.all_to_all_single(self.recv_buf, self.send_buf, group=get_ep_group())
        else:
            self.recv_buf.copy_(self.send_buf)

        # Reshape for the inner masked-grouped-gemm: [E_local, N * M_max, H].
        # Layout: rows [s*M_max : (s+1)*M_max] for expert e are from source rank s.
        # Validity is per-(s, e) — but since unused slots are zero-padded (sender zeroed
        # send_buf and only filled `local_counts[r, e]` slots), the math is correct
        # as long as we set masked_m[e] >= valid count. We use masked_m[e] = N*M_max
        # (process everything, zero rows produce zero output — harmless).
        #
        # Memory note: writing into the persistent `hidden_recv` workspace (instead
        # of `recv_buf.permute(...).contiguous()`) is the difference between a
        # ~1 GB transient allocation per layer (which OOMs on 24 GB cards during
        # cuda-graph warmup) and a single in-place strided copy.
        self.hidden_recv.view(E_local, N, M_max, H).copy_(self.recv_buf.permute(1, 0, 2, 3))
        hidden_recv = self.hidden_recv.view(E_local, N * M_max, H)
        masked_m = torch.full((E_local,), N * M_max, dtype=torch.int32, device=hidden_states.device)

        return TokMetaEPLL(
            hidden_recv=hidden_recv,
            masked_m=masked_m,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            original_indices=self.original_indices,
            T_local=T,
        )
