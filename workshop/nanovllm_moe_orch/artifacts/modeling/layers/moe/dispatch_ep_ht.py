"""DispatchEPHT — Expert-Parallel High-Throughput dispatcher (eager-only).

Differs from EP-LL in three key ways:
  1. Send/recv buffers are RAGGED (compact, no zero padding) — variable per-rank
     send sizes computed via bincount + host sync.
  2. Receive side runs the STANDARD single-rank fused_moe (block-tiled triton)
     by invoking sgl_kernel.moe_align_block_size on the received local-expert ids.
  3. NOT cuda-graph compatible (variable-size NCCL a2a needs Python int lists).
     The trade-off vs EP-LL: lower kernel FLOPs (no zero padding), higher latency
     per layer (host sync + variable a2a overhead), so this wins for large batches.

Pipeline (sender side):
  hidden[T, H], router_logits[T, E_global]
  -> topk_ids[T, K], topk_w[T, K]
  -> target_rank[T*K] = topk_ids // E_local
  -> sort token-replicas by target_rank:
       sorted_hidden  : [T*K, H]
       sorted_eid     : [T*K]            local expert id at destination
       sorted_topk_w  : [T*K]            weight for the (t, k) replica
       sort_perm      : [T*K]            inverse map for combine
  -> send_counts[N] = bincount(target_rank)
  -> all_to_all on counts to learn recv_counts[N]
  -> host-sync both, then all_to_all_single (3 calls, one per payload) with
     explicit input/output split sizes
  -> recv_hidden[total_recv, H], recv_eid[total_recv], recv_topk_w[total_recv]
  -> sgl_kernel.moe_align_block_size on recv_eid to build the standard
     (sorted_token_ids, expert_ids, num_tokens_post_padded) for the inner kernel.

Buffers:
  - All payloads are dynamically sized per-call (no persistent ragged buffer).
  - The receive-side (sorted_token_ids etc.) buffers are sized for
    T_cap * K worst-case.
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


class TokMetaEPHT(NamedTuple):
    """EP-HT metadata flowing DispatchEPHT -> ExpertsEPHT -> CombineEPHT."""
    # Output of dispatch a2a, ready for the standard single-rank fused_moe.
    recv_hidden: torch.Tensor          # [total_recv, H]              bf16
    recv_topk_ids: torch.Tensor        # [total_recv, 1]              int32   (local expert id)
    recv_topk_weights: torch.Tensor    # [total_recv, 1]              fp32    (always 1.0)
    sorted_token_ids: torch.Tensor     # int32, output of moe_align
    expert_ids: torch.Tensor           # int32
    num_tokens_post_padded: torch.Tensor  # int32 [1]

    # Per-(t,k) topk weights for combine (kept on this rank).
    topk_weights: torch.Tensor         # [T, K]   fp32
    topk_ids: torch.Tensor             # [T, K]   int32

    # Sort permutation: position-in-sorted -> (t * K + k). Needed by combine
    # to undo the permutation on the reverse path.
    sort_perm: torch.Tensor            # [T*K]   int64
    send_counts: list[int]             # [N]     host-side
    recv_counts: list[int]             # [N]     host-side
    T_local: int


class DispatchEPHT(Artifact, nn.Module):
    @property
    def name(self) -> str:
        return "DispatchEPHT"

    def __init__(
        self,
        num_experts_global: int,
        top_k: int,
        block_size_m: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.E_global = int(num_experts_global)
        self.K = int(top_k)
        self.BLOCK_M = int(block_size_m)
        self.norm_topk_prob = norm_topk_prob

        self.world_size = get_ep_world_size() if dist.is_initialized() else 1
        self.rank = get_ep_rank() if dist.is_initialized() else 0
        assert self.E_global % self.world_size == 0, (
            f"num_experts={self.E_global} must be divisible by ep_size={self.world_size}"
        )
        self.E_local = self.E_global // self.world_size

        # Buffers populated by orchestrator (registered from MoeBackend).
        self.sorted_token_ids_buf: torch.Tensor    # int32
        self.expert_ids_buf: torch.Tensor          # int32
        self.num_tokens_post_padded: torch.Tensor  # int32 [1]
        self.cumsum_buffer: torch.Tensor           # int32 [E_local + 2]

    def _ensure_buffers(self, T_cap: int, device: torch.device):
        """Lazy allocation for standalone tests."""
        if hasattr(self, "sorted_token_ids_buf") and isinstance(
            self.sorted_token_ids_buf, torch.Tensor
        ):
            return
        # Worst case: total_recv = T_cap * K (every (t, k) replica lands here).
        max_in = T_cap * self.K
        max_padded = max_in + (self.E_local + 1) * (self.BLOCK_M - 1)
        max_blocks = (max_padded + self.BLOCK_M - 1) // self.BLOCK_M
        self.sorted_token_ids_buf = torch.empty(max_padded, dtype=torch.int32, device=device)
        self.expert_ids_buf = torch.empty(max_blocks, dtype=torch.int32, device=device)
        self.num_tokens_post_padded = torch.zeros(1, dtype=torch.int32, device=device)
        self.cumsum_buffer = torch.empty(self.E_local + 2, dtype=torch.int32, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [T, H]   bf16
        router_logits: torch.Tensor,   # [T, E_global]
    ) -> TokMetaEPHT:
        from sgl_kernel import moe_align_block_size

        T, H = hidden_states.shape
        K = self.K
        N = self.world_size
        E_local = self.E_local
        device = hidden_states.device
        dtype = hidden_states.dtype

        T_cap = T  # for standalone tests; lazy buffers sized to current T_cap is fine
        self._ensure_buffers(T_cap, device)

        # ---- 1. topk + softmax (cuda-graph compatible — pure GPU ops) ----
        logits_fp32 = router_logits.float()
        topk_vals, topk_ids = torch.topk(logits_fp32, K, dim=-1)         # [T, K]
        topk_weights = torch.softmax(topk_vals, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        topk_ids = topk_ids.to(torch.int32)

        # ---- 2. flatten + sort by target rank ----
        flat_eid = topk_ids.view(-1)                           # [T*K]   int32
        target_rank = (flat_eid // E_local).long()             # [T*K]
        flat_topk_w = topk_weights.view(-1)                    # [T*K]   fp32
        flat_local_eid = (flat_eid - target_rank.to(torch.int32) * E_local).to(torch.int32)

        # Sort permutation: tokens going to the same rank are now contiguous.
        sort_perm = torch.argsort(target_rank, stable=True)    # [T*K]
        sorted_local_eid = flat_local_eid[sort_perm].contiguous()
        sorted_topk_w = flat_topk_w[sort_perm].contiguous()

        # Replicate hidden by K, then permute. We do `repeat_interleave` then index_select
        # to get the right sorted ordering. Slightly wasteful for memory but simple.
        # Replication: hidden[t] appears K times consecutively at flat positions [t*K, t*K+K).
        # sort_perm maps sorted-position -> flat-position; flat-position // K -> token index.
        sorted_token_idx = (sort_perm // K).contiguous()               # [T*K]   int64
        sorted_hidden = hidden_states.index_select(0, sorted_token_idx)  # [T*K, H]   bf16

        # ---- 3. Per-rank counts, exchange them ----
        send_counts_t = torch.bincount(target_rank, minlength=N).to(torch.int32)  # [N]
        recv_counts_t = torch.empty_like(send_counts_t)
        if N > 1:
            dist.all_to_all_single(recv_counts_t, send_counts_t, group=get_ep_group())
        else:
            recv_counts_t.copy_(send_counts_t)

        # Host sync — REQUIRED to feed Python int lists to all_to_all_single.
        # This is what makes EP-HT incompatible with cuda graph.
        send_counts = send_counts_t.tolist()
        recv_counts = recv_counts_t.tolist()
        total_recv = int(sum(recv_counts))

        # ---- 4. Variable-size all_to_all on the three payloads ----
        recv_hidden = torch.empty(total_recv, H, dtype=dtype, device=device)
        recv_local_eid = torch.empty(total_recv, dtype=torch.int32, device=device)
        recv_topk_w = torch.empty(total_recv, dtype=torch.float32, device=device)

        if N > 1:
            ep_grp = get_ep_group()
            dist.all_to_all_single(
                recv_hidden, sorted_hidden,
                output_split_sizes=recv_counts, input_split_sizes=send_counts,
                group=ep_grp,
            )
            dist.all_to_all_single(
                recv_local_eid, sorted_local_eid,
                output_split_sizes=recv_counts, input_split_sizes=send_counts,
                group=ep_grp,
            )
            dist.all_to_all_single(
                recv_topk_w, sorted_topk_w,
                output_split_sizes=recv_counts, input_split_sizes=send_counts,
                group=ep_grp,
            )
        else:
            recv_hidden.copy_(sorted_hidden)
            recv_local_eid.copy_(sorted_local_eid)
            recv_topk_w.copy_(sorted_topk_w)

        # ---- 5. Build standard fused_moe metadata on recv tokens ----
        # Treat as K=1 routing: each received row goes to exactly one local expert.
        recv_topk_ids = recv_local_eid.view(-1, 1)                # [total_recv, 1]
        # Inner kernel applies weights via topk_weights * row_output. We pass 1.0
        # here and let CombineEPHT apply the real weights after the reverse a2a
        # (saves precision; the 1.0 weight just means "don't reweight in kernel").
        recv_topk_weights = torch.ones_like(recv_topk_w).view(-1, 1)

        # If total_recv is 0 (no tokens routed to this rank's experts), fall back
        # to a no-op: skip moe_align_block_size and let CombineEPHT see an empty
        # expert_out.
        if total_recv > 0:
            self.num_tokens_post_padded.zero_()
            moe_align_block_size(
                recv_topk_ids,
                E_local,
                self.BLOCK_M,
                self.sorted_token_ids_buf,
                self.expert_ids_buf,
                self.num_tokens_post_padded,
                self.cumsum_buffer,
            )

        return TokMetaEPHT(
            recv_hidden=recv_hidden,
            recv_topk_ids=recv_topk_ids,
            recv_topk_weights=recv_topk_weights,
            sorted_token_ids=self.sorted_token_ids_buf,
            expert_ids=self.expert_ids_buf,
            num_tokens_post_padded=self.num_tokens_post_padded,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sort_perm=sort_perm,
            send_counts=send_counts,
            recv_counts=recv_counts,
            T_local=T,
        )
