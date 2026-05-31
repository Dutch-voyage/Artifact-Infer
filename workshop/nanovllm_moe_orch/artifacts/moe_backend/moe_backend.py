"""MoeBackend artifact.

Owns the persistent workspaces consumed by Dispatch / Experts / Combine
(see docs/moe/design.md §9 for the buffer ownership policy) and exposes:

  - workspace tensors as StateCells (registered onto each submodule by ModelRunner)
  - `prepare_metadata_for_moe(num_tokens)` — per-batch reset called by ModelRunner
  - `run_experts(...)` — MethodCell registered onto every Experts module

The actual GEMM body lives in `torch_fused_moe.py` (reference) or
`triton_fused_moe.py` (fast). Backend selection is provided explicitly by the
bazaar recipe that wires this artifact into ModelRunner.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from src.core.service import BaseService
from workshop.nanovllm_moe_orch.services.utils.context import set_moe_capacity


class MoeBackend(BaseService):
    @property
    def name(self) -> str:
        return "MoeBackend"

    def __init__(
        self,
        config,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        moe_intermediate_size: int,
        *,
        impl: str,
        block_size_m: int,
        ll_m_max: int = -1,
        use_cuda_graph: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.E = int(num_experts)
        self.K = int(top_k)
        self.H = int(hidden_size)
        self.N = int(moe_intermediate_size)
        self.T_cap = int(config.max_num_batched_tokens)
        self.BLOCK_M = int(block_size_m)
        self.impl = str(impl)
        self.ll_m_max = int(ll_m_max)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.device = torch.device("cuda")
        self.dtype = config.hf_config.torch_dtype if config.hf_config is not None else torch.bfloat16
        self.__post_init__()

    def __post_init__(self) -> None:
        self.is_ep_ll = self.impl in ("ep_ll_torch", "ep_ll_triton")
        self.is_ep_ht = self.impl == "ep_ht"

        if self.is_ep_ll:
            self._init_ep_ll_buffers()
        elif self.is_ep_ht:
            self._init_ep_ht_buffers()
        else:
            self._init_single_rank_buffers()

        # Publish workspace dims to Context so layers can read them globally.
        set_moe_capacity(self.T_cap, self.BLOCK_M)

        # Pick inner-kernel implementation.
        if self.impl == "torch":
            # Reference impl uses Python loops + tensor.nonzero() (host syncs) — not
            # CUDA-graph capturable. Refuse the combo upfront so the user gets a clear
            # error rather than a cryptic "operation failed during capture" later.
            if self.use_cuda_graph:
                raise RuntimeError(
                    "The torch MoE backend is not compatible with cuda graph capture. "
                    "Select an eager torch recipe or a Triton cuda-graph recipe."
                )
            from workshop.nanovllm_moe_orch.artifacts.moe_backend.torch_fused_moe import (
                torch_fused_moe,
            )
            self._fused_moe_fn = torch_fused_moe
        elif self.impl == "triton":
            from workshop.nanovllm_moe_orch.artifacts.moe_backend.triton_fused_moe import (
                triton_fused_moe,
            )
            self._fused_moe_fn = triton_fused_moe
        elif self.impl == "ep_ll_torch":
            # Reference EP-LL inner kernel uses Python loops in DispatchEPLL;
            # not cuda-graph compatible.
            if self.use_cuda_graph:
                raise RuntimeError(
                    "The EP-LL torch backend uses host-side bucketing loops and is "
                    "not compatible with cuda graph capture. Select the eager EP-LL "
                    "torch recipe or the EP-LL Triton recipe."
                )
            from workshop.nanovllm_moe_orch.artifacts.moe_backend.torch_masked_grouped_gemm import (
                torch_masked_grouped_gemm,
            )
            self._inner_kernel_ll = torch_masked_grouped_gemm
        elif self.impl == "ep_ll_triton":
            from workshop.nanovllm_moe_orch.artifacts.moe_backend.triton_masked_grouped_gemm import (
                triton_masked_grouped_gemm,
            )
            self._inner_kernel_ll = triton_masked_grouped_gemm
        elif self.impl == "ep_ht":
            # EP-HT reuses the standard single-rank Triton fused_moe — only the
            # surrounding Dispatch/Combine differ. Variable-size NCCL a2a in
            # DispatchEPHT requires Python int splits → not cuda-graph compatible.
            if self.use_cuda_graph:
                raise RuntimeError(
                    "The EP-HT recipe uses variable-size NCCL all_to_all_single "
                    "with host-known split sizes and is not cuda-graph compatible."
                )
            from workshop.nanovllm_moe_orch.artifacts.moe_backend.triton_fused_moe import (
                triton_fused_moe,
            )
            self._fused_moe_fn = triton_fused_moe
        else:
            raise ValueError(
                f"Unknown MoE backend impl: {self.impl!r} "
                f"(expected 'torch' / 'triton' / 'ep_ll_torch' / 'ep_ll_triton' / 'ep_ht')"
            )

    def _init_single_rank_buffers(self) -> None:
        # === Intermediate caches (per-(token, k) row buffers) ===
        self.intermediate_cache1 = torch.empty(
            (self.T_cap, self.K, 2 * self.N), dtype=self.dtype, device=self.device,
        )
        self.intermediate_cache2 = torch.empty(
            (self.T_cap, self.K, self.N), dtype=self.dtype, device=self.device,
        )
        self.intermediate_cache3 = torch.empty(
            (self.T_cap, self.K, self.H), dtype=self.dtype, device=self.device,
        )

        # === moe_align_block_size buffers (sized for the worst-case batch) ===
        max_padded = self.T_cap * self.K + (self.E + 1) * (self.BLOCK_M - 1)
        max_blocks = (max_padded + self.BLOCK_M - 1) // self.BLOCK_M
        self.sorted_token_ids_buf = torch.empty(
            (max_padded,), dtype=torch.int32, device=self.device,
        )
        self.expert_ids_buf = torch.empty(
            (max_blocks,), dtype=torch.int32, device=self.device,
        )
        self.num_tokens_post_padded = torch.empty(
            (1,), dtype=torch.int32, device=self.device,
        )
        self.cumsum_buffer = torch.empty(
            (self.E + 2,), dtype=torch.int32, device=self.device,
        )

        # === topk outputs ===
        self.topk_weights_buf = torch.empty(
            (self.T_cap, self.K), dtype=torch.float32, device=self.device,
        )
        self.topk_ids_buf = torch.empty(
            (self.T_cap, self.K), dtype=torch.int32, device=self.device,
        )

    def _init_ep_ll_buffers(self) -> None:
        """Allocate persistent EP-LL workspaces, sized once for max load.

        Layout matches DispatchEPLL/CombineEPLL:
          send_buf, recv_buf : [N_ranks, E_local, M_max, H]   bf16
          rev_send,  rev_recv: [N_ranks, E_local, M_max, H]   bf16
          original_indices   : [N_ranks, E_local, M_max, 2]   int32
          masked_m_buf       : [E_local]                       int32

        We DON'T allocate the inner-kernel intermediate buffer
        ([E_local, N_ranks*M_max, N_intermediate]) here — the kernel allocates
        it on each call (small enough it doesn't matter for correctness).
        """
        from workshop.nanovllm_moe_orch.services.utils.parallel import (
            get_ep_world_size, get_ep_rank,
        )
        assert dist.is_initialized(), (
            "MoeBackend with EP-LL requires torch.distributed initialized "
            "(call dist.init_process_group(backend='nccl', ...) before constructing the engine)"
        )
        # Use EP subgroup, not world group — matters for TP × EP composition.
        self.world_size = get_ep_world_size()
        self.rank = get_ep_rank()
        assert self.E % self.world_size == 0, (
            f"num_experts={self.E} must be divisible by ep_size={self.world_size}"
        )
        self.E_local = self.E // self.world_size

        # Auto-size M_max if user didn't specify. Default: 4× the average count
        # under perfectly balanced routing.
        if self.ll_m_max <= 0:
            avg_per_bucket = math.ceil(self.T_cap * self.K / (self.world_size * self.E_local))
            self.M_max = max(8, avg_per_bucket * 4)
        else:
            self.M_max = self.ll_m_max

        N, E_local, M, H = self.world_size, self.E_local, self.M_max, self.H
        self.send_buf = torch.zeros((N, E_local, M, H), dtype=self.dtype, device=self.device)
        self.recv_buf = torch.empty((N, E_local, M, H), dtype=self.dtype, device=self.device)
        self.rev_send = torch.empty((N, E_local, M, H), dtype=self.dtype, device=self.device)
        self.rev_recv = torch.empty((N, E_local, M, H), dtype=self.dtype, device=self.device)
        self.original_indices = torch.full(
            (N, E_local, M, 2), -1, dtype=torch.int32, device=self.device,
        )
        self.local_counts = torch.zeros((N, E_local), dtype=torch.int32, device=self.device)
        self.masked_m_buf = torch.empty((E_local,), dtype=torch.int32, device=self.device)
        # Persistent workspace for the inner-kernel input. Without this, every
        # DispatchEPLL.forward materializes `recv_buf.permute(...).contiguous()` —
        # a fresh 1 GB tensor per layer that blows up cuda-graph capture and
        # warmup memory. With it, dispatch does an in-place transposed copy.
        # Layout: [E_local, N*M, H] same as the kernel expects.
        self.hidden_recv = torch.empty((E_local, N * M, H), dtype=self.dtype, device=self.device)

        # topk buffers — same shapes as single-rank, used by DispatchEPLL.
        self.topk_weights_buf = torch.empty(
            (self.T_cap, self.K), dtype=torch.float32, device=self.device,
        )
        self.topk_ids_buf = torch.empty(
            (self.T_cap, self.K), dtype=torch.int32, device=self.device,
        )

        # Workspace for the masked grouped GEMM: [E_local, N*M_max, N_intermediate].
        # Allocated here so DispatchEPLL/ExpertsEPLL can share it via the orchestrator.
        self.ll_inter_workspace = torch.empty(
            (self.E_local, self.world_size * self.M_max, self.N), dtype=self.dtype, device=self.device,
        )
        self.ll_out_workspace = torch.empty(
            (self.E_local, self.world_size * self.M_max, self.H), dtype=self.dtype, device=self.device,
        )

    def _init_ep_ht_buffers(self) -> None:
        """Allocate persistent EP-HT workspaces.

        EP-HT layout differs from single-rank in two ways:
          1. Inner-kernel intermediate caches are sized for `total_recv` rows
             with K_in=1 (each received row routes to exactly one local expert).
             We size to `2 * T_cap * K_global` as a safety margin against
             routing imbalance — production engines size based on EPLB bounds.
          2. moe_align_block_size buffers are sized for E_local experts and
             worst-case `2 * T_cap * K_global` recv rows.
          3. Variable-size all_to_all payload buffers are NOT pre-allocated —
             they're created per-call in DispatchEPHT/CombineEPHT (sizes
             depend on per-batch routing).
        """
        from workshop.nanovllm_moe_orch.services.utils.parallel import (
            get_ep_world_size, get_ep_rank,
        )
        assert dist.is_initialized(), (
            "MoeBackend with EP-HT requires torch.distributed initialized "
            "(call dist.init_process_group(backend='nccl', ...) before constructing the engine)"
        )
        self.world_size = get_ep_world_size()
        self.rank = get_ep_rank()
        assert self.E % self.world_size == 0, (
            f"num_experts={self.E} must be divisible by ep_size={self.world_size}"
        )
        self.E_local = self.E // self.world_size

        # Worst-case recv rows on this rank; safety margin = 2x balanced average.
        # If exceeded at runtime we'll get a clear assertion in triton_fused_moe.
        self.T_recv_cap = 2 * self.T_cap * self.K
        K_in = 1   # each received row routes to one local expert

        # === Intermediate caches sized for [T_recv_cap, 1, X] ===
        self.intermediate_cache1 = torch.empty(
            (self.T_recv_cap, K_in, 2 * self.N), dtype=self.dtype, device=self.device,
        )
        self.intermediate_cache2 = torch.empty(
            (self.T_recv_cap, K_in, self.N), dtype=self.dtype, device=self.device,
        )
        self.intermediate_cache3 = torch.empty(
            (self.T_recv_cap, K_in, self.H), dtype=self.dtype, device=self.device,
        )

        # === moe_align_block_size buffers sized for E_local experts ===
        max_padded = self.T_recv_cap + (self.E_local + 1) * (self.BLOCK_M - 1)
        max_blocks = (max_padded + self.BLOCK_M - 1) // self.BLOCK_M
        self.sorted_token_ids_buf = torch.empty(
            (max_padded,), dtype=torch.int32, device=self.device,
        )
        self.expert_ids_buf = torch.empty(
            (max_blocks,), dtype=torch.int32, device=self.device,
        )
        self.num_tokens_post_padded = torch.zeros(
            (1,), dtype=torch.int32, device=self.device,
        )
        self.cumsum_buffer = torch.empty(
            (self.E_local + 2,), dtype=torch.int32, device=self.device,
        )

        # topk buffers — same shapes as single-rank.
        self.topk_weights_buf = torch.empty(
            (self.T_cap, self.K), dtype=torch.float32, device=self.device,
        )
        self.topk_ids_buf = torch.empty(
            (self.T_cap, self.K), dtype=torch.int32, device=self.device,
        )

    # ------------------------------------------------------------------------
    # Methods registered onto host modules via the orchestrator
    # ------------------------------------------------------------------------

    def prepare_metadata_for_moe(self, num_tokens: int) -> None:
        """Per-batch reset, called by ModelRunner.prepare_{prefill,decode}."""
        if self.is_ep_ll:
            # No per-batch reset needed for EP-LL: send_buf is fully rewritten by
            # DispatchEPLL on every call (zero + populate). original_indices is also
            # refilled each call.
            return
        # Single-rank Triton path AND EP-HT: the kernel's grid reads
        # num_tokens_post_padded[0] from device memory; zero it so a stale
        # value from the previous batch is never used.
        self.num_tokens_post_padded.zero_()

    def run_experts(
        self,
        hidden_states: torch.Tensor,   # [T, H]
        w1: torch.Tensor,              # [E, 2N, H]
        w2: torch.Tensor,              # [E, H, N]
        topk_weights: torch.Tensor,    # [T, K]    fp32
        topk_ids: torch.Tensor,        # [T, K]    int32
        sorted_token_ids: torch.Tensor,         # int32, prefilled by Dispatch
        expert_ids: torch.Tensor,               # int32
        num_tokens_post_padded: torch.Tensor,   # int32 [1]
    ) -> torch.Tensor:
        """Returns a view of `intermediate_cache3` of shape [T, K, H]."""
        return self._fused_moe_fn(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            intermediate_cache1=self.intermediate_cache1,
            intermediate_cache2=self.intermediate_cache2,
            intermediate_cache3=self.intermediate_cache3,
            block_size_m=self.BLOCK_M,
        )

    def run_experts_ll(
        self,
        hidden_recv: torch.Tensor,    # [E_local, N_ranks * M_max, H]   bf16
        w1: torch.Tensor,             # [E_local, 2N, H]                bf16
        w2: torch.Tensor,             # [E_local, H, N]                 bf16
        masked_m: torch.Tensor,       # [E_local]                       int32
    ) -> torch.Tensor:
        """EP-LL inner kernel call. Returns [E_local, N_ranks * M_max, H]."""
        return self._inner_kernel_ll(
            hidden_states=hidden_recv,
            w1=w1,
            w2=w2,
            masked_m=masked_m,
            out=self.ll_out_workspace,
        ) if self.impl == "ep_ll_torch" else self._inner_kernel_ll(
            hidden_states=hidden_recv,
            w1=w1,
            w2=w2,
            masked_m=masked_m,
            out=self.ll_out_workspace,
            intermediate=self.ll_inter_workspace,
        )
