"""Pure-torch reference implementation of bf16 fused MoE experts.

Slow but correct. Used as the parity oracle for `triton_fused_moe.py` and as the
default `impl="torch"` path during bring-up. The signature is identical to the
Triton entry point so `MoeBackend.run_experts` can swap between them.

Contract (matches docs/moe/design.md §5.4):
- Reads `hidden_states[sorted_token_ids // K]` rather than re-permuting.
- Writes into `intermediate_cache3[:T, :K, :]`, with `topk_weights` already folded
  into the down GEMM. `Combine` is responsible for the final sum across K.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def torch_fused_moe(
    hidden_states: torch.Tensor,   # [T, H]   bf16
    w1: torch.Tensor,              # [E, 2N, H]  bf16
    w2: torch.Tensor,              # [E, H, N]   bf16
    topk_weights: torch.Tensor,    # [T, K]   fp32
    topk_ids: torch.Tensor,        # [T, K]   int32
    sorted_token_ids: torch.Tensor,        # noqa: ARG001 (unused; kept for signature parity)
    expert_ids: torch.Tensor,              # noqa: ARG001
    num_tokens_post_padded: torch.Tensor,  # noqa: ARG001
    intermediate_cache1: torch.Tensor,     # noqa: ARG001 (kept to match triton sig)
    intermediate_cache2: torch.Tensor,     # noqa: ARG001
    intermediate_cache3: torch.Tensor,     # [T_cap, K, H]
    block_size_m: int,                     # noqa: ARG001
) -> torch.Tensor:
    """Returns a view of intermediate_cache3 of shape [T, K, H]."""
    T, H = hidden_states.shape
    E, two_N, _ = w1.shape
    N = two_N // 2
    K = topk_ids.shape[1]

    out = intermediate_cache3[:T, :K]  # [T, K, H]
    out.zero_()

    # Iterate per expert. For non-empty experts: gather rows, GEMM-activate-GEMM,
    # scatter weighted output back into the [T, K, H] cache.
    for e in range(E):
        mask = topk_ids == e                       # [T, K] bool
        rows, ks = mask.nonzero(as_tuple=True)     # each [n_e]
        n_e = rows.numel()
        if n_e == 0:
            continue
        x = hidden_states.index_select(0, rows)    # [n_e, H]
        gate_up = x @ w1[e].T                      # [n_e, 2N]
        gate, up = gate_up.split(N, dim=-1)
        inter = F.silu(gate) * up                  # [n_e, N]
        down = inter @ w2[e].T                     # [n_e, H]
        w = topk_weights[rows, ks].to(down.dtype).unsqueeze(-1)  # [n_e, 1]
        # Scatter weighted output. (rows, ks) is a unique 2D index by construction.
        out[rows, ks] = down * w

    return out
