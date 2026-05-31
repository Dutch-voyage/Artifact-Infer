"""Torch reference for the EP-LL inner kernel: bf16 masked grouped GEMM.

Semantics (matches DeepGEMM / FlashInfer cute_dsl masked variants):

  Input layout per receiving rank:
    hidden_states : [E_local, M_max, H]   (rows beyond masked_m[e] are padding)
    w1            : [E_local, 2N, H]
    w2            : [E_local, H, N]
    masked_m      : [E_local]              int32 — number of valid rows per expert

  Output:
    out           : [E_local, M_max, H]   (padding rows are zero or undefined)

  Math per expert e:
    valid = masked_m[e]
    gate_up           = hidden_states[e, :valid] @ w1[e].T          # [valid, 2N]
    gate, up          = gate_up.split(N, dim=-1)
    intermediate      = silu(gate) * up                              # [valid, N]
    out[e, :valid]    = intermediate @ w2[e].T                       # [valid, H]
    out[e, valid:]    = 0  (or undefined; see CORRECTNESS NOTE)

  CORRECTNESS NOTE: We zero the padding rows. The Triton variant may leave them
  with garbage from previous launches — Combine.scatter_reduce only reads the
  valid prefix (lookup via original_indices), so padding is never observed. We
  zero here in torch ref to make tests easier (allclose against zeros).

  topk_weights are NOT applied here. EP-LL applies them inside Combine on the
  source rank, after the reverse all-to-all, when it scatter-reduces by (t, k).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def torch_masked_grouped_gemm(
    hidden_states: torch.Tensor,   # [E, M_max, H]   bf16
    w1: torch.Tensor,              # [E, 2N, H]      bf16
    w2: torch.Tensor,              # [E, H, N]       bf16
    masked_m: torch.Tensor,        # [E]             int32
    out: torch.Tensor | None = None,  # [E, M_max, H] bf16, optional pre-allocated
) -> torch.Tensor:
    """Pure-torch reference. Iterates per-expert (E is small, ≤ 32 in EP-LL setups)."""
    E, M_max, H = hidden_states.shape
    _, two_N, _ = w1.shape
    N = two_N // 2
    assert w2.shape == (E, H, N), f"w2 shape mismatch: {w2.shape} vs ({E}, {H}, {N})"
    assert masked_m.shape == (E,), f"masked_m shape mismatch: {masked_m.shape} vs ({E},)"

    if out is None:
        out = torch.empty((E, M_max, H), dtype=hidden_states.dtype, device=hidden_states.device)
    else:
        assert out.shape == (E, M_max, H)

    out.zero_()  # padding rows stay zero

    masked_m_cpu = masked_m.tolist()  # one host sync — unavoidable in reference path

    for e in range(E):
        m = int(masked_m_cpu[e])
        if m == 0:
            continue
        x = hidden_states[e, :m]                       # [m, H]
        gate_up = x @ w1[e].t()                         # [m, 2N]
        gate, up = gate_up.split(N, dim=-1)
        intermediate = F.silu(gate) * up                # [m, N]
        out[e, :m] = intermediate @ w2[e].t()           # [m, H]

    return out
