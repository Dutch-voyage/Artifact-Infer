"""Triton bf16 masked grouped GEMM for EP-LL.

Layout (matches torch_masked_grouped_gemm.py — see that file's docstring):
  hidden_states : [E, M_max, H]   bf16
  w1            : [E, 2N, H]      bf16
  w2            : [E, H, N]       bf16
  masked_m      : [E]             int32

  out           : [E, M_max, H]   bf16   (padding rows are zero)

Implementation strategy:
  We split into two Triton kernels:
    1. `_gate_up_kernel`: hidden @ w1.T -> gate_up [E, M_max, 2N], then
       silu(gate)*up   in-place to produce intermediate [E, M_max, N].
    2. `_down_kernel`:   intermediate @ w2.T -> out [E, M_max, H].

  Both kernels use a 3D grid (BLOCK_M tiles × BLOCK_N tiles × E experts).
  Per-expert masking: each program checks `masked_m[expert_idx]`; if its
  M-tile is fully outside the valid range, it stores zeros and returns.
  Otherwise it computes the GEMM with row-wise masking on writes.

  This is a CORRECT-FIRST kernel: we don't skip FLOPs on padding rows
  inside the K loop (we just mask the writes). The optimization for
  masked variants in DeepGEMM/FlashInfer (skip entire CTA when out of
  range, plus K-loop early termination) is left as a follow-up.

Why two kernels not one fused: H=2048 > N=768 (typical Qwen3-MoE), so the
intermediate buffer fits in DRAM but not in shared mem. Standard pattern.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up_kernel(
    # Pointers
    x_ptr,        # [E, M_max, H]   bf16
    w1_ptr,       # [E, 2N, H]      bf16
    out_ptr,      # [E, M_max, N]   bf16   (intermediate after silu*up)
    masked_m_ptr, # [E]             int32
    # Strides (in elements)
    sx_e, sx_m, sx_h,
    sw_e, sw_n, sw_h,
    so_e, so_m, so_n,
    # Sizes (compile-time)
    M_max: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute out[e, m, n] = silu(x[e, m] @ w1[e, :N].T)[n] * (x[e, m] @ w1[e, N:].T)[n].

    Grid: (cdiv(M_max, BLOCK_M), cdiv(N, BLOCK_N), E).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)

    valid_m = tl.load(masked_m_ptr + pid_e)
    # If the entire M-tile is past the valid range, store zeros and return.
    m_start = pid_m * BLOCK_M
    if m_start >= valid_m:
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M_max
        mask_n = offs_n < N
        out_ptrs = out_ptr + pid_e * so_e + offs_m[:, None] * so_m + offs_n[None, :] * so_n
        tl.store(out_ptrs, tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])
        return

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_base = x_ptr + pid_e * sx_e + offs_m[:, None] * sx_m
    # gate uses rows [0, N) of w1; up uses rows [N, 2N).
    w_gate_base = w1_ptr + pid_e * sw_e + offs_n[None, :] * sw_n
    w_up_base = w1_ptr + pid_e * sw_e + (offs_n[None, :] + N) * sw_n

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m_full = offs_m < M_max  # M_max bound (always ≥ valid_m)
    mask_n = offs_n < N

    for k in range(0, H, BLOCK_K):
        offs_k_iter = k + offs_k
        mask_k = offs_k_iter < H
        x_tile = tl.load(
            x_base + offs_k_iter[None, :] * sx_h,
            mask=mask_m_full[:, None] & mask_k[None, :],
            other=0.0,
        )
        w_gate_tile = tl.load(
            w_gate_base + offs_k_iter[:, None] * sw_h,
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
        )
        w_up_tile = tl.load(
            w_up_base + offs_k_iter[:, None] * sw_h,
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
        )
        acc_gate += tl.dot(x_tile, w_gate_tile)
        acc_up += tl.dot(x_tile, w_up_tile)

    # silu(gate) * up
    gate_sigmoid = tl.sigmoid(acc_gate)
    inter = (acc_gate * gate_sigmoid) * acc_up
    inter_bf16 = inter.to(tl.bfloat16)

    # Store with valid_m masking on rows (zero out padding rows for correctness).
    mask_valid_m = offs_m < valid_m
    inter_out = tl.where(mask_valid_m[:, None], inter_bf16, tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.bfloat16))
    out_ptrs = out_ptr + pid_e * so_e + offs_m[:, None] * so_m + offs_n[None, :] * so_n
    tl.store(out_ptrs, inter_out, mask=mask_m_full[:, None] & mask_n[None, :])


@triton.jit
def _down_kernel(
    inter_ptr,    # [E, M_max, N]   bf16
    w2_ptr,       # [E, H, N]       bf16
    out_ptr,      # [E, M_max, H]   bf16
    masked_m_ptr, # [E]             int32
    si_e, si_m, si_n,
    sw_e, sw_h, sw_n,
    so_e, so_m, so_h,
    M_max: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """out[e, m, h] = sum_k inter[e, m, k] * w2[e, h, k]   over k in [0, N)."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_e = tl.program_id(2)

    valid_m = tl.load(masked_m_ptr + pid_e)
    m_start = pid_m * BLOCK_M
    if m_start >= valid_m:
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_m = offs_m < M_max
        mask_h = offs_h < H
        out_ptrs = out_ptr + pid_e * so_e + offs_m[:, None] * so_m + offs_h[None, :] * so_h
        tl.store(out_ptrs, tl.zeros([BLOCK_M, BLOCK_H], dtype=tl.bfloat16), mask=mask_m[:, None] & mask_h[None, :])
        return

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)

    inter_base = inter_ptr + pid_e * si_e + offs_m[:, None] * si_m
    w2_base = w2_ptr + pid_e * sw_e + offs_h[None, :] * sw_h

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    mask_m_full = offs_m < M_max
    mask_h = offs_h < H

    for k in range(0, N, BLOCK_K):
        offs_k_iter = k + offs_k
        mask_k = offs_k_iter < N
        inter_tile = tl.load(
            inter_base + offs_k_iter[None, :] * si_n,
            mask=mask_m_full[:, None] & mask_k[None, :],
            other=0.0,
        )
        w2_tile = tl.load(
            w2_base + offs_k_iter[:, None] * sw_n,
            mask=mask_h[None, :] & mask_k[:, None],
            other=0.0,
        )
        acc += tl.dot(inter_tile, w2_tile)

    out_bf16 = acc.to(tl.bfloat16)
    mask_valid_m = offs_m < valid_m
    out_masked = tl.where(mask_valid_m[:, None], out_bf16, tl.zeros([BLOCK_M, BLOCK_H], dtype=tl.bfloat16))
    out_ptrs = out_ptr + pid_e * so_e + offs_m[:, None] * so_m + offs_h[None, :] * so_h
    tl.store(out_ptrs, out_masked, mask=mask_m_full[:, None] & mask_h[None, :])


def triton_masked_grouped_gemm(
    hidden_states: torch.Tensor,   # [E, M_max, H]   bf16
    w1: torch.Tensor,              # [E, 2N, H]      bf16
    w2: torch.Tensor,              # [E, H, N]       bf16
    masked_m: torch.Tensor,        # [E]             int32
    out: torch.Tensor | None = None,
    intermediate: torch.Tensor | None = None,  # [E, M_max, N] bf16, optional pre-allocated workspace
    block_m: int = 16,
    block_n: int = 64,
    block_h: int = 64,
    block_k_gate: int = 32,
    block_k_down: int = 32,
) -> torch.Tensor:
    """bf16 masked grouped GEMM. Defaults are conservative; tune if perf matters."""
    assert hidden_states.dtype == torch.bfloat16
    assert w1.dtype == torch.bfloat16
    assert w2.dtype == torch.bfloat16
    assert masked_m.dtype == torch.int32
    E, M_max, H = hidden_states.shape
    _, two_N, _ = w1.shape
    N = two_N // 2
    assert w2.shape == (E, H, N)

    if out is None:
        out = torch.empty((E, M_max, H), dtype=torch.bfloat16, device=hidden_states.device)
    if intermediate is None:
        intermediate = torch.empty((E, M_max, N), dtype=torch.bfloat16, device=hidden_states.device)

    grid_gate = (triton.cdiv(M_max, block_m), triton.cdiv(N, block_n), E)
    _gate_up_kernel[grid_gate](
        hidden_states, w1, intermediate, masked_m,
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        w1.stride(0), w1.stride(1), w1.stride(2),
        intermediate.stride(0), intermediate.stride(1), intermediate.stride(2),
        M_max=M_max, H=H, N=N,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k_gate,
    )

    grid_down = (triton.cdiv(M_max, block_m), triton.cdiv(H, block_h), E)
    _down_kernel[grid_down](
        intermediate, w2, out, masked_m,
        intermediate.stride(0), intermediate.stride(1), intermediate.stride(2),
        w2.stride(0), w2.stride(1), w2.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        M_max=M_max, H=H, N=N,
        BLOCK_M=block_m, BLOCK_H=block_h, BLOCK_K=block_k_down,
    )

    return out
