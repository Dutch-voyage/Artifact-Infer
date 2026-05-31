"""bf16-only Triton fused MoE — vendored from sglang's
`fused_moe_triton/fused_moe_triton_kernels.py` (which is itself adapted from
vllm), with all FP4/FP8/INT8/AWQ/GPTQ/TMA/swap_ab/all-reduce branches removed.

What the kernel does (one launch = one GEMM):
  Inputs:
    A: [T, K]                 (token features in this GEMM dim, "K" here)
    B: [E, N, K]              (per-expert weights, output dim N)
    sorted_token_ids: [T*topk + pad]  int32, output of moe_align_block_size
    expert_ids:       [n_blocks]     int32 (−1 for padding-only blocks)
    topk_weights:     [T*topk]       fp32, only used when MUL_ROUTED_WEIGHT
    num_tokens_post_padded: [1]      int32, device-side scalar (CUDA-graph friendly)
  Output:
    C: [T*topk, N]            (sorted-by-(token,k) when c_sorted=True
                               else indexed by offs_token directly)

Two launches make the full FusedMoE: GEMM1 (gate_up) → silu_and_mul → GEMM2 (down).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---- Hard-coded fixed config (autotuning is F5 in docs/moe/design.md §13) ----
_FIXED_BLOCK_SIZE_N = 64
_FIXED_BLOCK_SIZE_K = 32
_FIXED_GROUP_SIZE_M = 8
_FIXED_NUM_WARPS = 4
_FIXED_NUM_STAGES = 3


@triton.jit
def _bf16_fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    c_sorted: tl.constexpr,
    even_Ks: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if c_sorted:
            c_ptrs = c_ptr + stride_cm * offs_token_id[:, None] + stride_cn * offs_cn[None, :]
        else:
            c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c_ptr.dtype.element_ty), mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am
                      + offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr
              + off_experts * stride_be
              + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs,
                        mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                        other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if c_sorted:
        c_ptrs = c_ptr + stride_cm * offs_token_id[:, None] + stride_cn * offs_cn[None, :]
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _silu_and_mul_kernel(
    in_ptr,
    out_ptr,
    n_rows,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """in : [n_rows, 2N] viewed as gate || up; out : [n_rows, N] = silu(gate) * up."""
    pid = tl.program_id(axis=0)
    if pid >= n_rows:
        return
    offs_n = tl.arange(0, BLOCK)
    mask = offs_n < N
    gate_ptrs = in_ptr + pid * (2 * N) + offs_n
    up_ptrs = in_ptr + pid * (2 * N) + N + offs_n
    out_ptrs = out_ptr + pid * N + offs_n
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    silu_gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
    out = (silu_gate * up).to(out_ptr.dtype.element_ty)
    tl.store(out_ptrs, out, mask=mask)


def _invoke_gemm(
    a: torch.Tensor,             # [M, K_in]
    b: torch.Tensor,              # [E, N_out, K_in]
    c: torch.Tensor,             # [M_out, N_out] where M_out = T*top_k or sorted_pad
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor | None,
    *,
    top_k: int,
    block_size_m: int,
    mul_routed_weight: bool,
    c_sorted: bool,
):
    EM = sorted_token_ids.size(0)
    N = b.size(1)
    K_in = b.size(2)
    assert a.shape[1] == K_in, f"a K {a.shape[1]} != b K {K_in}"
    assert c.shape[1] == N

    grid = (
        triton.cdiv(EM, block_size_m) * triton.cdiv(N, _FIXED_BLOCK_SIZE_N),
    )
    even_Ks = (K_in % _FIXED_BLOCK_SIZE_K) == 0
    _bf16_fused_moe_kernel[grid](
        a, b, c,
        topk_weights if topk_weights is not None else a,  # placeholder when unused
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N=N, K=K_in,
        EM=EM, num_valid_tokens=topk_weights.numel() if topk_weights is not None else (a.size(0) * top_k),
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_be=b.stride(0), stride_bn=b.stride(1), stride_bk=b.stride(2),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=_FIXED_BLOCK_SIZE_N,
        BLOCK_SIZE_K=_FIXED_BLOCK_SIZE_K,
        GROUP_SIZE_M=_FIXED_GROUP_SIZE_M,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        c_sorted=c_sorted,
        even_Ks=even_Ks,
        num_warps=_FIXED_NUM_WARPS,
        num_stages=_FIXED_NUM_STAGES,
    )


def triton_fused_moe(
    hidden_states: torch.Tensor,   # [T, H]   bf16
    w1: torch.Tensor,              # [E, 2N, H]
    w2: torch.Tensor,              # [E, H, N]
    topk_weights: torch.Tensor,    # [T, K]   fp32
    topk_ids: torch.Tensor,        # [T, K]   int32  (unused; used only by Dispatch)  # noqa: ARG001
    sorted_token_ids: torch.Tensor,        # int32
    expert_ids: torch.Tensor,              # int32
    num_tokens_post_padded: torch.Tensor,  # int32 [1]
    intermediate_cache1: torch.Tensor,     # [T_cap, K, 2N]
    intermediate_cache2: torch.Tensor,     # [T_cap, K, N]
    intermediate_cache3: torch.Tensor,     # [T_cap, K, H]
    block_size_m: int,
) -> torch.Tensor:
    """Returns a view of intermediate_cache3 of shape [T, K, H]."""
    T = hidden_states.size(0)
    K_top = topk_weights.size(1)
    N = w1.size(1) // 2
    H = hidden_states.size(1)
    assert w2.size(1) == H and w2.size(2) == N
    assert intermediate_cache1.size(0) >= T and intermediate_cache1.size(1) == K_top and intermediate_cache1.size(2) == 2 * N
    assert intermediate_cache2.size(0) >= T and intermediate_cache2.size(1) == K_top and intermediate_cache2.size(2) == N
    assert intermediate_cache3.size(0) >= T and intermediate_cache3.size(1) == K_top and intermediate_cache3.size(2) == H

    cache1 = intermediate_cache1[:T].view(T * K_top, 2 * N)
    cache2 = intermediate_cache2[:T].view(T * K_top, N)
    cache3 = intermediate_cache3[:T].view(T * K_top, H)
    topk_weights_flat = topk_weights.view(-1)  # [T*K]

    # GEMM1: cache1[offs_token] = hidden_states[offs_token // top_k] @ w1[expert].T
    # c_sorted=False writes outputs at offs_token (the original flat (t,k) id),
    # so cache1, cache2, and cache3 are all indexed in the same coordinate
    # system — making GEMM2's read of cache2 consistent.
    _invoke_gemm(
        a=hidden_states, b=w1, c=cache1,
        sorted_token_ids=sorted_token_ids, expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        topk_weights=topk_weights_flat,  # only used for num_valid_tokens
        top_k=K_top,
        block_size_m=block_size_m,
        mul_routed_weight=False,
        c_sorted=False,
    )

    # silu_and_mul: cache2[i] = silu(cache1[i, :N]) * cache1[i, N:]
    n_rows = cache1.size(0)
    _silu_and_mul_kernel[(n_rows,)](
        cache1, cache2,
        n_rows=n_rows, N=N, BLOCK=triton.next_power_of_2(N),
    )

    # GEMM2: cache3[offs_token] = cache2[offs_token] @ w2[expert].T * topk_weight
    # `a` is now cache2, also indexed by offs_token — set top_k=1 so the
    # kernel's `offs_token // top_k` becomes a no-op.
    _invoke_gemm(
        a=cache2, b=w2, c=cache3,
        sorted_token_ids=sorted_token_ids, expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        topk_weights=topk_weights_flat,
        top_k=1,
        block_size_m=block_size_m,
        mul_routed_weight=True,
        c_sorted=False,
    )

    return cache3.view(T, K_top, H)
