"""Triton dispatch kernel for EP-LL.

Replaces `DispatchEPLL`'s host-side bucketing loop with a pure-GPU kernel so
the dispatch is CUDA-graph capturable.

Pattern (mirrors sglang's ep_scatter, simplified for our no-compaction layout):
  - 1 program per token; each program handles the full hidden_size H in registers.
  - For each (token, k), atomic_add on `local_counts[target_rank, target_local]`
    returns a unique slot index. We scatter:
      send_buf[target_rank, target_local, slot, :]  = hidden[t, :]
      original_indices[target_rank, target_local, slot, :] = (t, k)
  - Slots beyond M_max are silently dropped (graph mode can't raise). Caller is
    responsible for sizing M_max conservatively.

Pre-conditions on the caller:
  - local_counts must be zeroed.
  - original_indices should be -1-filled (the kernel only writes valid slots,
    so untouched slots stay -1 and CombineEPLL.scatter_reduce skips them).

Post-conditions:
  - send_buf may have STALE data in unused slots — that's intentional. Those
    rows produce stale outputs through the kernel and reverse a2a, which the
    sender skips because original_indices == -1 there.

Why not zero send_buf:
  - 256 MB write per layer × 48 layers = 12 GB of DRAM traffic per token. Skipping
    saves ~0.5 ms/layer at 25 GB/s HBM. Correctness preserved by the -1 sentinel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ep_ll_dispatch_kernel(
    hidden_ptr,             # [T, H]                  bf16
    topk_ids_ptr,           # [T, K]                  int32
    send_buf_ptr,           # [N, E_local, M_max, H]  bf16   — flat
    original_indices_ptr,   # [N, E_local, M_max, 2]  int32  — flat
    local_counts_ptr,       # [N, E_local]            int32  — must be zero
    T,
    K: tl.constexpr,
    H: tl.constexpr,
    H_PAD: tl.constexpr,    # next_power_of_2(H)
    E_local: tl.constexpr,
    M_max: tl.constexpr,
):
    """Grid: (min(T, MAX_GRID),). Each program iterates over T via stride."""
    start_t = tl.program_id(0)
    grid = tl.num_programs(0)

    offs_h = tl.arange(0, H_PAD)
    mask_h = offs_h < H

    for t_int32 in range(start_t, T, grid):
        t = t_int32.to(tl.int64)

        # Load this token's hidden state once.
        src_data = tl.load(hidden_ptr + t * H + offs_h, mask=mask_h)

        # For each top-k expert, atomic-allocate a slot and scatter.
        for k_int32 in tl.range(0, K, num_stages=2):
            k = k_int32.to(tl.int64)
            eid = tl.load(topk_ids_ptr + t * K + k)
            target_rank = eid // E_local
            target_local = eid - target_rank * E_local

            # Atomic-fetch-add returns the OLD counter value = our slot.
            # Bump first so the counter reflects the true count even if we drop
            # this token (caller can detect overflow by checking count > M_max).
            slot_int32 = tl.atomic_add(
                local_counts_ptr + target_rank * E_local + target_local, 1
            )

            if slot_int32 < M_max:
                slot = slot_int32.to(tl.int64)
                # send_buf[target_rank, target_local, slot, :]
                row_offset = (
                    target_rank * E_local * M_max + target_local * M_max + slot
                ) * H
                tl.store(send_buf_ptr + row_offset + offs_h, src_data, mask=mask_h)

                # original_indices[target_rank, target_local, slot, :] = (t, k)
                idx_offset = (
                    target_rank * E_local * M_max + target_local * M_max + slot
                ) * 2
                tl.store(original_indices_ptr + idx_offset + 0, t.to(tl.int32))
                tl.store(original_indices_ptr + idx_offset + 1, k.to(tl.int32))


def triton_ep_ll_dispatch(
    hidden_states: torch.Tensor,    # [T, H]                    bf16
    topk_ids: torch.Tensor,         # [T, K]                    int32
    send_buf: torch.Tensor,         # [N, E_local, M_max, H]    bf16   (in/out)
    original_indices: torch.Tensor, # [N, E_local, M_max, 2]    int32  (out, must be pre-filled with -1)
    local_counts: torch.Tensor,     # [N, E_local]              int32  (out, must be pre-zeroed)
    M_max: int,
    E_local: int,
) -> None:
    """Bucket tokens into per-(target_rank, local_expert) slots.

    Caller owns all buffers. This function does NOT zero local_counts or fill
    original_indices with -1 — those resets must happen separately
    (typically in DispatchEPLL.forward via Tensor.zero_()/.fill_(-1) which
    are CUDA-graph compatible).
    """
    assert hidden_states.dtype == torch.bfloat16
    assert topk_ids.dtype == torch.int32
    assert send_buf.dtype == torch.bfloat16
    assert original_indices.dtype == torch.int32
    assert local_counts.dtype == torch.int32

    T, H = hidden_states.shape
    K = topk_ids.shape[1]
    N, E, M, H_buf = send_buf.shape
    assert E == E_local
    assert M == M_max
    assert H_buf == H
    assert original_indices.shape == (N, E_local, M_max, 2)
    assert local_counts.shape == (N, E_local)

    # Grid: cap at 4096 to avoid launching way more programs than SMs.
    # Each program processes ceil(T/grid) tokens via stride.
    grid_size = min(T, 4096)
    H_pad = triton.next_power_of_2(H)

    _ep_ll_dispatch_kernel[(grid_size,)](
        hidden_states,
        topk_ids,
        send_buf,
        original_indices,
        local_counts,
        T,
        K=K,
        H=H,
        H_PAD=H_pad,
        E_local=E_local,
        M_max=M_max,
    )
