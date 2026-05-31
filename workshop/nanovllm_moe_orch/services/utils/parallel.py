"""Tensor-Parallel × Expert-Parallel group helpers.

Layout for TP=Tp × EP=Ep on world_size = Tp * Ep ranks:

  rank = ep_rank * Tp + tp_rank

  TP subgroup per ep_rank:  [ep_rank * Tp, ..., ep_rank * Tp + Tp - 1]    (size Tp)
  EP subgroup per tp_rank:  [tp_rank, tp_rank + Tp, ..., tp_rank + (Ep-1)*Tp]  (size Ep)

Layers behave as follows:
  - TP-aware layers (Linear, Embedding, attention) read world_size/rank from the
    TP subgroup → they shard within Tp.
  - EP-aware layers (Dispatch/Experts/Combine in EP-LL/EP-HT) use the EP subgroup
    for all_to_all_single → they shard experts within Ep.
  - Pure EP (Tp=1):  TP subgroup is size-1 (every rank holds full attention),
                     EP subgroup is the world group.
  - Pure TP (Ep=1):  EP subgroup is size-1 (every rank holds all experts),
                     TP subgroup is the world group.
  - Single rank (Tp=Ep=1): both groups are None → fall back to default world group.

`init_parallel_groups(tp_size, world_size)` is the canonical entry point: it
constructs the right subgroups (collectively across all ranks) and stashes them.
After it returns, `get_tp_group()`, `get_ep_group()`, etc. work everywhere.
"""
from __future__ import annotations

import torch.distributed as dist

_TP_GROUP = None      # None means use default world group
_EP_GROUP = None      # None means use default world group


def init_parallel_groups(tp_size: int, world_size: int) -> None:
    """Create TP and EP subgroups for a TP × EP layout.

    All ranks must call this collectively in the same order. After it returns,
    `get_tp_group()` and `get_ep_group()` return this rank's subgroups (or None
    when they would coincide with the world group).
    """
    global _TP_GROUP, _EP_GROUP

    if not dist.is_initialized():
        _TP_GROUP = None
        _EP_GROUP = None
        return

    assert world_size % tp_size == 0, (
        f"world_size={world_size} must be divisible by tp_size={tp_size}"
    )
    ep_size = world_size // tp_size
    my_rank = dist.get_rank()

    # ---- TP subgroups: rank r belongs to TP group (r // tp_size) ----
    # Each TP group is contiguous: [g*tp_size, g*tp_size + tp_size - 1].
    # `dist.new_group` is collective — every rank participates in every call.
    my_tp_group = None
    for g in range(ep_size):
        ranks = list(range(g * tp_size, (g + 1) * tp_size))
        grp = dist.new_group(ranks=ranks)
        if my_rank in ranks:
            my_tp_group = grp

    # ---- EP subgroups: rank r belongs to EP group (r % tp_size) ----
    # Each EP group is strided: [t, t + tp_size, t + 2*tp_size, ...].
    my_ep_group = None
    for t in range(tp_size):
        ranks = list(range(t, world_size, tp_size))
        grp = dist.new_group(ranks=ranks)
        if my_rank in ranks:
            my_ep_group = grp

    _TP_GROUP = my_tp_group
    _EP_GROUP = my_ep_group


def set_tp_group(group) -> None:
    global _TP_GROUP
    _TP_GROUP = group


def set_ep_group(group) -> None:
    global _EP_GROUP
    _EP_GROUP = group


def get_tp_group():
    return _TP_GROUP


def get_ep_group():
    return _EP_GROUP


def _gws(group) -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=group)


def _grank(group) -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank(group=group)


def get_tp_world_size() -> int:
    return _gws(_TP_GROUP)


def get_tp_rank() -> int:
    return _grank(_TP_GROUP)


def get_ep_world_size() -> int:
    return _gws(_EP_GROUP)


def get_ep_rank() -> int:
    return _grank(_EP_GROUP)
