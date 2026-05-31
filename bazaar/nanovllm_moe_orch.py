"""Shared utilities for recipe-first MoE orchestration recipes.

This module intentionally does not assemble an engine. Each concrete recipe in
`bazaar/nanovllm_moe_orch_*.py` shows its own outer graph and ModelRunner-local
graph. The only shared code here is graph-summary formatting and SPMD spawn
boilerplate.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from collections.abc import Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workshop.nanovllm_moe_orch.services.sampling_params import SamplingParams


IMPLEMENTATION_KWARGS = {
    "moe_impl",
    "moe_block_size_m",
    "moe_ll_m_max",
    "enforce_eager",
    "ep_ll_dispatch_kernel",
}


ENGINE_BINDINGS = (
    (
        "BlockManager",
        "Scheduler",
        ("can_allocate", "allocate", "can_append", "may_append", "deallocate"),
    ),
    ("Scheduler", "LLMEngine", ("add", "schedule", "postprocess", "is_finished")),
    ("ModelRunner", "LLMEngine", ("run",)),
)


def describe_outer_graph(local_graph: str | None = None) -> str:
    lines = ["Outer graph:"]
    for source, target, attrs in ENGINE_BINDINGS:
        lines.append(f"- {source} -> {target}: {', '.join(attrs)}")
    if local_graph:
        lines.append("")
        lines.append(local_graph.strip())
    return "\n".join(lines)


def reject_implementation_kwargs(kwargs: dict) -> None:
    rejected = sorted(set(kwargs) & IMPLEMENTATION_KWARGS)
    if rejected:
        joined = ", ".join(rejected)
        raise TypeError(
            f"Implementation choices are selected by the recipe, not engine kwargs: {joined}"
        )


def _spmd_worker(
    rank: int,
    world_size: int,
    recipe_module: str,
    eval_fn: Callable,
    eval_fn_kwargs: dict,
    engine_kwargs: dict,
    master_addr: str,
    master_port: str,
):
    try:
        import torch
        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )
        torch.set_default_device(f"cuda:{rank}")

        recipe = importlib.import_module(recipe_module)
        engine, sampling_params_cls = recipe.combine(**engine_kwargs)
        eval_fn(engine, sampling_params_cls, rank, world_size, **eval_fn_kwargs)
        sys.exit(0)
    except Exception as exc:
        print(f"[rank {rank}] EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def spawn_recipe(
    recipe_module: str,
    world_size: int,
    engine_kwargs: dict,
    eval_fn: Callable,
    eval_fn_kwargs: dict | None = None,
    *,
    master_addr: str = "127.0.0.1",
    master_port: str = "29555",
):
    """Spawn SPMD workers for a concrete recipe module."""
    import torch.multiprocessing as mp

    if eval_fn_kwargs is None:
        eval_fn_kwargs = {}

    print(
        f"[{recipe_module}] spawning {world_size} workers "
        f"(tp_size={engine_kwargs.get('tensor_parallel_size', 1)})",
        flush=True,
    )
    mp.spawn(
        _spmd_worker,
        args=(
            world_size,
            recipe_module,
            eval_fn,
            eval_fn_kwargs,
            engine_kwargs,
            master_addr,
            master_port,
        ),
        nprocs=world_size,
        join=True,
    )


__all__ = [
    "ENGINE_BINDINGS",
    "IMPLEMENTATION_KWARGS",
    "SamplingParams",
    "describe_outer_graph",
    "reject_implementation_kwargs",
    "spawn_recipe",
]
