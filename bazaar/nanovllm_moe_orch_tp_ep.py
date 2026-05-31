"""Recipe: tensor-parallel x expert-parallel MoE serving.

Outer graph:
    BlockManager -> Scheduler -> LLMEngine
    ModelRunner  -> LLMEngine

Local ModelRunner graph:
    Attention -> ModelRunner/layers
    MoeBackend(ep_ll_triton) -> DispatchEPLL + ExpertsEPLL + CombineEPLL
    DispatchEPLL -> triton_ep_ll_dispatch
    ExpertsEPLL.run_experts_ll -> triton_masked_grouped_gemm

`tensor_parallel_size` is a resource argument. It must be provided by the caller
and does not select the implementation; this recipe always uses the EP-LL Triton
local graph.
"""

from __future__ import annotations

from bazaar.nanovllm_moe_orch import (
    SamplingParams,
    describe_outer_graph,
    reject_implementation_kwargs,
    spawn_recipe,
)
from bazaar.nanovllm_moe_orch_ep_ll_triton import (
    BLOCK_SIZE_M,
    DISPATCH_KERNEL,
    LL_M_MAX,
    USE_CUDA_GRAPH,
    wire_components,
)
from src.core.orchestrator import RegistryOrchestrator
from workshop.nanovllm_moe_orch.artifacts.block_mngr.block_manager import BlockManager
from workshop.nanovllm_moe_orch.services.engine.llm_engine import LLMEngine
from workshop.nanovllm_moe_orch.services.engine.scheduler import Scheduler
from workshop.nanovllm_moe_orch.services.model_runner.model_runner import ModelRunner


BACKEND_IMPL = "ep_ll_triton"

LOCAL_GRAPH = """Local ModelRunner graph:
- Attention -> ModelRunner: init/replay metadata, prefill/decode metadata
- Attention -> Qwen3/Qwen3Moe attention layers: attn
- MoeBackend -> DispatchEPLL: send_buf, recv_buf, original_indices, local_counts, topk buffers, hidden_recv
- MoeBackend -> ExpertsEPLL: run_experts_ll
- MoeBackend -> CombineEPLL: rev_send, rev_recv
- DispatchEPLL -> triton_ep_ll_dispatch
- MoeBackend.run_experts_ll -> triton_masked_grouped_gemm
- TP resource split is determined only by tensor_parallel_size
"""


def describe_graph() -> str:
    return describe_outer_graph(LOCAL_GRAPH)


def _require_tensor_parallel_resource(kwargs: dict) -> None:
    if int(kwargs.get("tensor_parallel_size", 1)) <= 1:
        raise ValueError(
            "bazaar.nanovllm_moe_orch_tp_ep requires tensor_parallel_size > 1 "
            "as an engine resource argument."
        )


def combine(**kwargs):
    reject_implementation_kwargs(kwargs)
    _require_tensor_parallel_resource(kwargs)
    orch = RegistryOrchestrator()

    engine = orch.add(LLMEngine(**kwargs))
    config = engine.config
    model_runner = orch.add(
        ModelRunner(
            config,
            wire_components=wire_components,
            use_cuda_graph=USE_CUDA_GRAPH,
        )
    )
    block_manager = orch.add(BlockManager(config.num_kvcache_blocks, config.kvcache_block_size))
    scheduler = orch.add(Scheduler(config))

    engine.model_runner = model_runner
    engine.block_mngr = block_manager
    engine.scheduler = scheduler

    for name in ("can_allocate", "allocate", "can_append", "may_append", "deallocate"):
        orch.register(block_manager, name, scheduler)
    for name in ("add", "schedule", "postprocess", "is_finished"):
        orch.register(scheduler, name, engine)
    orch.register(model_runner, "run", engine)
    orch.finalize()

    return engine, SamplingParams


def spawn_eval(world_size, engine_kwargs, eval_fn, eval_fn_kwargs=None, **kwargs):
    reject_implementation_kwargs(engine_kwargs)
    _require_tensor_parallel_resource(engine_kwargs)
    return spawn_recipe(__name__, world_size, engine_kwargs, eval_fn, eval_fn_kwargs, **kwargs)


__all__ = [
    "BACKEND_IMPL",
    "BLOCK_SIZE_M",
    "DISPATCH_KERNEL",
    "LL_M_MAX",
    "LOCAL_GRAPH",
    "USE_CUDA_GRAPH",
    "combine",
    "describe_graph",
    "spawn_eval",
    "wire_components",
]
