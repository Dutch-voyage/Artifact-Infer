"""Recipe: EP-LL MoE serving with Triton dispatch/GEMM and cuda graph decode.

Outer graph:
    BlockManager -> Scheduler -> LLMEngine
    ModelRunner  -> LLMEngine

Local ModelRunner graph:
    Attention -> ModelRunner/layers
    MoeBackend(ep_ll_triton) -> DispatchEPLL + ExpertsEPLL + CombineEPLL
    DispatchEPLL -> triton_ep_ll_dispatch
    ExpertsEPLL.run_experts_ll -> triton_masked_grouped_gemm
"""

from __future__ import annotations

from bazaar.nanovllm_moe_orch import (
    SamplingParams,
    describe_outer_graph,
    reject_implementation_kwargs,
    spawn_recipe,
)
from src.core.orchestrator import RegistryOrchestrator
from workshop.nanovllm_moe_orch.artifacts.attention_backend.flashinfer_attention import (
    Attention as FlashinferAttention,
)
from workshop.nanovllm_moe_orch.artifacts.block_mngr.block_manager import BlockManager
from workshop.nanovllm_moe_orch.artifacts.modeling.layers.moe.combine_ep_ll import CombineEPLL
from workshop.nanovllm_moe_orch.artifacts.modeling.layers.moe.dispatch_ep_ll import DispatchEPLL
from workshop.nanovllm_moe_orch.artifacts.modeling.layers.moe.experts_ep_ll import ExpertsEPLL
from workshop.nanovllm_moe_orch.artifacts.modeling.models.qwen3 import Qwen3ForCausalLM
from workshop.nanovllm_moe_orch.artifacts.modeling.models.qwen3_moe import Qwen3MoeForCausalLM
from workshop.nanovllm_moe_orch.artifacts.moe_backend import MoeBackend
from workshop.nanovllm_moe_orch.services.engine.llm_engine import LLMEngine
from workshop.nanovllm_moe_orch.services.engine.scheduler import Scheduler
from workshop.nanovllm_moe_orch.services.model_runner.model_runner import ModelRunner


BACKEND_IMPL = "ep_ll_triton"
BLOCK_SIZE_M = 64
DISPATCH_KERNEL = "triton"
LL_M_MAX = -1
USE_CUDA_GRAPH = True

LOCAL_GRAPH = """Local ModelRunner graph:
- Attention -> ModelRunner: init/replay metadata, prefill/decode metadata
- Attention -> Qwen3/Qwen3Moe attention layers: attn
- MoeBackend -> DispatchEPLL: send_buf, recv_buf, original_indices, local_counts, topk buffers, hidden_recv
- MoeBackend -> ExpertsEPLL: run_experts_ll
- MoeBackend -> CombineEPLL: rev_send, rev_recv
- DispatchEPLL -> triton_ep_ll_dispatch
- MoeBackend.run_experts_ll -> triton_masked_grouped_gemm
"""


def describe_graph() -> str:
    return describe_outer_graph(LOCAL_GRAPH)


def wire_components(orch: RegistryOrchestrator, runner: ModelRunner, config):
    attention = orch.add(FlashinferAttention(config))

    is_moe = getattr(config.hf_config, "model_type", "") == "qwen3_moe"
    if is_moe:
        hf = config.hf_config
        moe_backend = orch.add(
            MoeBackend(
                config=config,
                num_experts=hf.num_experts,
                top_k=hf.num_experts_per_tok,
                hidden_size=hf.hidden_size,
                moe_intermediate_size=hf.moe_intermediate_size,
                impl=BACKEND_IMPL,
                block_size_m=BLOCK_SIZE_M,
                ll_m_max=LL_M_MAX,
                use_cuda_graph=USE_CUDA_GRAPH,
            )
        )
        model = orch.add(
            Qwen3MoeForCausalLM(
                hf,
                moe_block_size_m=BLOCK_SIZE_M,
                moe_mode="ep_ll",
                m_max=moe_backend.M_max,
                ep_ll_dispatch_kernel=DISPATCH_KERNEL,
            )
        )
    else:
        moe_backend = None
        model = orch.add(Qwen3ForCausalLM(config.hf_config))

    orch.register(attention, "init_forward_metadata_capture_cuda_graph", runner)
    orch.register(attention, "init_forward_metadata_replay_cuda_graph", runner)
    orch.register(attention, "prepare_metadata_for_attn_decode", runner)
    orch.register(attention, "prepare_metadata_for_attn_prefill", runner)

    for module in model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            orch.register(attention, "attn", module)

        if moe_backend is None:
            continue
        if isinstance(module, DispatchEPLL):
            for name in (
                "send_buf",
                "recv_buf",
                "original_indices",
                "local_counts",
                "topk_weights_buf",
                "topk_ids_buf",
                "hidden_recv",
            ):
                orch.register(moe_backend, name, module)
        if isinstance(module, ExpertsEPLL):
            orch.register(moe_backend, "run_experts_ll", module)
        if isinstance(module, CombineEPLL):
            for name in ("rev_send", "rev_recv"):
                orch.register(moe_backend, name, module)

    if moe_backend is not None:
        orch.register(moe_backend, "prepare_metadata_for_moe", runner)
    return model


def combine(**kwargs):
    reject_implementation_kwargs(kwargs)
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
