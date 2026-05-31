"""Recipe: single-rank MoE with Triton fused-MoE and cuda graph decode.

Outer graph:
    BlockManager -> Scheduler -> LLMEngine
    ModelRunner  -> LLMEngine

Local ModelRunner graph:
    Attention -> ModelRunner/layers
    MoeBackend(triton) -> Dispatch + Experts + Combine
    Experts.run_experts -> triton_fused_moe
"""

from __future__ import annotations

from bazaar.nanovllm_moe_orch import (
    SamplingParams,
    describe_outer_graph,
    reject_implementation_kwargs,
)
from src.core.orchestrator import RegistryOrchestrator
from workshop.nanovllm_moe_orch.artifacts.attention_backend.flashinfer_attention import (
    Attention as FlashinferAttention,
)
from workshop.nanovllm_moe_orch.artifacts.block_mngr.block_manager import BlockManager
from workshop.nanovllm_moe_orch.artifacts.modeling.layers.moe import Combine, Dispatch, Experts
from workshop.nanovllm_moe_orch.artifacts.modeling.models.qwen3 import Qwen3ForCausalLM
from workshop.nanovllm_moe_orch.artifacts.modeling.models.qwen3_moe import Qwen3MoeForCausalLM
from workshop.nanovllm_moe_orch.artifacts.moe_backend import MoeBackend
from workshop.nanovllm_moe_orch.services.engine.llm_engine import LLMEngine
from workshop.nanovllm_moe_orch.services.engine.scheduler import Scheduler
from workshop.nanovllm_moe_orch.services.model_runner.model_runner import ModelRunner


BACKEND_IMPL = "triton"
BLOCK_SIZE_M = 64
USE_CUDA_GRAPH = True

LOCAL_GRAPH = """Local ModelRunner graph:
- Attention -> ModelRunner: init/replay metadata, prefill/decode metadata
- Attention -> Qwen3/Qwen3Moe attention layers: attn
- MoeBackend -> Dispatch: sorted_token_ids_buf, expert_ids_buf, num_tokens_post_padded, cumsum_buffer, topk buffers
- MoeBackend -> Experts: intermediate caches, run_experts
- MoeBackend -> Combine: intermediate_cache3
- MoeBackend.run_experts -> triton_fused_moe
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
                use_cuda_graph=USE_CUDA_GRAPH,
            )
        )
        model = orch.add(
            Qwen3MoeForCausalLM(
                hf,
                moe_block_size_m=BLOCK_SIZE_M,
                moe_mode="single",
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
        if isinstance(module, Dispatch):
            for name in (
                "sorted_token_ids_buf",
                "expert_ids_buf",
                "num_tokens_post_padded",
                "cumsum_buffer",
                "topk_weights_buf",
                "topk_ids_buf",
            ):
                orch.register(moe_backend, name, module)
        if isinstance(module, Experts):
            for name in (
                "intermediate_cache1",
                "intermediate_cache2",
                "intermediate_cache3",
                "run_experts",
            ):
                orch.register(moe_backend, name, module)
        if isinstance(module, Combine):
            orch.register(moe_backend, "intermediate_cache3", module)

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


__all__ = [
    "BACKEND_IMPL",
    "BLOCK_SIZE_M",
    "LOCAL_GRAPH",
    "USE_CUDA_GRAPH",
    "combine",
    "describe_graph",
    "wire_components",
]
