# nanovllm_moe_orch

KVC-style MoE serving variant.

This tree is copied from `workshop/nanovllm_moe`, but the outer engine graph is
assembled from `bazaar/` instead of inside `LLMEngine`.

Boundary:

- `LLMEngine`: tokenizer, request loop, `step`, `generate`, `reset`.
- `bazaar/nanovllm_moe_orch_*.py`: concrete recipes. Each one wires the outer
  running graph (`LLMEngine`, `Scheduler`, `ModelRunner`, `BlockManager`) and
  passes a local wiring function into `ModelRunner`.
- `ModelRunner`: in-instance/local orchestrator for model architecture choices:
  `FlashinferAttention`, dense/MoE model, `MoeBackend`, `Dispatch`, `Experts`, `Combine`.
- `MoeBackend`: persistent MoE workspaces and expert-kernel methods.

Implementation choices such as eager vs cuda graph, MoE backend, EP layout,
dispatch kernel, and MoE block size are recipe constants, not `Config` fields.

This variant intentionally leaves `src/core/` unchanged.
