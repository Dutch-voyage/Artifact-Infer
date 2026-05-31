# Artifact-Infer Structure Summary

## Current Structure

- `src/core/` defines the composition mechanism.
- `Artifact` owns state or implementation. It exposes state through `StateCell` and methods through `MethodCell`.
- `BaseService` is also an `Artifact`; it is the public surface that receives registered child methods and states.
- `RegistryOrchestrator` builds a DAG, validates it, then propagates registered cells upward so parent services can call child artifacts directly.
- `DistOrchestrator` extends the registry path with subprocess workers, shared memory control, and `DistMethodCell` method broadcast.
- `bazaar/` defines runnable combinations. Dense/KV-compression variants show the outer engine graph directly. The original MoE entry keeps high-level assembly inside `LLMEngine`; the `nanovllm_moe_orch_*` recipes are the KVC-style alternatives where bazaar owns the outer graph and recipe-provided local model-runner graph.

## Coding Principles

- Put implementation/state in artifacts; expose workflow through services.
- Register only the methods/states required by the parent service.
- Keep runtime dependencies explicit through orchestrator registration, not implicit imports between components.
- Keep implementation choices in recipes. End-to-end config should describe model path, resource limits, parallel size, sequence limits, and layer trimming.
- Reuse one public engine API: `combine(**kwargs) -> (engine, SamplingParams)`.
- Keep distributed execution behind an orchestration boundary.

## Current Composable Assets

- Core assets: `Artifact`, `StateCell`, `MethodCell`, `DistMethodCell`, `BaseService`, `RegistryOrchestrator`, `DistOrchestrator`.
- Engine assets: `LLMEngine`, `Scheduler`, `ModelRunner`, `BlockManager`, `SamplingParams`.
- Attention asset: `FlashinferAttention`.
- KV/compression assets: normal block manager, headwise block manager, query block manager, model-runner compression hooks.
- MoE assets: `MoeBackend`, `FusedMoE`, `Dispatch`, `Experts`, `Combine`, plus EP-LL and EP-HT variants.
- MoE kernels: torch reference fused MoE, Triton fused MoE, torch masked grouped GEMM, Triton masked grouped GEMM.

## MoE Running Shape

`LLMEngine` owns request flow: `add_request -> schedule -> ModelRunner.run -> postprocess`.
Each `bazaar.nanovllm_moe_orch_*` recipe owns the outer `LLMEngine` / `Scheduler` / `ModelRunner` / `BlockManager` registration.
`ModelRunner` owns model execution and accepts a recipe-provided local wiring function that registers attention plus MoE buffers/methods through its own local orchestrator.
`MoeBackend` owns persistent MoE workspaces and method cells; per-layer MoE modules consume them as `Dispatch -> Experts -> Combine`.
