# Bazaar Recipes

`bazaar/` is the project cookbook. A recipe is a small runnable composition that
shows how engine components are connected for one inference design. It should be
readable as a programming guide, not only as an entrypoint.

The MoE recipe-first variants are:

| Structure | Recipe | Purpose |
| --- | --- | --- |
| `single_triton` | `nanovllm_moe_orch_single_triton.py` | single-rank MoE, Triton fused MoE, cuda graph decode |
| `single_torch` | `nanovllm_moe_orch_single_torch.py` | single-rank reference/debug path, eager only |
| `ep_ll_triton` | `nanovllm_moe_orch_ep_ll_triton.py` | EP low-latency path, Triton dispatch and masked grouped GEMM |
| `ep_ll_torch` | `nanovllm_moe_orch_ep_ll_torch.py` | EP low-latency reference/debug path, eager only |
| `ep_ht` | `nanovllm_moe_orch_ep_ht.py` | EP high-throughput path, ragged all-to-all, eager only |
| `tp_ep` | `nanovllm_moe_orch_tp_ep.py` | tensor parallel x EP using the EP-LL Triton local graph |

`nanovllm_moe.py` is the older MoE entrypoint. Keep it as a reference. New
recipe-first MoE work should use the `nanovllm_moe_orch_*` files.

## Recipe Contract

Each recipe should expose two orchestration layers.

The outer engine graph:

```text
BlockManager -> Scheduler
Scheduler    -> LLMEngine
ModelRunner  -> LLMEngine
```

The local `ModelRunner` graph:

```text
Attention  -> ModelRunner/layers
MoeBackend -> Dispatch / Experts / Combine
```

For EP recipes, the local graph names the EP-specific cells:

```text
EP-LL: DispatchEPLL + ExpertsEPLL + CombineEPLL
EP-HT: DispatchEPHT + ExpertsEPHT + CombineEPHT
```

A recipe should make implementation choices explicit as module constants, for
example:

```python
BACKEND_IMPL = "ep_ll_triton"
BLOCK_SIZE_M = 64
DISPATCH_KERNEL = "triton"
USE_CUDA_GRAPH = True
```

These are recipe decisions. They should not be passed as end-user config in eval
or e2e scripts. End-user config should describe resources and generation:
model path, `world_size`, `tensor_parallel_size`, memory limits, sequence limits,
layer trimming, and sampling.

## Important Functions

`wire_components(orch, runner, config)` describes the lower-level architecture.
This is where the recipe instantiates attention, model, `MoeBackend`, and MoE
cell types, then registers the exact state/method cells.

`combine(**kwargs)` describes the outer engine architecture. It constructs
`LLMEngine`, `ModelRunner`, `BlockManager`, and `Scheduler`, then registers the
outer cells.

`describe_graph()` prints a human-readable graph summary.

`spawn_eval(...)` is present on multi-rank recipes and is only spawn boilerplate.
It should not hide component wiring.

## Debug With Eval Script

The main MoE eval/debug entrypoint is:

```bash
uv run python -m eval.test_bazaar_moe --help
```

Visualize recipe graphs directly from each recipe's `describe_graph()`:

```bash
uv run python -m eval.visualize_bazaar_moe \
  --recipes all \
  --output eval_results/moe_recipe_graphs.md
```

Generate only one recipe:

```bash
uv run python -m eval.visualize_bazaar_moe \
  --recipes ep_ll_triton \
  --format markdown \
  --output eval_results/ep_ll_triton_graph.md
```

Print Mermaid only, useful when pasting into a renderer:

```bash
uv run python -m eval.visualize_bazaar_moe \
  --recipes tp_ep \
  --format mermaid
```

List all supported MoE structures and their implementation details:

```bash
uv run python -m eval.test_bazaar_moe \
  --list-moe-structures \
  --model-path /home/yyx/models/Qwen3-30B-A3B
```

Print the selected recipe/resource config before running:

```bash
uv run python -m eval.test_bazaar_moe \
  --moe-structure ep_ll_triton \
  --model-path /home/yyx/models/Qwen3-30B-A3B \
  --world-size 2 \
  --tp-size 1 \
  --num-problems 1 \
  --num-layers 1 \
  --max-num-batched-tokens 128 \
  --max-model-len 128 \
  --max-tokens 1
```

Disable the config print when running larger evals:

```bash
uv run python -m eval.test_bazaar_moe \
  --moe-structure single_triton \
  --no-print-moe-config
```

## Smoke Recipes

Single-rank Triton:

```bash
CUDA_VISIBLE_DEVICES=0 MOE_NO_MEM=1 uv run python -m eval.test_bazaar_moe \
  --moe-structure single_triton \
  --model-path /home/yyx/models/Qwen3-30B-A3B \
  --num-problems 1 \
  --num-layers 1 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --max-model-len 128 \
  --max-tokens 1
```

Single-rank torch reference:

```bash
CUDA_VISIBLE_DEVICES=0 MOE_NO_MEM=1 uv run python -m eval.test_bazaar_moe \
  --moe-structure single_torch \
  --model-path /home/yyx/models/Qwen3-30B-A3B \
  --num-problems 1 \
  --num-layers 1 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --max-model-len 128 \
  --max-tokens 1
```

EP-LL Triton:

```bash
CUDA_VISIBLE_DEVICES=0,1 MOE_NO_MEM=1 uv run python -m eval.test_bazaar_moe \
  --moe-structure ep_ll_triton \
  --model-path /home/yyx/models/Qwen3-30B-A3B \
  --world-size 2 \
  --tp-size 1 \
  --num-problems 1 \
  --num-layers 1 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --max-model-len 128 \
  --max-tokens 1
```

EP-HT:

```bash
CUDA_VISIBLE_DEVICES=0,1 MOE_NO_MEM=1 uv run python -m eval.test_bazaar_moe \
  --moe-structure ep_ht \
  --model-path /home/yyx/models/Qwen3-30B-A3B \
  --world-size 2 \
  --tp-size 1 \
  --num-problems 1 \
  --num-layers 1 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --max-model-len 128 \
  --max-tokens 1
```

TP x EP:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MOE_NO_MEM=1 uv run python -m eval.test_bazaar_moe \
  --moe-structure tp_ep \
  --model-path /home/yyx/models/Qwen3-30B-A3B \
  --world-size 4 \
  --tp-size 2 \
  --num-problems 1 \
  --num-layers 1 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --max-model-len 128 \
  --max-tokens 1
```

## EP-LL M_max

For EP-LL recipes, `LL_M_MAX = -1` means auto-size. The current default is:

```text
M_max = max(8, ceil(max_num_batched_tokens * top_k / num_experts) * 4)
```

For Qwen3-30B-A3B with `top_k=8` and `num_experts=128`:

```text
max_num_batched_tokens=8192 -> M_max=2048
max_num_batched_tokens=128  -> M_max=32
```

The eval script prints this estimate for `ep_ll_triton`, `ep_ll_torch`, and
`tp_ep`.

## Debug Checklist

Start with static visibility:

```bash
uv run python -m eval.test_bazaar_moe --list-moe-structures
```

Then use a one-layer, one-problem smoke. Keep `MOE_NO_MEM=1` off when debugging
memory phases; leave it on when you want less log noise.

If a recipe fails at construction, inspect:

1. `describe_graph()` in the recipe.
2. `wire_components(...)` registrations for missing state/method cells.
3. `BACKEND_IMPL`, `USE_CUDA_GRAPH`, and EP cell classes in the recipe.
4. `max_num_batched_tokens`, `world_size`, and `tp_size`; these drive workspace
   sizes and EP group layout.

If EP-LL overflows, lower `max_num_batched_tokens` for smoke tests or create a
new recipe with a larger `LL_M_MAX`. Do not add `LL_M_MAX` as a generic eval
flag unless the goal is specifically to test recipe variants.

If TP x EP hangs after generation, first verify the same model/resources on
`ep_ll_triton` with `tp_size=1`. Then inspect TP group sampling/broadcast and
rank completion behavior, because the local MoE graph may already have produced
rank-0 output before shutdown issues appear.
