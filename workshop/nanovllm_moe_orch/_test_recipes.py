"""Recipe-first smoke runner for `bazaar.nanovllm_moe_orch_*`.

This deliberately accepts only recipe module, resource, and generation options.
Backend/kernel/eager choices live inside the imported recipe.
"""

from __future__ import annotations

import argparse
import importlib
import os

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-hidden-layers-override", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = importlib.import_module(args.recipe)

    if "RANK" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        torch.set_default_device(f"cuda:{local_rank}")

    engine, sampling_params_cls = recipe.combine(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        num_hidden_layers_override=args.num_hidden_layers_override,
    )
    out = engine.generate(
        [args.prompt],
        sampling_params_cls(
            temperature=0.0,
            max_tokens=args.max_tokens,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )
    assert len(out) == 1
    assert len(out[0]["token_ids"]) == args.max_tokens

    rank = int(os.environ.get("RANK", "0"))
    if not dist.is_initialized() or rank == 0:
        print(out, flush=True)


if __name__ == "__main__":
    main()
