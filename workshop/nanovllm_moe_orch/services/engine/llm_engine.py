import atexit
from dataclasses import fields
import os
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.distributed as dist

from ..config import Config
from ..sampling_params import SamplingParams
from .sequence import Sequence
from src.core.service import BaseService

DUMMY_CREATION = os.getenv("DUMMY_CREATION", False)


def _ensure_distributed(tp_size: int = 1):
    """Initialize torch.distributed and set up the TP × EP subgroups.

    Single-GPU path: initialize a one-rank gloo group so `dist.get_rank()` works,
    and put us on cuda:0 so subsequent `torch.empty(...)` lands on GPU.

    Multi-GPU path: the parent (e.g. an mp.spawn'd worker) is expected to have
    already called `dist.init_process_group(backend='nccl', world_size=N, rank=r,
    device_id=torch.device(f'cuda:{r}'))` and `torch.cuda.set_device(r)`. We then
    construct TP × EP subgroups (see services/utils/parallel.py) where:
      - `tp_size == 1`           → pure EP (current default).
      - `tp_size == world_size`  → pure TP (every rank holds 1/N of every expert).
      - `1 < tp_size < world_size` → TP × EP composition.

    Layout convention: rank = ep_rank * tp_size + tp_rank.
    """
    import torch
    from workshop.nanovllm_moe_orch.services.utils.parallel import init_parallel_groups

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12399")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        torch.cuda.set_device(0)

    world_size = dist.get_world_size()
    init_parallel_groups(tp_size=tp_size, world_size=world_size)

    torch.set_default_device(f"cuda:{torch.cuda.current_device()}")


class LLMEngine(BaseService):

    def __init__(self, model, **kwargs):
        super().__init__()
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)
        if not DUMMY_CREATION:
            self.__post_init__()

    def __post_init__(self):
        _ensure_distributed(tp_size=self.config.tensor_parallel_size)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True)
        self.config.eos = self.tokenizer.eos_token_id
        atexit.register(self.reset)

    def reset(self):
        """Clear scheduler and block-manager state. Idempotent."""
        if not hasattr(self, "scheduler"):
            return
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self.block_mngr.reset()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence.from_prompt(prompt, sampling_params, self.config.kvcache_block_size)
        self.add(seq)

    def step(self):
        seqs, is_prefill = self.schedule()
        t = perf_counter()
        token_ids = self.run(seqs, is_prefill)
        excution_time = perf_counter() - t
        self.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens, excution_time

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            output, num_tokens, excution_time = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / excution_time
                else:
                    decode_throughput = -num_tokens / excution_time
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
            if use_tqdm:
                pbar.update(1) 
        self.reset()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
