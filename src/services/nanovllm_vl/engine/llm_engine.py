import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoProcessor
import torch.multiprocessing as mp

from ..config import Config
from ..sampling_params import SamplingParams
from .sequence import Sequence
from .scheduler import Scheduler
from src.services.nanovllm_vl.model_runner import ModelRunner
from src.services.nanovllm_vl.model_runner.processor.qwen_vl import QwenVLImageProcessor

from src.services.nanovllm_vl.engine.mm_io_struct import ImageInputs
from src.core.service_base import BaseService

class LLMEngine(BaseService):
    @property
    def name(self):
        return "nanovllm_vl_engine"

    def __init__(self, model, **kwargs):
        super().__init__()
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.model_runner.model._register_method("pad_input_ids", self)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        _processor = AutoProcessor.from_pretrained(config.model, use_fast=True)
        self.mm_data_processor = QwenVLImageProcessor(config.hf_config, config, _processor, transport_mode="default")
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams, image_data=None):
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        
        # handle multimodal data (image data only for now)
        if image_data is not None:
            if not isinstance(image_data, list):
                image_data = [image_data]
            image_inputs = self.mm_data_processor.process(
                image_data=image_data,
                input_text=prompt,
            )
        else:
            image_inputs = None
        if image_inputs and "input_ids" in image_inputs:
            token_ids = image_inputs["input_ids"]
        image_inputs = ImageInputs.from_dict(image_inputs) if image_inputs is not None else None

        origin_input_ids = self.pad_input_ids(token_ids, image_inputs)
        seq = Sequence(origin_input_ids, sampling_params, image_inputs=image_inputs)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        image_data: list[str] | list[list[str]] | None = None,
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp, image in zip(prompts, sampling_params, image_data):
            self.add_request(prompt, sp, image)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
            pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
