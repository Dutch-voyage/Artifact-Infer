import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoProcessor
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.processors.qwen_vl import Qwen2_5VLMultiModalProcessor
from nanovllm.engine.mm_io_struct import MultimodalInputs


class LLMEngine:

    def __init__(self, model, **kwargs):
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(config.model)
        self.mm_data_processor = Qwen2_5VLMultiModalProcessor(config.hf_config, config, self.processor, transport_mode="default")
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    async def add_request(
        self,
        prompt: str | list[int] | dict,
        sampling_params: SamplingParams,
        image_data=None,
        video_data=None,
    ) -> None:
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        elif isinstance(prompt, list):
            token_ids = prompt 
            
        if image_data is not None:
            if not isinstance(image_data, list):
                image_data = [image_data]
        if video_data is not None:
            if not isinstance(video_data, list):
                video_data = [video_data]

        mm_inputs = None
        if image_data is not None or video_data is not None:
            mm_inputs = await self.mm_data_processor.process_mm_data_async(
                image_data=image_data,
                video_data=video_data,
                input_text=prompt,
            )
            token_ids = mm_inputs["input_ids"]
        
            mm_inputs = MultimodalInputs.from_dict(mm_inputs) if mm_inputs is not None else None

        origin_input_ids = self.model_runner.model.pad_input_ids(token_ids, mm_inputs)
        seq = Sequence(origin_input_ids, sampling_params, mm_inputs=mm_inputs)
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

    async def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        image_data=None,
        video_data=None,
        benchmark: bool = False,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            await self.add_request(prompt, sp, image_data=image_data, video_data=video_data)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        step_start_time = perf_counter()
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            step_time = perf_counter() - t
            
            # Record timing for benchmark mode
            if benchmark:
                current_time = perf_counter()
                for seq in self.scheduler.waiting + self.scheduler.running:
                    # Record first token time (after prefill)
                    if seq.first_token_time is None and seq.num_completion_tokens == 1:
                        seq.first_token_time = current_time
            
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / step_time
                else:
                    decode_throughput = -num_tokens / step_time
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # Calculate benchmark metrics
        benchmark_metrics = None
        if benchmark:
            end_time = perf_counter()
            total_time = end_time - step_start_time
            
            # Collect timing data from sequences
            ttfts = []
            tpots = []
            for seq in self.scheduler.finished:
                if seq.first_token_time is not None:
                    ttft = seq.first_token_time - step_start_time
                    ttfts.append(ttft)
                    
                    # Calculate TPOT (time per output token after first token)
                    num_output_tokens = seq.num_completion_tokens
                    if num_output_tokens > 1:
                        total_decode_time = end_time - seq.first_token_time
                        tpot = total_decode_time / (num_output_tokens - 1)
                        tpots.append(tpot)
            
            if ttfts:
                avg_ttft = sum(ttfts) / len(ttfts)
                avg_tpot = sum(tpots) / len(tpots) if tpots else 0
                benchmark_metrics = {
                    "avg_ttft": avg_ttft,
                    "avg_tpot": avg_tpot,
                    "total_time": total_time,
                    "num_sequences": len(ttfts),
                }
        
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        
        if benchmark:
            return outputs, benchmark_metrics
        return outputs
