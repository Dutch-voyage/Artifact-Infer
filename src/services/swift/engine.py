import asyncio
import argparse
import sys
from src.core.service_base import AsyncBaseService
from src.core.utils import GB
import dataclasses
from collections import deque
from enum import Enum
from src.artifacts.swift.structs import StepOutput
from src.artifacts.swift.request_pool import RequestPoolArtifact, RequestPoolArgs
from src.artifacts.swift.scheduler import SchedulerArtifact, SchedulerArgs
from src.artifacts.swift.tokenizer import TokenizerArtifact, TokenizerArgs
from src.artifacts.swift.model_runner import ModelRunnerArtifact, ModelRunnerArgs
from src.artifacts.swift.block_manager import BlockManagerArtifact, BlockManagerArgs
from src.services.swift.args import EngineArgs

import pdb

@dataclasses.dataclass
class DefaultEngineArtifacts:
    request_pool: RequestPoolArtifact
    tokenizer: TokenizerArtifact
    model_runner: ModelRunnerArtifact
    
    scheduler: SchedulerArtifact
    
    # cpu_block_manager: BlockManagerArtifact
    # gpu_block_manager: BlockManagerArtifact
    
    @classmethod
    def init_new(cls, engine_args: EngineArgs):
        request_pool = RequestPoolArtifact(RequestPoolArgs.init_new(engine_args))
        
        print("[Engine] Initializing model...")
        model_runner = ModelRunnerArtifact(ModelRunnerArgs.init_new(engine_args))
        
        print("[Engine] Loading weights...")
        if engine_args.use_dummy:
            model_runner.dummy_load_weights()
        else:
            model_runner.load_weights()
        
        num_gpu_blocks = model_runner.profile_num_blocks()
        num_cpu_blocks = engine_args.num_cpu_blocks

        block_size_bytes = engine_args.block_size * model_runner.model_config.get_kvslot_size()
        print(f"[Engine] Number of GPU blocks: {num_gpu_blocks} ({num_gpu_blocks*block_size_bytes/GB:.2f} GB)")
        print(f"[Engine] Number of CPU blocks: {num_cpu_blocks} ({num_cpu_blocks*block_size_bytes/GB:.2f} GB)")
        
        print("[Engine] Allocating kv cache and swap...")
        if engine_args.use_dummy:
            model_runner.dummy_init_kvcache_and_swap(num_gpu_blocks)
        else:
            model_runner.init_kvcache_and_swap(num_gpu_blocks)

        model_runner.gpu_block_manager = BlockManagerArtifact(
            BlockManagerArgs.init_new(
                "GPU",
                num_gpu_blocks,
                engine_args, 
            )
        )
        model_runner.cpu_block_manager = BlockManagerArtifact(
            BlockManagerArgs.init_new(
                "CPU",
                num_cpu_blocks,
                engine_args,      
            )            
        )

        print("[Engine] Initializing scheduler...")
        scheduler = SchedulerArtifact(SchedulerArgs.init_new(num_gpu_blocks, engine_args))
        
        print("[Engine] Initializing tokenization engine...")
        tokenizer = TokenizerArtifact(TokenizerArgs.init_new(engine_args))
        
        print("[Engine] Model initialized")
        
        return cls(
            request_pool=request_pool, 
            tokenizer=tokenizer, 
            model_runner=model_runner, 
            scheduler=scheduler, 
            # cpu_block_manager=cpu_block_manager, 
            # gpu_block_manager=gpu_block_manager
        )

    def register(self, service: AsyncBaseService):
        self.request_pool.register(service)
        self.scheduler.register(service)
        self.tokenizer.register(service)
        self.model_runner.register(service)
        # self.cpu_block_manager.register(service)
        # self.gpu_block_manager.register(service)
        

class EngineService(AsyncBaseService):
    def __init__(self, engine_args: EngineArgs):
        super().__init__()
        self.args = engine_args
        
        from src.artifacts.swift.structs import SwiftRequest
        
        # self.waiting_q = deque()
        # self.running_q: list[SwiftRequest] = []
        # self.swapped_q = deque()
        
        # self.untokenized_raw_requests: list[tuple[SwiftRequest, str]] = []
        
        self.artifacts = DefaultEngineArtifacts.init_new(engine_args)
        self.artifacts.register(self)

    async def start_all_event_loops(self):
        await asyncio.gather(
            self._tokenize_raw_request_event_loop(),
            self._main_event_loop(),
        )
    
    async def _tokenize_raw_request_event_loop(self):
        while True: 
            if not self.untokenized_raw_requests:
                await asyncio.sleep(1e-3)
                continue
        
            cur_untokenized_raw_requests = self.untokenized_raw_requests
            self.untokenized_raw_requests = []
            
            prompts = [prompt for _, prompt in cur_untokenized_raw_requests]
            prompt_token_ids = await self._wrap_as_async(
                self.batched_tokenize, 
                prompts
            )

            new_requests = []
            for (request, _), prompt_token_id in zip(cur_untokenized_raw_requests, prompt_token_ids):
                request.prompt_token_ids = prompt_token_id
                request.prompt_len = len(prompt_token_id)
                new_requests.append(request)
            
            self.on_request_arrival(new_requests)
            await asyncio.sleep(1e-3)

    async def _main_event_loop(self):
        while True:
            cur_batch, cur_swap_in, cur_swap_out = self.get_next_batch()
            if len(cur_batch) == 0 and len(cur_swap_in) == 0 and len(cur_swap_out) == 0:
                await asyncio.sleep(5e-3)
                continue
            # print(f"{len(cur_batch)}, {len(cur_swap_in)}, {len(cur_swap_out)}")
            if len(cur_swap_out) > 0: 
                await self._wrap_as_async(
                    self.swap_out_seqs, 
                    [req.request_id for req in cur_swap_out]
                )
            
            if len(cur_swap_in) > 0: 
                await self._wrap_as_async(
                    self.swap_in_seqs,
                    [req.request_id for req in cur_swap_in]
                )

            # output_tokens = await self._wrap_as_async(
            #     self.forward, 
            #     cur_batch
            # )
            
            # print(f"[Engine] length of current running queue {len(self.running_q)}, {id(self.running_q)}")
            
            output_tokens = self.forward(cur_batch)

            # breakpoint() 
            finished_req_ids = []
            for req, output_token in zip(cur_batch, output_tokens):
                # print(f"[ENGINE] Processing request {id(req)} at {id(req.finished_event)}: current output {len(req.output_token_ids)}/{req.output_len}")
                req.output_token_ids.append(output_token)
                req.output_q.put_nowait(StepOutput(output_token, req))
                if req.is_finished():
                    finished_req_ids.append(req.request_id)
                    # print(f"[ENGINE] Setting event for request {id(req)} at {id(req.finished_event)}")
                    req.finished_event.set()
            self.free_seqs_resources(finished_req_ids)
            # await self._wrap_as_async(
            #     self.free_seqs_resources, 
            #     finished_req_ids, 
            # )
            
            self.on_batch_finish(cur_batch)

    