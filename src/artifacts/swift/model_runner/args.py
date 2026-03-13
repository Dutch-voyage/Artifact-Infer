import dataclasses

from src.services.swift.args import EngineArgs


@dataclasses.dataclass
class ModelRunnerArgs:
    model_path: str
    use_dummy: bool
    gpu_mem_utilization: float
    max_tokens_in_batch: int
    max_blocks_per_seq: int
    max_batch_size: int
    block_size: int
    num_cpu_blocks: int
    
    @classmethod
    def init_new(cls, engine_args: EngineArgs):
        return cls(
            engine_args.model_path, 
            engine_args.use_dummy, 
            engine_args.gpu_mem_utilization, 
            engine_args.max_tokens_in_batch, 
            engine_args.max_blocks_per_seq, 
            engine_args.max_batch_size, 
            engine_args.block_size, 
            engine_args.num_cpu_blocks
        )