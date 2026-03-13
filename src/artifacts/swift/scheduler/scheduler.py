from src.core.artifact_base import Artifact
from src.services.swift.args import EngineArgs
from src.artifacts.swift.structs import RequestIdManager, SwiftRequest
from src.core.service_base import AsyncBaseService
from src.core.utils import cdiv
from collections import deque
import dataclasses

@dataclasses.dataclass
class SchedulerArgs:
    num_gpu_blocks: int
    num_cpu_blocks: int
    block_size: int

    max_batch_size: int
    max_tokens_in_batch: int
    max_seqs_in_block_table: int

    @classmethod
    def init_new(cls, num_gpu_blocks:int, args: EngineArgs):
        return cls(
            num_gpu_blocks=num_gpu_blocks, 
            num_cpu_blocks=args.num_cpu_blocks, 
            block_size=args.block_size,
            max_batch_size=args.max_batch_size, 
            max_tokens_in_batch=args.max_tokens_in_batch,
            max_seqs_in_block_table=args.max_seqs_in_block_table
        )

class SchedulerArtifact(Artifact):
    def __init__(self, args: SchedulerArgs):
        super().__init__()
        self.args = args
        self.waiting_q = deque()
        self.running_q: list[SwiftRequest] = []
        self.swapped_q = deque()
        
        self.num_gpu_blocks = args.num_gpu_blocks
        self.num_decoding_gpu_blocks = 0
        self.num_free_cpu_blocks = args.num_cpu_blocks

        self.request_id_manager = RequestIdManager(args.max_seqs_in_block_table)

    @property
    def name(self):
        return "swift_scheduler"

    def register(self, service: AsyncBaseService):
        objs_to_register = ["waiting_q", "running_q", "swapped_q"]
        for obj in objs_to_register:
             self._register_obj(obj, service)
        
        methods_to_register = ["on_request_arrival", "get_next_batch", "on_batch_finish"]
        for method in methods_to_register:
            self._register_method(method, service)
        
        
    def _get_block_needed(self, request: SwiftRequest) -> int:
        return cdiv(
            request.prompt_len + request.get_cur_output_len(), self.args.block_size
        )

    def on_request_arrival(self, requests: list[SwiftRequest]):
        self.waiting_q.extend(requests)

    def get_next_batch(
        self,
    ) -> tuple[list[SwiftRequest], list[SwiftRequest], list[SwiftRequest]]:
        if not self.swapped_q:
            cur_batch = []
            cur_batch_block_needed = 0
            cur_num_tokens_sum = 0
            while self.waiting_q:
                cur_seq: SwiftRequest = self.waiting_q[0]
                cur_seq_block_needed = self._get_block_needed(cur_seq)
                if (
                    len(self.running_q) + len(cur_batch) + 1 <= self.args.max_batch_size
                    and cur_batch_block_needed
                    + cur_seq_block_needed
                    + self.num_decoding_gpu_blocks
                    <= self.num_gpu_blocks
                    and cur_num_tokens_sum + cur_seq.prompt_len
                    <= self.args.max_tokens_in_batch
                ):
                    cur_batch.append(cur_seq)
                    cur_batch_block_needed += cur_seq_block_needed
                    cur_num_tokens_sum += cur_seq.prompt_len
                    self.waiting_q.popleft()
                else:
                    break
            if cur_batch:
                # TODO why get request id here
                for req in cur_batch:
                    req.request_id = self.request_id_manager.get_id()
                self.running_q.extend(cur_batch)
                self.num_decoding_gpu_blocks += cur_batch_block_needed
                return cur_batch, [], []
        self.num_decoding_gpu_blocks = sum(
            self._get_block_needed(req) for req in self.running_q
        )

        newly_swapped_out = []
        while (
            len(self.running_q) > self.args.max_batch_size
            or self.num_decoding_gpu_blocks > self.num_gpu_blocks
        ):
            victim = self.running_q.pop()
            self.num_deocding_gpu_blocks -= self._get_block_needed(victim)
            newly_swapped_out.append(victim)

        newly_swapped_in = []
        if newly_swapped_in:
            self.swapped_q.extendleft(newly_swapped_out)
        else:
            while self.swapped_q:
                cur_seq = self.swapped_q[0]
                num_cur_seq_blocks = self._get_block_needed(cur_seq)
                if (
                    len(self.running_q) + 1 <= self.args.max_batch_size
                    and self.num_decoding_gpu_blocks + num_cur_seq_blocks
                    <= self.num_gpu_blocks
                ):
                    self.running_q.append(cur_seq)
                    self.num_decoding_gpu_blcoks += num_cur_seq_blocks
                    self.swapped_q.popleft()
                    newly_swapped_in.append(cur_seq)
                else:
                    break
        
        return self.running_q, newly_swapped_in, list(reversed(newly_swapped_out))

    def on_batch_finish(self, batch: list[SwiftRequest]):
        self.request_id_manager.free_ids(
            [req.request_id for req in batch if req.is_finished()]
        )
        self.running_q = [req for req in self.running_q if not req.is_finished()]
        # print(self)
        # print(f"[Scheduler] length of current running queue {len(self.running_q)}, {id(self.running_q)}")