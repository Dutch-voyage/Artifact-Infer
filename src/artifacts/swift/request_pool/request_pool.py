from src.core.artifact_base import Artifact
from src.services.swift.args import EngineArgs
from typing import List, Tuple, Dict, Any, AsyncGenerator
from enum import Enum
import dataclasses
import asyncio

class RequestType(Enum):
    SINGLE = 0
    BATCH = 1

@dataclasses.dataclass
class RequestPoolArgs:

    @classmethod
    def init_new(cls, engine_args: EngineArgs):
        return cls()

from src.artifacts.swift.structs import SwiftRequest, StepOutput

class RequestPoolArtifact(Artifact):
    def __init__(self, args: RequestPoolArgs):
        super().__init__()
        self.untokenized_raw_requests: list[tuple[SwiftRequest, str]] = []

    @property
    def name(self):
        return "swift_requestpool"
    
    def register(self, service):
        objs_to_register = ["untokenized_raw_requests"]
        for obj in objs_to_register:
            self._register_obj(obj, service)
        
        methods_to_register = ["add_request_and_wait", "add_request_and_stream"]
        for method in methods_to_register:
            self._register_method(method, service)        

    # TODO dispatch requests to different scheduler according to different request properties
    def dispatch_requests(self, request: Tuple[RequestType, Dict[str, Any]]) -> None:
        pass

    # TODO add IPC communication for async tokenization
    async def add_request_and_wait(
        self, raw_request: Dict[str, Any]
    ) -> Tuple[SwiftRequest, List[int]]:
        request = SwiftRequest(raw_request)
        self.untokenized_raw_requests.append((request, raw_request["prompt"]))
        # self.send_to_tokenizer()

        # while True:
        #     await asyncio.sleep(1e-3)
        #     print(f"[POOL] Checking request {id(request)} at {id(request.finished_event)}: {request.finished_event.is_set()}")
        #     if request.finished_event.is_set():
        #         print(f"[POOL] Event detected for request {id(request)}, breaking loop")
        #         break
        
        await request.finished_event.wait()
        return (request, request.output_token_ids)

    async def add_request_and_stream(
        self, raw_request: Dict[str, Any]
    ) -> AsyncGenerator:
        request = SwiftRequest(raw_request)
        self.untokenized_raw_requests.append((request, raw_request["prompt"]))

        while True:
            step_output = await request.output_q.get()
            yield step_output
            request.output_q.task_done()
            if step_output.request.is_finished():
                break