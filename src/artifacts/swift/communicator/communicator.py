from src.core.artifact_base import Artifact
from src.core.utils import get_zmq_socket
import zmq
import dataclasses

@dataclasses.dataclass
class EngineCommunicatorArgs:
    engine_output_port: str
    tokenizer_input_port: str


class EngineCommunicatorArtifact(Artifact):
    def __init__(self, args: EngineCommunicatorArgs):
        super().__init__()
        context = zmq.Context(2)
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, args.engine_output_port, bind=False
        )
        self.recv_from_tokenizer = get_zmq_socket(
            context, zmq.PULL, args.tokenizer_input_port, bind=False
        )

    def register(self, service):
        obj_to_register = ["send_to_tokenizer", "recv_from_tokenizer"]
        for obj in obj_to_register:
            self._register_obj(obj, service)
