from transformers import AutoTokenizer
from src.core.artifact_base import Artifact
from src.core.service_base import AsyncBaseService
from src.services.swift.args import EngineArgs
import dataclasses

@dataclasses.dataclass
class TokenizerArgs:
    model_path: str
    
    @classmethod
    def init_new(cls, engine_args: EngineArgs):
        return cls(
            model_path=engine_args.model_path
        )

class TokenizerArtifact(Artifact):
    def __init__(self, args: TokenizerArgs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    @property
    def name(self):
        return "swift_tokenizer"
    
    def register(self, service: AsyncBaseService):
        methods_to_register = ["batched_tokenize", "decode"]
        for method in methods_to_register:
            self._register_method(method, service)
    
    def batched_tokenize(self, prompts: list[str]) -> list[list[int]]:
        prompt_token_ids = self.tokenizer(prompts, return_attention_mask=False)["input_ids"]
        return prompt_token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        
