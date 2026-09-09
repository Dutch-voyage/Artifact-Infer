"""Minimum example: compile the registered execution path."""

from src.core.artifact import Artifact
from src.core.orchestrator import RegistryOrchestrator
from src.core.service import BaseService


class Attention(Artifact):
    def forward(self, tokens):
        return tokens


class ModelRunner(BaseService):
    pass


def compile_example():
    registry = RegistryOrchestrator()
    attention = registry.add(
        Attention(),
        component_id="attention",
        extension="torch",
        tags=("decode",),
    )
    runner = registry.add(
        ModelRunner(),
        component_id="model_runner",
        extension="engine",
    )
    registry.register(attention, "forward", runner)
    return registry.compile()


if __name__ == "__main__":
    print(compile_example().to_json())
