"""Minimum example: render the compiled registry as Mermaid."""

from examples.registry_compile import compile_example
from src.core.graph import render_mermaid


graph = compile_example()
print(render_mermaid(graph))
