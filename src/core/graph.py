"""Dependency-free renderers for compiled Artifact-Infer registries."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Literal

from src.core.registry import BindingSpec, RegistryGraph


GraphFormat = Literal["json", "mermaid", "dot"]
GraphDirection = Literal["TB", "TD", "BT", "LR", "RL"]


def filter_registry_graph(
    graph: RegistryGraph,
    *,
    extensions: Iterable[str] = (),
    tags: Iterable[str] = (),
) -> RegistryGraph:
    """Select components and retain only edges fully inside the selection."""
    selected_extensions = {item for item in extensions if item}
    required_tags = {item for item in tags if item}
    components = tuple(
        component
        for component in graph.components
        if (
            not selected_extensions or component.extension in selected_extensions
        )
        and required_tags.issubset(component.tags)
    )
    component_ids = {component.component_id for component in components}
    bindings = tuple(
        binding
        for binding in graph.bindings
        if binding.provider_id in component_ids and binding.consumer_id in component_ids
    )
    return RegistryGraph(components=components, bindings=bindings)


def _group_bindings(bindings: Iterable[BindingSpec]) -> list[tuple[str, str, str]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for binding in bindings:
        if binding.cell_name is not None:
            grouped[(binding.provider_id, binding.consumer_id)].append(
                binding.cell_name
            )
        else:
            grouped[(binding.provider_id, binding.consumer_id)]
    return [
        (provider_id, consumer_id, ", ".join(sorted(set(labels))))
        for (provider_id, consumer_id), labels in sorted(grouped.items())
    ]


def _mermaid_text(value: str) -> str:
    return value.replace('"', "'").replace("\n", " ")


def render_mermaid(
    graph: RegistryGraph,
    *,
    direction: GraphDirection = "LR",
) -> str:
    if direction not in {"TB", "TD", "BT", "LR", "RL"}:
        raise ValueError(f"Unsupported Mermaid direction {direction!r}")

    node_ids = {
        component.component_id: f"node_{index}"
        for index, component in enumerate(graph.components)
    }
    lines = [f"flowchart {direction}"]
    extensions: dict[str, list] = defaultdict(list)
    for component in graph.components:
        extensions[component.extension].append(component)

    for index, (extension, components) in enumerate(sorted(extensions.items())):
        lines.append(
            f'  subgraph extension_{index}["{_mermaid_text(extension)}"]'
        )
        for component in components:
            label = _mermaid_text(
                f"{component.component_id} [{component.component_type}]"
            )
            lines.append(f'    {node_ids[component.component_id]}["{label}"]')
        lines.append("  end")

    for provider_id, consumer_id, label in _group_bindings(graph.bindings):
        edge = f"  {node_ids[provider_id]} --> {node_ids[consumer_id]}"
        if label:
            safe_label = _mermaid_text(label)
            edge = (
                f"  {node_ids[provider_id]} -->|\"{safe_label}\"| "
                f"{node_ids[consumer_id]}"
            )
        lines.append(edge)
    return "\n".join(lines)


def _dot_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_dot(graph: RegistryGraph, *, rank_direction: str = "LR") -> str:
    if rank_direction not in {"TB", "BT", "LR", "RL"}:
        raise ValueError(f"Unsupported DOT rank direction {rank_direction!r}")

    lines = ["digraph registry {", f"  rankdir={rank_direction};"]
    extensions: dict[str, list] = defaultdict(list)
    for component in graph.components:
        extensions[component.extension].append(component)

    for index, (extension, components) in enumerate(sorted(extensions.items())):
        lines.append(f"  subgraph cluster_{index} {{")
        lines.append(f'    label="{_dot_text(extension)}";')
        for component in components:
            node = _dot_text(component.component_id)
            label = _dot_text(
                f"{component.component_id}\n{component.component_type}"
            )
            lines.append(f'    "{node}" [label="{label}"];')
        lines.append("  }")

    for provider_id, consumer_id, label in _group_bindings(graph.bindings):
        edge = f'  "{_dot_text(provider_id)}" -> "{_dot_text(consumer_id)}"'
        if label:
            edge += f' [label="{_dot_text(label)}"]'
        lines.append(edge + ";")
    lines.append("}")
    return "\n".join(lines)


def render_registry_graph(
    graph: RegistryGraph,
    *,
    format: GraphFormat = "mermaid",
    direction: GraphDirection = "LR",
) -> str:
    if format == "json":
        return graph.to_json()
    if format == "mermaid":
        return render_mermaid(graph, direction=direction)
    if format == "dot":
        dot_direction = "TB" if direction == "TD" else direction
        return render_dot(graph, rank_direction=dot_direction)
    raise ValueError(f"Unsupported graph format {format!r}")


__all__ = [
    "GraphDirection",
    "GraphFormat",
    "filter_registry_graph",
    "render_dot",
    "render_mermaid",
    "render_registry_graph",
]
