from __future__ import annotations

import json

import pytest

from eval.visualize_registry import main
from src.core.graph import (
    filter_registry_graph,
    render_dot,
    render_mermaid,
    render_registry_graph,
)
from src.core.registry import (
    BindingSpec,
    CellSpec,
    ComponentSpec,
    RegistryCompileError,
    RegistryGraph,
)


def example_graph() -> RegistryGraph:
    return RegistryGraph(
        components=(
            ComponentSpec(
                component_id="moe.dispatch",
                component_type="Dispatch",
                module="extensions.moe",
                extension="moe",
                tags=("gpu",),
                provides=(CellSpec("route", "method"),),
            ),
            ComponentSpec(
                component_id="runner",
                component_type="ModelRunner",
                module="engine.runner",
                extension="engine",
                tags=("gpu",),
                requires=(CellSpec("route", "method"),),
            ),
            ComponentSpec(
                component_id="scheduler",
                component_type="Scheduler",
                module="engine.scheduler",
                extension="engine",
            ),
        ),
        bindings=(
            BindingSpec("moe.dispatch", "runner", "route", "method"),
            BindingSpec("moe.dispatch", "runner", "workspace", "state"),
            BindingSpec("scheduler", "runner", None, "component"),
        ),
    )


def test_mermaid_groups_extensions_and_combines_edge_labels():
    rendered = render_mermaid(example_graph(), direction="LR")

    assert rendered.startswith("flowchart LR")
    assert 'subgraph extension_0["engine"]' in rendered
    assert 'subgraph extension_1["moe"]' in rendered
    assert 'moe.dispatch [Dispatch]' in rendered
    assert '-->|"route, workspace"|' in rendered
    assert rendered.count("moe.dispatch [Dispatch]") == 1


def test_dot_is_dependency_free_and_preserves_topology_edges():
    rendered = render_dot(example_graph(), rank_direction="TB")

    assert rendered.startswith("digraph registry {")
    assert "rankdir=TB;" in rendered
    assert 'label="moe";' in rendered
    assert '"scheduler" -> "runner";' in rendered
    assert 'label="route, workspace"' in rendered


def test_filter_keeps_only_internal_edges_for_selected_extensions_and_tags():
    engine = filter_registry_graph(example_graph(), extensions=("engine",))
    gpu = filter_registry_graph(example_graph(), tags=("gpu",))

    assert [item.component_id for item in engine.components] == [
        "runner",
        "scheduler",
    ]
    assert [item.to_dict() for item in engine.bindings] == [
        BindingSpec("scheduler", "runner", None, "component").to_dict()
    ]
    assert [item.component_id for item in gpu.components] == [
        "moe.dispatch",
        "runner",
    ]


def test_registry_graph_round_trips_and_rejects_unknown_endpoints():
    graph = example_graph()
    assert RegistryGraph.from_dict(graph.to_dict()) == graph

    broken = graph.to_dict()
    broken["bindings"][0]["provider_id"] = "missing"
    with pytest.raises(RegistryCompileError, match="unknown components"):
        RegistryGraph.from_dict(broken)


def test_generic_renderer_supports_all_formats():
    graph = example_graph()

    assert json.loads(render_registry_graph(graph, format="json")) == graph.to_dict()
    assert render_registry_graph(graph, format="mermaid").startswith("flowchart")
    assert render_registry_graph(graph, format="dot").startswith("digraph")


def test_cli_reads_compiled_json_and_writes_selected_format(tmp_path):
    source = tmp_path / "graph.json"
    output = tmp_path / "graph.md"
    source.write_text(example_graph().to_json(), encoding="utf-8")

    main(
        [
            str(source),
            "--format",
            "mermaid",
            "--extension",
            "moe",
            "--output",
            str(output),
        ]
    )

    rendered = output.read_text(encoding="utf-8")
    assert "moe.dispatch [Dispatch]" in rendered
    assert "runner [ModelRunner]" not in rendered
