from __future__ import annotations

import json

import pytest

from src.core.artifact import Artifact
from src.core.orchestrator import RegistryOrchestrator
from src.core.registry import REGISTRY_GRAPH_SCHEMA, RegistryCompileError
from src.core.service import BaseService


class Provider(Artifact):
    def __init__(self, value: int = 7):
        super().__init__()
        self.value = value

    def ping(self):
        return "pong"


class Consumer(BaseService):
    pass


def test_compile_records_the_actual_registered_graph():
    orchestrator = RegistryOrchestrator()
    provider = orchestrator.add(
        Provider(),
        component_id="attention",
        extension="flashinfer",
        tags=("gpu", "decode"),
    )
    consumer = orchestrator.add(
        Consumer(),
        component_id="runner",
        extension="engine",
    )

    orchestrator.register(provider, "ping", consumer)
    orchestrator.register(provider, "value", consumer)
    graph = orchestrator.compile()

    assert graph.schema == REGISTRY_GRAPH_SCHEMA
    assert graph.component("attention").extension == "flashinfer"
    assert graph.component("attention").tags == ("decode", "gpu")
    assert graph.component("attention").provides == graph.component("runner").requires
    assert [binding.to_dict() for binding in graph.bindings] == [
        {
            "provider_id": "attention",
            "consumer_id": "runner",
            "cell_name": "ping",
            "kind": "method",
        },
        {
            "provider_id": "attention",
            "consumer_id": "runner",
            "cell_name": "value",
            "kind": "state",
        },
    ]

    encoded = json.loads(graph.to_json())
    assert encoded == graph.to_dict()


def test_finalize_compiles_before_binding_methods_and_states():
    orchestrator = RegistryOrchestrator()
    provider = orchestrator.add(Provider())
    consumer = orchestrator.add(Consumer())
    orchestrator.register(provider, "ping", consumer)
    orchestrator.register(provider, "value", consumer)

    orchestrator.finalize()

    assert consumer.ping() == "pong"
    assert consumer.value == 7
    assert orchestrator.compiled_graph.component("Provider").provides[0].name == "ping"


def test_compile_preserves_topology_only_connections():
    orchestrator = RegistryOrchestrator()
    provider = orchestrator.add(Provider(), component_id="worker")
    consumer = orchestrator.add(Consumer(), component_id="engine")

    orchestrator.connect(provider, consumer)

    assert [binding.to_dict() for binding in orchestrator.compile().bindings] == [
        {
            "provider_id": "worker",
            "consumer_id": "engine",
            "cell_name": None,
            "kind": "component",
        }
    ]


def test_explicit_component_ids_must_be_unique():
    orchestrator = RegistryOrchestrator()
    orchestrator.add(Provider(), component_id="shared")

    with pytest.raises(RegistryCompileError, match="already registered"):
        orchestrator.add(Provider(), component_id="shared")


def test_implicit_component_ids_are_disambiguated_for_existing_recipes():
    orchestrator = RegistryOrchestrator()
    first = orchestrator.add(Provider())
    second = orchestrator.add(Provider())

    assert orchestrator.component_id(first) == "Provider"
    assert orchestrator.component_id(second) == "Provider#2"


def test_compile_rejects_cycles_by_object_identity():
    orchestrator = RegistryOrchestrator()
    left = orchestrator.add(Consumer(), component_id="left")
    right = orchestrator.add(Consumer(), component_id="right")
    orchestrator.connect(left, right)
    orchestrator.connect(right, left)

    with pytest.raises(RegistryCompileError, match="left -> right -> left"):
        orchestrator.compile()


def test_mutation_invalidates_a_previously_compiled_graph():
    orchestrator = RegistryOrchestrator()
    orchestrator.add(Provider())
    orchestrator.compile()
    orchestrator.add(Consumer())

    with pytest.raises(RuntimeError, match="has not been compiled"):
        _ = orchestrator.compiled_graph
