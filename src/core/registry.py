"""Serializable description of an Artifact-Infer component registry.

The runtime objects remain ordinary Python objects. These records are the
small, deterministic result of compiling their registered connections.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal


REGISTRY_GRAPH_SCHEMA = "artifact-registry-graph.v1"
CellKind = Literal["method", "state"]
BindingKind = Literal["method", "state", "component"]


@dataclass(frozen=True, order=True)
class CellSpec:
    name: str
    kind: CellKind

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind}


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    component_type: str
    module: str
    extension: str = "core"
    tags: tuple[str, ...] = ()
    provides: tuple[CellSpec, ...] = ()
    requires: tuple[CellSpec, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "module": self.module,
            "extension": self.extension,
            "tags": list(self.tags),
            "provides": [cell.to_dict() for cell in self.provides],
            "requires": [cell.to_dict() for cell in self.requires],
        }


@dataclass(frozen=True)
class BindingSpec:
    provider_id: str
    consumer_id: str
    cell_name: str | None
    kind: BindingKind

    def to_dict(self) -> dict[str, str | None]:
        return {
            "provider_id": self.provider_id,
            "consumer_id": self.consumer_id,
            "cell_name": self.cell_name,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class RegistryGraph:
    components: tuple[ComponentSpec, ...]
    bindings: tuple[BindingSpec, ...]
    schema: str = REGISTRY_GRAPH_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "components": [component.to_dict() for component in self.components],
            "bindings": [binding.to_dict() for binding in self.bindings],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def component(self, component_id: str) -> ComponentSpec:
        for component in self.components:
            if component.component_id == component_id:
                return component
        raise KeyError(component_id)


class RegistryCompileError(ValueError):
    """Raised when a registered component graph cannot be compiled."""


__all__ = [
    "BindingKind",
    "BindingSpec",
    "CellKind",
    "CellSpec",
    "ComponentSpec",
    "REGISTRY_GRAPH_SCHEMA",
    "RegistryCompileError",
    "RegistryGraph",
]
