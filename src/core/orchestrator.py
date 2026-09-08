from typing import Any, Dict, Iterable, List

from .artifact import (Artifact, 
                       Cell, 
                       DistMethodCell, 
                       DistMethodCell, 
                       MethodCell, 
                       IPCWorkerWrapper)
from .service import BaseService
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

import pickle
import torch
import torch.distributed as dist

import atexit

from src.core import artifact
from src.core.registry import (
    BindingSpec,
    CellSpec,
    ComponentSpec,
    RegistryCompileError,
    RegistryGraph,
)

class RegistryOrchestrator:
    def __init__(self):
        self.registry: List[Artifact] = []
        self._component_metadata: dict[int, dict[str, Any]] = {}
        self._component_ids: dict[str, int] = {}
        self._bindings: list[BindingSpec] = []
        self._topology_edges: list[tuple[int, int]] = []
        self._compiled_graph: RegistryGraph | None = None

    def add(
        self,
        artifact: Artifact,
        *,
        component_id: str | None = None,
        extension: str | None = None,
        tags: Iterable[str] = (),
    ):
        """Add an artifact and attach optional graph-inspection metadata.

        Explicit component ids must be unique. Repeated implicit names receive
        a deterministic ``#N`` suffix so existing recipes remain valid.
        """
        key = id(artifact)
        metadata = self._component_metadata.get(key)
        if metadata is not None:
            if component_id is not None and component_id != metadata["component_id"]:
                raise RegistryCompileError(
                    f"Component {metadata['component_id']!r} is already registered; "
                    f"cannot rename it to {component_id!r}"
                )
            return artifact

        artifact_registry_id = getattr(artifact, "registry_id", None)
        explicit_id = component_id is not None or artifact_registry_id is not None
        requested_id = component_id or artifact_registry_id or artifact.name
        requested_id = str(requested_id).strip()
        if not requested_id:
            raise RegistryCompileError("Component id must not be empty")
        resolved_id = self._claim_component_id(requested_id, explicit=explicit_id)

        resolved_extension = str(
            extension or getattr(artifact, "registry_extension", "core")
        ).strip()
        if not resolved_extension:
            raise RegistryCompileError("Component extension must not be empty")

        normalized_tags = tuple(
            sorted({str(tag).strip() for tag in tags if str(tag).strip()})
        )
        self._component_metadata[key] = {
            "component_id": resolved_id,
            "extension": resolved_extension,
            "tags": normalized_tags,
        }
        self._component_ids[resolved_id] = key
        if not self._contains_identity(self.registry, artifact):
            self.registry.append(artifact)
        self._invalidate_compiled_graph()
        return artifact
    
    def register(self, child: Artifact, attr_name: str, parent: Artifact):
        """Unified entry point for both states and methods."""
        # Ensure it exists in the artifact's cells
        if attr_name not in child._cells:
            # Auto-wrap if it's a method not yet defined as a cell
            attr = getattr(child, attr_name)
            if callable(attr):
                child.define_method(attr_name)
            else:
                child.define_state(attr_name, attr)
        
        self.add(child)
        self.add(parent)

        cell = child._cells[attr_name]
        kind = "method" if isinstance(cell, (MethodCell, DistMethodCell)) else "state"
        binding = BindingSpec(
            provider_id=self.component_id(child),
            consumer_id=self.component_id(parent),
            cell_name=attr_name,
            kind=kind,
        )
        if binding not in self._bindings:
            self._bindings.append(binding)

        if not self._contains_identity(child.parents, parent):
            child.parents.append(parent)
        self._record_topology_edge(child, parent)
        self._invalidate_compiled_graph()

    def connect(self, child: Artifact, parent: Artifact):
        """Phase 1: Build the Topology."""
        self.add(child)
        self.add(parent)
        if not self._contains_identity(child.parents, parent):
            child.parents.append(parent)
        self._record_topology_edge(child, parent)
        self._invalidate_compiled_graph()

    def component_id(self, artifact: Artifact) -> str:
        try:
            return self._component_metadata[id(artifact)]["component_id"]
        except KeyError as exc:
            raise RegistryCompileError(
                f"Component {artifact!r} is not registered"
            ) from exc

    @property
    def compiled_graph(self) -> RegistryGraph:
        if self._compiled_graph is None:
            raise RuntimeError("Registry graph has not been compiled")
        return self._compiled_graph

    def compile(self) -> RegistryGraph:
        """Validate and freeze the actual registered component graph."""
        self._check_cycles()

        bindings = list(self._bindings)
        bound_pairs = {
            (binding.provider_id, binding.consumer_id) for binding in bindings
        }
        for provider_key, consumer_key in self._topology_edges:
            provider_id = self._component_metadata[provider_key]["component_id"]
            consumer_id = self._component_metadata[consumer_key]["component_id"]
            if (provider_id, consumer_id) not in bound_pairs:
                bindings.append(
                    BindingSpec(
                        provider_id=provider_id,
                        consumer_id=consumer_id,
                        cell_name=None,
                        kind="component",
                    )
                )

        bindings.sort(
            key=lambda item: (
                item.provider_id,
                item.consumer_id,
                item.cell_name or "",
                item.kind,
            )
        )
        provided: dict[str, set[CellSpec]] = {}
        required: dict[str, set[CellSpec]] = {}
        for binding in bindings:
            if binding.kind == "component" or binding.cell_name is None:
                continue
            cell = CellSpec(binding.cell_name, binding.kind)
            provided.setdefault(binding.provider_id, set()).add(cell)
            required.setdefault(binding.consumer_id, set()).add(cell)

        components = []
        for component in self.registry:
            metadata = self._component_metadata[id(component)]
            component_id = metadata["component_id"]
            component_type = type(component)
            components.append(
                ComponentSpec(
                    component_id=component_id,
                    component_type=component_type.__name__,
                    module=component_type.__module__,
                    extension=metadata["extension"],
                    tags=metadata["tags"],
                    provides=tuple(sorted(provided.get(component_id, ()))),
                    requires=tuple(sorted(required.get(component_id, ()))),
                )
            )
        components.sort(key=lambda item: item.component_id)

        self._compiled_graph = RegistryGraph(
            components=tuple(components),
            bindings=tuple(bindings),
        )
        return self._compiled_graph

    def finalize(self):
        """Phase 2: Validate and Bind."""
        # 1. Validate and freeze the graph before propagation changes parents.
        self.compile()
        
        # 2. Propagation
        # We propagate every cell defined in every artifact to all its ancestors
        for artifact in self.registry:
            # We only propagate cells that were "Defined" on this specific artifact
            # (i.e., not the ones it inherited)
            for attr_name, cell in artifact._cells.items():
                # We skip inherited cells during this primary loop to avoid redundant paths
                if isinstance(cell, MethodCell) and cell.origin != artifact:
                    continue
                origin_alias = f"{artifact.name}_{attr_name}"
                for parent in artifact.parents:
                    self._propagate(parent, attr_name, origin_alias, cell)

    def _propagate(self, service: Artifact, local_alias: str, origin_alias: str, cell: Cell):
        # Bind reference
        origin_name = cell.origin.name if hasattr(cell, "origin") else "<state>"
        print(f"Propagating {local_alias} from {origin_name} to {service.name} as {origin_alias} with cell: {id(cell)}")
        service._cells[local_alias] = cell
        service._state_map[local_alias] = origin_alias
        
        # Climb the DAG
        for parent in service.parents:
            self._propagate(parent, local_alias, origin_alias, cell)

    def _check_cycles(self):
        visited: set[int] = set()
        stack: list[Artifact] = []
        active: set[int] = set()

        def dfs(node):
            key = id(node)
            visited.add(key)
            active.add(key)
            stack.append(node)
            for p in node.parents:
                parent_key = id(p)
                if parent_key not in visited:
                    dfs(p)
                elif parent_key in active:
                    cycle_start = next(
                        index for index, item in enumerate(stack) if item is p
                    )
                    cycle = stack[cycle_start:] + [p]
                    labels = " -> ".join(self.component_id(item) for item in cycle)
                    raise RegistryCompileError(f"Cycle detected: {labels}")
            stack.pop()
            active.remove(key)

        for art in self.registry:
            if id(art) not in visited:
                dfs(art)

    def _claim_component_id(self, requested_id: str, *, explicit: bool) -> str:
        if requested_id not in self._component_ids:
            return requested_id
        if explicit:
            raise RegistryCompileError(
                f"Component id {requested_id!r} is already registered"
            )
        suffix = 2
        while f"{requested_id}#{suffix}" in self._component_ids:
            suffix += 1
        return f"{requested_id}#{suffix}"

    def _record_topology_edge(self, child: Artifact, parent: Artifact) -> None:
        edge = (id(child), id(parent))
        if edge not in self._topology_edges:
            self._topology_edges.append(edge)

    def _invalidate_compiled_graph(self) -> None:
        self._compiled_graph = None

    @staticmethod
    def _contains_identity(items: Iterable[Artifact], target: Artifact) -> bool:
        return any(item is target for item in items)
        
def _boostrap(subproc_cls, rank, world_size, rank_event, **kwargs):
    print(f"Child process for Rank {rank} initializing.")
    shm = SharedMemory(name="nanovllm")
    wrapped_instance = IPCWorkerWrapper.create_from_cls(subproc_cls, rank, world_size, rank_event, shm, **kwargs)
    wrapped_instance.loop()

class DistOrchestrator(RegistryOrchestrator):
    def __init__(self, world_size: int, shm_size: int = 2**20):
        super().__init__()
        self.world_size = world_size
        self.ctx = mp.get_context("spawn")
        self.processes = []
        
        self.rank0_wrappers = []
        # Setup IPC primitives globally for this TP group
        if self.world_size > 1:
            self.shm = SharedMemory(name="nanovllm", create=True, size=shm_size)
            self.events = [self.ctx.Event() for _ in range(1, world_size)]
        else:
            self.shm, self.events = None, []
    
    def exit(self):
        self.wrapped_rank0_child.exit()
        for p in self.processes:
            p.join()
    
    def deploy_distributed_runner(self, parent: Artifact, child_cls: type, post_deploy_func=None, **kwargs) -> Artifact:
        print("[Reminder] You should always register the states/methods as MethodCell before deploy distributed processes")
        """
        Deploys the distributed group, links Rank 0 to the parent, and returns Rank 0.
        """
        
        if self.world_size > 1:
            for i in range(1, self.world_size):
                rank_event = self.events[i - 1]
                
                p = self.ctx.Process(
                    target=_boostrap, 
                    args=(child_cls, i, self.world_size, rank_event),
                    kwargs=kwargs 
                )
                p.start()
                self.processes.append(p)
        
        self.wrapped_rank0_child = IPCWorkerWrapper.create_master(child_cls, self.world_size, self.events, self.shm, post_deploy_func, **kwargs)

        rank0_child = self.wrapped_rank0_child.proc
        
        self.rank0_wrappers.append(self.wrapped_rank0_child)
        
        self.connect(child=rank0_child, parent=parent)
        
        # Wrap methods in the Distributed Broadcast Cell
        for name, cell in list(rank0_child._cells.items()):
            if isinstance(cell, MethodCell):
                rank0_child._cells[name] = DistMethodCell(
                    func=cell.func, 
                    origin=cell.origin, 
                    rank=0, 
                    world_size=self.world_size,
                    ipc_wrapper=self.wrapped_rank0_child # The cell uses the wrapper for write_shm
                )
        
        atexit.register(self.exit)
        # Return Rank 0 for manual pre-finalization hooks
        return rank0_child
    
    def finalize(self):
        """Phase 2: Build DAG and Sync Workers."""
        # 1. Standard DAG Propagation in the main process
        super().finalize()
        
        # 2. Late-Sync! 
        # Rank 0 now has all the propagated states from LLMEngine. 
        # Tell the wrappers to push this data to the waiting workers.
        for wrapper in self.rank0_wrappers:
            wrapper.sync_registry()
        
