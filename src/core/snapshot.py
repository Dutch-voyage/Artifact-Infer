"""Small tensor fingerprints for module-level accuracy debugging."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import nn
import xxhash


TENSOR_SNAPSHOT_SCHEMA = "tensor-snapshot.v1"
TENSOR_FINGERPRINT_SCHEMA = "tensor-fingerprint.v1"
ARTIFACT_XOR_HASH_ALGORITHM = "artifact-infer.indexed-xor64.v1"
TORCH_XOR_HASH_ALGORITHM = "torch.hash_tensor.xor64.v0"
XXH3_HASH_ALGORITHM = "xxh3-64.logical-bytes.v1"


@dataclass(frozen=True)
class TensorFingerprint:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    algorithm: str
    value: str
    schema: str = TENSOR_FINGERPRINT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "device": self.device,
            "numel": self.numel,
            "hash": {
                "algorithm": self.algorithm,
                "value": self.value,
            },
        }


@dataclass(frozen=True)
class SnapshotRecord:
    module: str
    call_index: int
    tensor: str
    fingerprint: TensorFingerprint
    metadata: Mapping[str, Any]

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.module, self.call_index, self.tensor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "call_index": self.call_index,
            "tensor": self.tensor,
            "fingerprint": self.fingerprint.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TensorSnapshot:
    records: tuple[SnapshotRecord, ...]
    metadata: Mapping[str, Any]
    schema: str = TENSOR_SNAPSHOT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "metadata": dict(self.metadata),
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class SnapshotDifference:
    key: tuple[str, int, str]
    reason: str
    reference: SnapshotRecord | None
    candidate: SnapshotRecord | None


@dataclass
class _PendingFingerprint:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    algorithm: str
    device_hash: torch.Tensor | None = None
    host_tensor: torch.Tensor | None = None
    source_tensor: torch.Tensor | None = None
    cuda_device: torch.device | None = None


@dataclass
class _PendingRecord:
    module: str
    call_index: int
    tensor: str
    fingerprint: _PendingFingerprint
    metadata: dict[str, Any]


def _artifact_xor_hash(tensor: torch.Tensor) -> torch.Tensor | None:
    """Call the optional in-repository XOR operator when it is loaded."""
    try:
        operator = torch.ops.artifact_infer.xor_hash
    except AttributeError:
        return None
    try:
        result = operator(tensor)
    except (NotImplementedError, RuntimeError, TypeError):
        return None
    return result if isinstance(result, torch.Tensor) else None


def _torch_xor_hash(tensor: torch.Tensor) -> torch.Tensor | None:
    """Call the development ``torch.hash_tensor`` extension when available."""
    operator = getattr(torch, "hash_tensor", None)
    if not callable(operator):
        return None
    try:
        result = operator(tensor)
    except (NotImplementedError, RuntimeError, TypeError):
        return None
    return result if isinstance(result, torch.Tensor) else None


def _device_hash(tensor: torch.Tensor) -> tuple[torch.Tensor, str] | None:
    for operator, algorithm in (
        (_artifact_xor_hash, ARTIFACT_XOR_HASH_ALGORITHM),
        (_torch_xor_hash, TORCH_XOR_HASH_ALGORITHM),
    ):
        result = operator(tensor)
        if result is not None:
            if result.numel() != 1:
                raise ValueError("A tensor hash operator must return one scalar")
            return result.reshape(()), algorithm
    return None


def _xxh3_tensor(tensor: torch.Tensor) -> str:
    logical = tensor.contiguous()
    if logical.ndim == 0:
        logical = logical.reshape(1)
    raw = logical.view(torch.uint8).reshape(-1).numpy()
    return xxhash.xxh3_64_hexdigest(raw)


def _named_tensors(value: Any, prefix: str) -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _named_tensors(value[key], f"{prefix}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _named_tensors(item, f"{prefix}.{index}")


class TensorSnapshotSession:
    """Capture fingerprints from selected ``nn.Module`` calls.

    Hooks are opt-in and removed by ``close()`` or when leaving the context.
    The default captures module outputs only.
    """

    def __init__(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.metadata = dict(metadata or {})
        self._pending: list[_PendingRecord] = []
        self._handles: list[Any] = []
        self._watched_names: set[str] = set()
        self._call_counts: dict[str, int] = {}
        self._finalized: TensorSnapshot | None = None

    def __enter__(self) -> TensorSnapshotSession:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def watch(
        self,
        module: nn.Module,
        name: str,
        *,
        inputs: bool = False,
        outputs: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._finalized is not None:
            raise RuntimeError("Cannot watch modules after finalizing a snapshot")
        if not inputs and not outputs:
            raise ValueError("At least one of inputs or outputs must be captured")
        name = name.strip()
        if not name:
            raise ValueError("Watched module name must not be empty")
        if name in self._watched_names:
            raise ValueError(f"Module name {name!r} is already watched")
        self._watched_names.add(name)
        record_metadata = dict(metadata or {})

        def capture_hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            result: Any,
        ) -> None:
            call_index = self._call_counts.get(name, 0)
            self._call_counts[name] = call_index + 1
            if inputs:
                for tensor_name, tensor in _named_tensors(args, "input.args"):
                    self.record(name, call_index, tensor_name, tensor, record_metadata)
                for tensor_name, tensor in _named_tensors(kwargs, "input.kwargs"):
                    self.record(name, call_index, tensor_name, tensor, record_metadata)
            if outputs:
                for tensor_name, tensor in _named_tensors(result, "output"):
                    self.record(name, call_index, tensor_name, tensor, record_metadata)

        handle = module.register_forward_hook(capture_hook, with_kwargs=True)
        self._handles.append(handle)

    def watch_named_modules(
        self,
        root: nn.Module,
        *,
        predicate: Callable[[str, nn.Module], bool],
        prefix: str = "",
        inputs: bool = False,
        outputs: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Watch named children selected by a layer/module predicate."""
        for name, module in root.named_modules():
            if name and predicate(name, module):
                qualified_name = f"{prefix}.{name}" if prefix else name
                self.watch(
                    module,
                    qualified_name,
                    inputs=inputs,
                    outputs=outputs,
                    metadata=metadata,
                )

    def watch_registry(
        self,
        orchestrator: Any,
        *,
        predicate: Callable[[str, nn.Module], bool] | None = None,
        inputs: bool = False,
        outputs: bool = True,
    ) -> None:
        """Watch registered components that are also ``nn.Module`` objects."""
        for component in orchestrator.registry:
            if not isinstance(component, nn.Module):
                continue
            name = orchestrator.component_id(component)
            if predicate is None or predicate(name, component):
                self.watch(component, name, inputs=inputs, outputs=outputs)

    def record(
        self,
        module: str,
        call_index: int,
        tensor_name: str,
        tensor: torch.Tensor,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one tensor manually at the current execution boundary."""
        if self._finalized is not None:
            raise RuntimeError("Cannot record tensors after finalizing a snapshot")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")

        detached = tensor.detach()
        pending = _PendingFingerprint(
            shape=tuple(detached.shape),
            stride=tuple(detached.stride()),
            dtype=str(detached.dtype).removeprefix("torch."),
            device=str(detached.device),
            numel=detached.numel(),
            algorithm=XXH3_HASH_ALGORITHM,
        )
        device_hash = _device_hash(detached)
        if device_hash is not None:
            pending.device_hash, pending.algorithm = device_hash
        elif detached.device.type == "cpu":
            pending.host_tensor = detached.contiguous().clone()
        elif detached.device.type == "cuda":
            logical = detached.contiguous()
            host = torch.empty(
                logical.shape,
                dtype=logical.dtype,
                device="cpu",
                pin_memory=True,
            )
            host.copy_(logical, non_blocking=True)
            pending.host_tensor = host
            pending.source_tensor = logical
            pending.cuda_device = detached.device
        else:
            pending.host_tensor = detached.to(device="cpu").contiguous()

        self._pending.append(
            _PendingRecord(
                module=module,
                call_index=call_index,
                tensor=tensor_name,
                fingerprint=pending,
                metadata=dict(metadata or {}),
            )
        )

    def finalize(self) -> TensorSnapshot:
        if self._finalized is not None:
            return self._finalized
        self.close()

        cuda_devices = {
            item.fingerprint.cuda_device
            for item in self._pending
            if item.fingerprint.cuda_device is not None
        }
        for device in cuda_devices:
            torch.cuda.synchronize(device)

        device_items: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
        for index, item in enumerate(self._pending):
            device_hash = item.fingerprint.device_hash
            if device_hash is not None:
                device_items.setdefault(device_hash.device, []).append(
                    (index, device_hash)
                )

        device_values: dict[int, int] = {}
        for items in device_items.values():
            stacked = torch.stack([value for _, value in items])
            for (index, _), value in zip(items, stacked.cpu().tolist()):
                device_values[index] = int(value) & ((1 << 64) - 1)

        records = []
        for index, item in enumerate(self._pending):
            pending = item.fingerprint
            if pending.device_hash is not None:
                value = f"0x{device_values[index]:016x}"
            else:
                assert pending.host_tensor is not None
                value = f"0x{_xxh3_tensor(pending.host_tensor)}"
            records.append(
                SnapshotRecord(
                    module=item.module,
                    call_index=item.call_index,
                    tensor=item.tensor,
                    fingerprint=TensorFingerprint(
                        shape=pending.shape,
                        stride=pending.stride,
                        dtype=pending.dtype,
                        device=pending.device,
                        numel=pending.numel,
                        algorithm=pending.algorithm,
                        value=value,
                    ),
                    metadata=item.metadata,
                )
            )

        self._pending.clear()
        self._finalized = TensorSnapshot(
            records=tuple(records),
            metadata=self.metadata,
        )
        return self._finalized

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def first_snapshot_difference(
    reference: TensorSnapshot,
    candidate: TensorSnapshot,
) -> SnapshotDifference | None:
    """Return the first missing, unexpected, or changed tensor record."""
    candidate_by_key = {record.key: record for record in candidate.records}
    reference_keys = {record.key for record in reference.records}
    for reference_record in reference.records:
        candidate_record = candidate_by_key.get(reference_record.key)
        if candidate_record is None:
            return SnapshotDifference(
                reference_record.key,
                "missing",
                reference_record,
                None,
            )
        if reference_record.fingerprint != candidate_record.fingerprint:
            return SnapshotDifference(
                reference_record.key,
                "fingerprint",
                reference_record,
                candidate_record,
            )
        if dict(reference_record.metadata) != dict(candidate_record.metadata):
            return SnapshotDifference(
                reference_record.key,
                "metadata",
                reference_record,
                candidate_record,
            )
    for candidate_record in candidate.records:
        if candidate_record.key not in reference_keys:
            return SnapshotDifference(
                candidate_record.key,
                "unexpected",
                None,
                candidate_record,
            )
    return None


__all__ = [
    "ARTIFACT_XOR_HASH_ALGORITHM",
    "SnapshotDifference",
    "SnapshotRecord",
    "TENSOR_FINGERPRINT_SCHEMA",
    "TENSOR_SNAPSHOT_SCHEMA",
    "TORCH_XOR_HASH_ALGORITHM",
    "TensorFingerprint",
    "TensorSnapshot",
    "TensorSnapshotSession",
    "XXH3_HASH_ALGORITHM",
    "first_snapshot_difference",
]
