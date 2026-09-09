from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from src.core.artifact import Artifact
from src.core.orchestrator import RegistryOrchestrator
from src.core.snapshot import (
    TORCH_XOR_HASH_ALGORITHM,
    TENSOR_SNAPSHOT_SCHEMA,
    TensorSnapshot,
    TensorSnapshotSession,
    first_snapshot_difference,
)


def test_manual_cpu_snapshot_contains_metadata_and_hash_only():
    session = TensorSnapshotSession(metadata={"run": "reference"})
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    session.record("layer.0", 0, "output", tensor, {"rank": 0})

    snapshot = session.finalize()
    encoded = snapshot.to_dict()
    record = encoded["records"][0]

    assert encoded["schema"] == TENSOR_SNAPSHOT_SCHEMA
    assert encoded["metadata"] == {"run": "reference"}
    assert record["module"] == "layer.0"
    assert record["fingerprint"]["shape"] == [3, 2]
    assert record["fingerprint"]["stride"] == [1, 3]
    assert record["fingerprint"]["dtype"] == "float32"
    assert record["fingerprint"]["hash"]["value"].startswith("0x")
    assert record["metadata"] == {"rank": 0}
    assert "payload" not in record
    assert "values" not in record


def test_named_module_hooks_capture_inputs_outputs_and_call_order():
    model = nn.Sequential(nn.Linear(3, 3, bias=False), nn.ReLU())
    session = TensorSnapshotSession()
    session.watch_named_modules(
        model,
        predicate=lambda name, _module: name in {"0", "1"},
        prefix="model",
        inputs=True,
    )

    with session:
        model(torch.ones(2, 3))
        model(torch.zeros(2, 3))
    snapshot = session.finalize()

    assert [record.key for record in snapshot.records] == [
        ("model.0", 0, "input.args.0"),
        ("model.0", 0, "output"),
        ("model.1", 0, "input.args.0"),
        ("model.1", 0, "output"),
        ("model.0", 1, "input.args.0"),
        ("model.0", 1, "output"),
        ("model.1", 1, "input.args.0"),
        ("model.1", 1, "output"),
    ]


def test_registry_hook_uses_compiled_component_ids():
    class Activation(Artifact, nn.Module):
        def forward(self, value):
            return torch.relu(value)

    orchestrator = RegistryOrchestrator()
    layer = orchestrator.add(Activation(), component_id="activation")
    orchestrator.compile()
    session = TensorSnapshotSession()
    session.watch_registry(orchestrator)

    with session:
        layer(torch.tensor([-1.0, 2.0]))

    assert session.finalize().records[0].module == "activation"


def test_first_snapshot_difference_reports_the_first_changed_layer():
    def make_snapshot(offset: float) -> TensorSnapshot:
        session = TensorSnapshotSession()
        session.record("layer.0", 0, "output", torch.tensor([1.0]))
        session.record("layer.1", 0, "output", torch.tensor([2.0 + offset]))
        return session.finalize()

    reference = make_snapshot(0.0)
    candidate = make_snapshot(1.0)
    difference = first_snapshot_difference(reference, candidate)

    assert difference is not None
    assert difference.key == ("layer.1", 0, "output")
    assert difference.reason == "fingerprint"


def test_first_snapshot_difference_reports_missing_and_unexpected_records():
    session = TensorSnapshotSession()
    session.record("layer.0", 0, "output", torch.tensor([1.0]))
    reference = session.finalize()

    missing = TensorSnapshot(records=(), metadata={})
    assert first_snapshot_difference(reference, missing).reason == "missing"
    assert first_snapshot_difference(missing, reference).reason == "unexpected"


def test_optional_device_hasher_is_recorded_without_tensor_payload(monkeypatch):
    monkeypatch.setattr(
        torch,
        "hash_tensor",
        lambda _tensor: torch.tensor(17, dtype=torch.int64),
        raising=False,
    )
    session = TensorSnapshotSession()
    session.record("layer", 0, "output", torch.ones(4))

    fingerprint = session.finalize().records[0].fingerprint

    assert fingerprint.algorithm == TORCH_XOR_HASH_ALGORITHM
    assert fingerprint.value == "0x0000000000000011"


def test_watch_rejects_duplicate_names_and_empty_capture():
    session = TensorSnapshotSession()
    session.watch(nn.Identity(), "layer")

    with pytest.raises(ValueError, match="already watched"):
        session.watch(nn.Identity(), "layer")
    with pytest.raises(ValueError, match="At least one"):
        TensorSnapshotSession().watch(
            nn.Identity(),
            "unused",
            inputs=False,
            outputs=False,
        )


def test_snapshot_is_stable_after_source_tensor_changes():
    tensor = torch.tensor([1.0, 2.0])
    session = TensorSnapshotSession()
    session.record("layer", 0, "output", tensor)
    tensor.add_(10)

    captured = session.finalize()
    fresh = TensorSnapshotSession()
    fresh.record("layer", 0, "output", torch.tensor([1.0, 2.0]))

    assert first_snapshot_difference(captured, fresh.finalize()) is None


def test_snapshot_json_round_trip_shape():
    session = TensorSnapshotSession(metadata={"rank": 0})
    session.record("layer", 0, "output", torch.tensor([1]))
    snapshot = session.finalize()

    copied = deepcopy(snapshot.to_dict())
    assert copied["records"][0]["fingerprint"]["numel"] == 1
