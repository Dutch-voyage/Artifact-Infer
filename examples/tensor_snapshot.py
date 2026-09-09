"""Minimum example: locate the first module with a different output."""

import torch
from torch import nn

from src.core.snapshot import TensorSnapshotSession, first_snapshot_difference


model = nn.Sequential(nn.Identity(), nn.Linear(2, 2, bias=False))
with torch.no_grad():
    model[1].weight.copy_(torch.eye(2))


def capture():
    session = TensorSnapshotSession()
    session.watch_named_modules(
        model,
        predicate=lambda name, _module: name in {"0", "1"},
        prefix="model",
    )
    with session:
        model(torch.tensor([1.0, 2.0]))
    return session.finalize()


reference = capture()
with torch.no_grad():
    model[1].weight.add_(1.0)
candidate = capture()

difference = first_snapshot_difference(reference, candidate)
print(f"First difference: {difference.key if difference else None}")
