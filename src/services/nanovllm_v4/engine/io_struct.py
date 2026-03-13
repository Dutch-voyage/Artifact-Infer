import dataclasses
import torch
from typing import Iterable
from .sequence import Sequence
from ..utils.context import get_context


@dataclasses.dataclass
class SamplingInfo:
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    min_ps: torch.Tensor

    @property
    def is_greedy_sampling(self):
        return torch.any(self.temperatures < 0)
    
    @property
    def need_min_p_sampling(self):
        return torch.any(self.min_ps > 0)

    @property
    def need_top_k_sampling(self):
        return torch.any(self.top_ks > 0)

    @classmethod
    def from_sequence(cls, seqs: list[Sequence]):
        temperatures = torch.tensor(
            [seq.temperature for seq in seqs], dtype=torch.float
        )
        top_ks = torch.tensor([seq.top_k for seq in seqs], dtype=torch.int)
        top_ps = torch.tensor([seq.top_p for seq in seqs], dtype=torch.float)
        min_ps = torch.tensor([seq.min_p for seq in seqs], dtype=torch.float)
        return cls(temperatures, top_ks, top_ps, min_ps)

    def to(self, device):
        self.temperatures = self.temperatures.to(device)
        self.top_ks = self.top_ks.to(device)
        self.top_ps = self.top_ps.to(device)
        self.min_ps = self.min_ps.to(device)
        return self
