from dataclasses import dataclass
import numpy as np
import torch
import os

@dataclass
class LogitsLog:
    steps: list[int] = None
    logits: list[np.ndarray] = None
    token_ids: list[list[int]] = None

class LogCollector:
    def __init__(self):
        self.occupied_pages: list[int] = []
        self.time_stamps: list[float] = []
        self.lse: list[torch.Tensor] = []
        # self.discrepancsy: list[torch.Tensor] = []

    def append_lse(self, lse: torch.Tensor):
        self.lse.append(lse)

    def append(self, time_stamp: float, occupied_pages: int):
        self.time_stamps.append(time_stamp)
        self.occupied_pages.append(occupied_pages)
    
    def reset(self):
        self.occupied_pages = []
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + "/pages.npy", {"occupied_pages": np.array(self.occupied_pages), "time_stamps": np.array(self.time_stamps)})

@dataclass
class Log:
    occupied_pages: int = 0
    discrepancy: torch.Tensor = None
    lse_log: list[torch.Tensor] = None
    logits_log: LogitsLog = None
    num_topp_log: list[torch.Tensor] = None
    selected_topp_indices: list[torch.Tensor] = None
_LOG = Log()

def get_log():
    return _LOG

def append_lse_log(lse: torch.Tensor):
    global _LOG
    if _LOG.lse_log is None:
        _LOG.lse_log = []
    _LOG.lse_log.append(lse)

def append_num_topp(num_topp: torch.Tensor):
    global _LOG
    if _LOG.num_topp_log is None:
        _LOG.num_topp_log = []
    
    _LOG.num_topp_log.append(num_topp)

def append_selected_indices(indices: torch.Tensor):
    global _LOG
    if _LOG.selected_topp_indices is None:
        _LOG.selected_topp_indices = []
    
    _LOG.selected_topp_indices.append(indices)

def reset_log():
    global _LOG
    _LOG = Log()
    
def set_log(log: Log):
    global _LOG
    _LOG = log
