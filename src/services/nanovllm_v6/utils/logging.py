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

    def append(self, time_stamp: float, occupied_pages: int):
        self.time_stamps.append(time_stamp)
        self.occupied_pages.append(occupied_pages)
    
    def reset(self):
        self.occupied_pages = []
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + "/pages.npy", {"occupied_pages": np.array(self.occupied_pages), "time_stamps": np.array(self.time_stamps)})


class Log:
    occupied_pages: int = 0
    discrepancy: torch.Tensor = None
    lse_log: list[torch.Tensor] = None
    logits_log: LogitsLog = None
    num_topp_log: list[torch.Tensor] = None
    selected_topp_indices: list[torch.Tensor] = None
    temperatures: list[torch.Tensor] = None
_LOG = Log()

def get_log():
    return _LOG

def append_item_to_log(name, item):
    global _LOG
    if getattr(_LOG, name, None) is None:
        setattr(_LOG, name, [])
    getattr(_LOG, name).append(item)

def append_lse_log(lse: torch.Tensor):
    global _LOG
    if getattr(_LOG, "lse_log", None) is None:
        _LOG.lse_log = []
    _LOG.lse_log.append(lse)

def append_num_topp(num_topp: torch.Tensor):
    global _LOG
    if getattr(_LOG, "num_topp_log", None) is None:
        _LOG.num_topp_log = []
    
    _LOG.num_topp_log.append(num_topp)

def append_selected_indices(indices: torch.Tensor):
    global _LOG
    if getattr(_LOG, "selected_topp_indices", None) is None:
        _LOG.selected_topp_indices = []
    
    _LOG.selected_topp_indices.append(indices)

def reset_log():
    global _LOG
    _LOG = Log()
    
def set_log(log: Log):
    global _LOG
    _LOG = log
