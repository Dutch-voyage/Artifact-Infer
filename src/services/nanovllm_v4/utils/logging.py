from dataclasses import dataclass
import numpy as np
import torch

class LogCollector:
    def __init__(self):
        self.occupied_pages: list[int] = []
        self.time_stamps: list[float] = []
        # self.discrepancsy: list[torch.Tensor] = []
    
    def append(self, time_stamp: float, occupied_pages: int):
        self.time_stamps.append(time_stamp)
        self.occupied_pages.append(occupied_pages)
    
    def reset(self):
        self.occupied_pages = []
    
    def save(self, path):
        np.save(path + "/snapKV_1.npy", {"occupied_pages": np.array(self.occupied_pages), "time_stamps": np.array(self.time_stamps)})

@dataclass
class Log:
    occupied_pages: int = 0
    discrepancy: torch.Tensor = None

_LOG = Log()

def get_log():
    global _LOG
    return _LOG

def set_log(new_log: Log):
    global _LOG
    _LOG = new_log
