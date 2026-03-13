import asyncio
import functools
import time
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.core.artifact_base import Artifact


class BaseService(Artifact):
    def __init__(self):
        super().__init__()
        # self.registered_methods_by: dict[str, list[str]] = {}  # (artifact_name: [method_names])
        # self.registered_objs_by: dict[str, list[str]] = {}
        self.registered_by: dict[str, Artifact] = {}


class AsyncBaseService(BaseService):
    """Base class using aysnc to coordinate different execution logic"""

    def __init__(self):
        super().__init__()
        self.event_loop = asyncio.get_event_loop()
    
    async def _wrap_as_async(self, func, *args, **kwargs):
        func_partial = functools.partial(func, *args, **kwargs)
        return await self.event_loop.run_in_executor(None, func_partial) 
    

