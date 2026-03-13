"""
Base interfaces for artifacts

Provides the fundamental building blocks for systematic implementations
that can be registered, called, and composed within the unified service architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
from pathlib import Path
import importlib.util
import inspect
import uuid

from collections import  deque

@dataclass
class ExecutionContext:
    """Context passed to artifacts during execution"""

    artifact_name: str
    metadata: Dict[str, Any]
    request_id: Optional[str] = None


class Artifact(ABC):
    """Base interface for all artifacts - single unified API"""

    # """TODO let's see if there are better abstraction for the 'runnning' of artifacts"""
    # @abstractmethod
    # def execute(self, input_data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
    #     """
    #     Single unified API for artifact execution

    #     Args:
    #         input_data: Dictionary containing all inputs
    #         context: Execution context with mode and metadata

    #     Returns:
    #         Dictionary with execution results
    #     """

    def __init__(self):
        super().__init__()
        self.registered_methods_to = {}
        self.registered_objs_to = {}

    @property
    def path(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def _register_method(self, obj_name: str, service: "Artifact"):
        if not hasattr(self, obj_name):
            raise ValueError("No such method in the artifact to register.")

        if service.name not in self.registered_methods_to.keys():
            self.registered_methods_to[service.name] = []
        self.registered_methods_to[service.name].append(obj_name)

        if self.name not in service.registered_by.keys():
            service.registered_by[self.name] = self

        original_method = getattr(self.__class__, obj_name)

        self_artifact = self
        self_service = service

        # Create a wrapper that replaces self references with artifact references
        def create_method_wrapper(original_func):

            def method_wrapper(dummy_self, *args, **kwargs):
                # Create a proxy object that routes attribute access
                class ArtifactProxy:
                    def __init__(self):
                        pass
                    
                    def __getattr__(self, name):
                        # First check if it's a registered method in the service
                        # assert not (
                        #     hasattr(self_service, name) and hasattr(self_artifact, name)
                        # ), f"Attribute name conflict for '{name}' between service and artifact. Please make sure that either {self_service.name} or {self_artifact.name} defines attribute '{name}'."

                        if hasattr(self_service, name):
                            return getattr(self_service, name)
                        cu_artifact = self_artifact
                        artifact_queue = deque([cu_artifact])
                        
                        searched_names = set()
                        
                        while (
                            len(artifact_queue) > 0
                        ):
                            cu_artifact = artifact_queue.popleft()
                            if cu_artifact.name in searched_names:
                                raise AttributeError(
                                    f"Found loop in registeration between {searched_names} when searching for attribute '{name}'"
                                )
                            else:
                                searched_names.add(cu_artifact.name)
                            if (hasattr(cu_artifact, name)):
                                return getattr(cu_artifact, name)
                            if hasattr(cu_artifact, "registered_by"):
                                artifact_queue.extend([art for art_name, art in cu_artifact.registered_by.items()])

                        raise AttributeError(
                            f"'{type(self_service).__name__}' object has no attribute '{name}' or the attribute is not registered from artifacts'"
                        )

                    def __setattr__(self, name, value):
                        # TODO: check if the following logic is right
                        # if service has the attribute already, set it with the new value
                        # else if the registered tree of the service has the attribute already, set it with the new value for the corresponding artifact
                        # else create the attribute for the service
                        
                        if hasattr(self_service, name):
                            setattr(self_service, name, value)
                        artifact_queue = deque([self_artifact])
                        while (
                            len(artifact_queue) > 0
                        ):
                            cu_artifact = artifact_queue.popleft()
                            if (hasattr(cu_artifact, name)):
                                return setattr(cu_artifact, name, value)
                            if hasattr(cu_artifact, "registered_by"):
                                artifact_queue.extend([art for art_name, art in cu_artifact.registered_by.items()])
                        
                        setattr(self_artifact, name, value)
                        # setattr(self_service, name, value)

                # Create the proxy and call the original method
                proxy = ArtifactProxy()

                # Call the original method with the proxy as self
                return original_func(proxy, *args, **kwargs)

            return method_wrapper

        # Create the wrapped method
        wrapped_method = create_method_wrapper(original_method)

        # Bind it to the service
        import types

        setattr(service, obj_name, types.MethodType(wrapped_method, service))

    def _register_obj(self, obj_name: str, service):
        if not hasattr(self, obj_name):
            raise ValueError("No such object in the artifact to register.")
        if service.name not in self.registered_objs_to.keys():
            self.registered_objs_to[service.name] = []
        self.registered_objs_to[service.name].append(obj_name)

        if self.name not in service.registered_by.keys():
            service.registered_by[self.name] = self

        setattr(service, obj_name, getattr(self, obj_name))

        # self_artifact = self
        # self_service = service

        # # # original_obj = getattr(self, obj_name)

        # # # Create an ObjectProxy that provides cross-artifact access similar to ArtifactProxy
        # class ObjectProxy:
        #     def __init__(self, artifact_name):
        #         # self._original_obj = original_obj
        #         self._artifact_name = artifact_name

        #     def __getattr__(self, name):
        #         # Then check if it's a registered method in the service
        #         if name in self_artifact.registered_objs:
        #             if hasattr(self_service, name):
        #                 return getattr(self_service, name)

        #             raise AttributeError(f"'{type(self_service).__name__}' object has no attribute '{name}'")
        #         else:
        #             return getattr(self_artifact, name)

        #     def __setattr__(self, name, value):
        #         if name in self_artifact.registered_objs:
        #             # Set attribute on the original object
        #             if hasattr(self_service, name):
        #                 setattr(self_service, name, value)

        #             raise AttributeError(f"'{type(self_service).__name__}' object has no attribute '{name}'")
        #         else:
        #             setattr(self_artifact, name, value)

        # # # Create the proxy object
        # proxy_obj = ObjectProxy(artifact_name)
        # setattr(self, obj_name, proxy_obj)

    # @abstractmethod
    # def get_schema(self) -> Dict[str, Any]:
    #     """Return JSON schema for input validation"""
    #     pass

    # @abstractmethod
    # def get_description(self) -> str:
    #     """Human-readable description"""
    #     return self.__class__.__doc__ or "No description provided"

    # @abstractmethod
    # def validate_input(self, input_data: Dict[str, Any]) -> bool:
    #     """Validate input against schema (optional override)"""
    #     return True  # Basic validation - can be overridden


class ArtifactRegistry:
    pass
