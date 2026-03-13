# Artifact Design Specification

## Overview

This specification defines the complete design for artifacts - systematic implementations that can be registered, called, and composed within the unified service architecture. Artifacts are the fundamental building blocks for horizontal scaling in the zero-cost RESTful system.

## Artifact Contract

### Core Requirements

Every artifact must implement a **single unified interface** that supports both zero-cost and RESTful execution:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class ExecutionContext:
    """Context passed to artifacts during execution"""
    artifact_name: str
    interface: str
    mode: str  # "zero_cost" or "restful"
    metadata: Dict[str, Any]

class Artifact(ABC):
    """Base interface for all artifacts"""
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """
        Single unified API for artifact execution
        
        Args:
            input_data: Dictionary containing all inputs
            context: Execution context with mode and metadata
            
        Returns:
            Dictionary with execution results
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema for input validation"""
        pass
    
    def get_description(self) -> str:
        """Human-readable description"""
        return self.__class__.__doc__ or "No description provided"
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input against schema (optional override)"""
        return True  # Basic validation - can be overridden
```

## Artifact Categories

### 1. Attention Artifacts

```python
class AttentionArtifact(Artifact):
    """Base for attention processing artifacts"""
    
    def execute(self, input_data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Process attention matrices with configured parameters"""
        attention_matrix = input_data.get("attention_matrix")
        if attention_matrix is None:
            raise ValueError("attention_matrix required")
            
        processed = self.process_attention(attention_matrix, **input_data)
        return {"processed_attention": processed, "artifact_used": context.artifact_name}
    
    @abstractmethod
    def process_attention(self, attention_matrix: Any, **kwargs) -> Any:
        """Core attention processing logic"""
        pass

# Example implementations
class SparseAttentionArtifact(AttentionArtifact):
    def process_attention(self, attention_matrix, sparsity_ratio: float = 0.8):
        # Implementation here
        return processed_matrix

class SlidingWindowAttentionArtifact(AttentionArtifact):
    def process_attention(self, attention_matrix, window_size: int = 128):
        # Implementation here
        return processed_matrix
```

### 2. Parallel Artifacts

```python
class ParallelArtifact(Artifact):
    """Base for parallelism artifacts"""
    
    def execute(self, input_data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Apply parallel processing with configured parameters"""
        tensor = input_data.get("tensor")
        if tensor is None:
            raise ValueError("tensor required")
            
        result = self.process_parallel(tensor, **input_data)
        return {"result": result, "artifact_used": context.artifact_name}
    
    @abstractmethod
    def process_parallel(self, tensor: Any, **kwargs) -> Any:
        """Core parallel processing logic"""
        pass

# Example implementations
class TensorParallelArtifact(ParallelArtifact):
    def process_parallel(self, tensor, num_splits: int = 2):
        # Implementation here
        return split_tensors

class PipelineParallelArtifact(ParallelArtifact):
    def process_parallel(self, tensor, num_stages: int = 4):
        # Implementation here
        return staged_result
```

### 3. Memory Artifacts

```python
class MemoryArtifact(Artifact):
    """Base for memory management artifacts"""
    
    def execute(self, input_data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Manage memory resources with configured parameters"""
        operation = input_data.get("operation", "get")
        
        if operation == "get":
            result = self.retrieve(**input_data)
        elif operation == "put":
            result = self.store(**input_data)
        elif operation == "clear":
            result = self.clear(**input_data)
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        return {"result": result, "artifact_used": context.artifact_name}
    
    @abstractmethod
    def retrieve(self, key: str, **kwargs) -> Any:
        """Retrieve from memory"""
        pass
    
    @abstractmethod
    def store(self, key: str, value: Any, **kwargs) -> bool:
        """Store to memory"""
        pass
    
    @abstractmethod
    def clear(self, **kwargs) -> bool:
        """Clear memory"""
        pass
```

## Zero-Cost Communication

Artifacts communicate through **service-mediated zero-cost interfaces**:

```python
class ArtifactCommunicationService:
    """Handles zero-cost communication between artifacts"""
    
    def __init__(self, registry):
        self.registry = registry
        self.artifact_cache = {}
    
    def call_artifact(self, artifact_name: str, input_data: Dict[str, Any], 
                     context: ExecutionContext) -> Dict[str, Any]:
        """Call another artifact with zero-cost"""
        artifact_spec = self.registry.get_artifact(artifact_name)
        if not artifact_spec:
            raise ValueError(f"Artifact {artifact_name} not found")
        
        # Direct function call - no serialization
        artifact_instance = artifact_spec.implementation
        return artifact_instance.execute(input_data, context)
    
    def chain_artifacts(self, pipeline: List[Dict[str, Any]], 
                       initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a chain of artifacts"""
        current_result = initial_input
        
        for step in pipeline:
            artifact_name = step["artifact"]
            params = step.get("params", {})
            
            # Merge params with current result
            input_data = {**current_result, **params}
            
            context = ExecutionContext(
                artifact_name=artifact_name,
                interface="chain",
                mode="zero_cost",
                metadata={"step": len(pipeline)}
            )
            
            current_result = self.call_artifact(artifact_name, input_data, context)
        
        return current_result
```

## RESTful API Design

### Artifact Endpoints

```python
# RESTful API for artifact management
class ArtifactRESTAPI:
    """RESTful API wrapper for artifacts"""
    
    async def execute_artifact(self, artifact_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute artifact via REST"""
        try:
            artifact_spec = self.registry.get_artifact(artifact_name)
            if not artifact_spec:
                return {"error": "Artifact not found"}
            
            context = ExecutionContext(
                artifact_name=artifact_name,
                interface=artifact_spec.interface,
                mode="restful",
                metadata={"request_id": str(uuid.uuid4())}
            )
            
            result = artifact_spec.implementation.execute(request_data, context)
            return {"status": "success", "data": result}
            
        except Exception as e:
            return {"error": str(e)}
    
    async def get_artifact_schema(self, artifact_name: str) -> Dict[str, Any]:
        """Get artifact schema for validation"""
        artifact_spec = self.registry.get_artifact(artifact_name)
        if not artifact_spec:
            return {"error": "Artifact not found"}
        
        return {
            "artifact": artifact_name,
            "schema": artifact_spec.implementation.get_schema()
        }
```

## Service Integration

### Service-Level Artifact Calling

Each service provides built-in artifact invocation:

```python
class SimplifiedModelService:
    """Model service with artifact integration"""
    
    def __init__(self, artifact_registry):
        self.registry = artifact_registry
        self.active_artifacts = {}
    
    def use_artifact(self, artifact_name: str, **params):
        """Configure model to use specific artifact"""
        artifact_spec = self.registry.get_artifact(artifact_name)
        if not artifact_spec:
            raise ValueError(f"Artifact {artifact_name} not found")
        
        self.active_artifacts[artifact_spec.interface] = {
            "artifact": artifact_spec.implementation,
            "params": params
        }
    
    def forward(self, input_ids: List[int]) -> ModelOutput:
        """Forward pass with active artifacts"""
        # Get attention artifact if configured
        if "attention" in self.active_artifacts:
            attention_config = self.active_artifacts["attention"]
            # Use artifact for attention processing
            
        # Standard forward pass
        return self.model(input_ids)
    
    async def post_forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """RESTful version of forward"""
        input_ids = data.get("input_ids", [])
        result = self.forward(input_ids)
        return {"output": result}
```

## Example Use Cases

### 1. Basic Artifact Registration

```python
# Register a new attention artifact
registry.register_artifact(
    name="custom_attention",
    implementation=CustomAttentionArtifact(),
    interface="attention",
    metadata={
        "description": "Custom attention mechanism",
        "parameters": ["sparsity_ratio", "temperature"]
    }
)
```

### 2. Zero-Cost Execution

```python
# Direct zero-cost execution
attention_artifact = registry.get_artifact("sparse_attention")
result = attention_artifact.implementation.execute({
    "attention_matrix": attention_tensor,
    "sparsity_ratio": 0.8
}, ExecutionContext("sparse_attention", "attention", "zero_cost", {}))
```

### 3. RESTful Execution

```python
# RESTful execution
await artifact_api.execute_artifact("sparse_attention", {
    "attention_matrix": [[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]],
    "sparsity_ratio": 0.8
})
```

### 4. Artifact Composition

```python
# Chain multiple artifacts
pipeline = [
    {"artifact": "sparse_attention", "params": {"sparsity_ratio": 0.8}},
    {"artifact": "tensor_parallel", "params": {"num_splits": 2}}
]

result = communication_service.chain_artifacts(pipeline, {
    "attention_matrix": initial_matrix
})
```

## File Structure

```
src/
├── core/
│   ├── artifact_base.py          # Base artifact interfaces
│   ├── artifact_registry.py      # Registration and discovery
│   └── artifact_communication.py # Zero-cost communication
├── artifacts/
│   ├── attention/
│   │   ├── sparse_attention.py
│   │   └── sliding_window.py
│   ├── parallel/
│   │   ├── tensor_parallel.py
│   │   └── pipeline_parallel.py
│   └── memory/
│       └── lru_cache.py
├── services/
│   ├── model_service.py
│   ├── tokenizer_service.py
│   └── memory_service.py
└── api/
    └── artifact_api.py          # RESTful endpoints
```

## Validation and Testing

### Artifact Validation Framework

```python
class ArtifactValidator:
    """Validates artifact implementations"""
    
    def validate_artifact(self, artifact: Artifact) -> tuple[bool, str]:
        """Validate artifact compliance"""
        
        # Check required methods
        if not hasattr(artifact, 'execute'):
            return False, "Missing execute method"
            
        if not hasattr(artifact, 'get_schema'):
            return False, "Missing get_schema method"
        
        # Test basic execution
        try:
            schema = artifact.get_schema()
            test_input = self.generate_test_input(schema)
            context = ExecutionContext("test", "test", "zero_cost", {})
            result = artifact.execute(test_input, context)
            
            if not isinstance(result, dict):
                return False, "execute must return dict"
                
        except Exception as e:
            return False, f"Execution failed: {str(e)}"
        
        return True, "Valid artifact"
```

This design provides a complete, abstract framework for artifacts that supports both zero-cost execution and RESTful APIs while maintaining the single-API requirement and enabling horizontal scaling.