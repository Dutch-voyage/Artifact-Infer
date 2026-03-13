# Artifact System - Unified Zero-Cost Abstraction

## Overview

The Artifact System provides a unified interface for systematic implementations that can be registered, called, and composed within the zero-cost RESTful architecture. Artifacts are the fundamental building blocks for horizontal scaling and replace the previous "algorithm" concept with a more generic abstraction.

## Key Features

- **Single Unified Interface**: Every artifact implements one `execute()` method
- **Zero-Cost Execution**: Direct function calls without serialization overhead
- **RESTful API Support**: Full REST endpoints for development and debugging
- **Horizontal Scaling**: Dynamic registration and hot-reloading of new artifacts
- **Type Safety**: JSON schema validation for all inputs
- **Composability**: Chain artifacts together in pipelines

## Architecture

```
src/
├── core/
│   ├── artifact_base.py      # Base artifact interfaces
│   ├── service_registry.py   # Artifact registration and discovery
│   └── unified_service.py    # Service integration
├── artifacts/
│   ├── attention/           # Attention processing artifacts
│   │   ├── sparse_attention.py
│   │   └── sliding_window.py
│   ├── parallel/            # Parallelism artifacts
│   │   ├── tensor_parallel.py
│   │   └── pipeline_parallel.py
│   └── memory/              # Memory management artifacts
│       └── lru_cache.py
└── main.py                  # CLI interface
```

## Artifact Categories

### 1. Attention Artifacts (`AttentionArtifact`)
Process attention matrices for efficiency:
- `sparse_attention`: Configurable sparsity for attention matrices
- `sliding_window`: Local attention patterns with configurable window size

### 2. Parallel Artifacts (`ParallelArtifact`)
Enable model scaling across devices:
- `tensor_parallel`: Split tensors for parallel processing
- `pipeline_parallel`: Sequential processing in pipeline stages

### 3. Memory Artifacts (`MemoryArtifact`)
Manage memory resources efficiently:
- `lru_cache`: Least Recently Used cache
- `fifo_cache`: First-In-First-Out cache (future)

## Usage Examples

### Zero-Cost Execution

```python
from core.service_registry import registry
from core.artifact_base import ExecutionMode

# Execute artifact directly
result = registry.execute_artifact(
    "sparse_attention",
    {"attention_matrix": [[1.0, 0.5], [0.5, 1.0]], "sparsity_ratio": 0.8},
    ExecutionMode.ZERO_COST
)
```

### RESTful Execution

```python
# RESTful execution (development mode)
result = registry.execute_artifact(
    "tensor_parallel",
    {"tensor": [1, 2, 3, 4, 5, 6], "num_splits": 2},
    ExecutionMode.RESTFUL
)
```

### CLI Usage

```bash
# List all artifacts
python src/main.py --list-artifacts

# Execute artifact with JSON input
python src/main.py --execute-artifact sparse_attention --artifact-input '{"attention_matrix": [[1,0.5],[0.5,1]], "sparsity_ratio": 0.5}'

# Start RESTful development server
python src/main.py --server --mode restful
```

### Creating Custom Artifacts

```python
from core.artifact_base import TransformerArtifact, ExecutionContext

class CustomTransformArtifact(TransformerArtifact):
    def transform(self, data, **kwargs):
        # Your transformation logic
        return transformed_data
    
    def get_schema(self):
        return {
            "type": "object",
            "properties": {
                "data": {"type": "any"},
                "param1": {"type": "number"}
            },
            "required": ["data"]
        }

# Register custom artifact
registry.register_artifact(
    name="custom_transform",
    implementation=CustomTransformArtifact(),
    interface="transform",
    metadata={"description": "Custom transformation artifact"}
)
```

## RESTful API Endpoints

When running in RESTful mode (`--mode restful --server`):

- `GET /` - System information
- `POST /artifacts/execute` - Execute artifact
- `GET /artifacts` - List all artifacts
- `GET /artifacts/{name}` - Get artifact details
- `POST /artifacts/{name}/execute` - Execute specific artifact

## Input Validation

All artifacts provide JSON schema validation:

```python
artifact = registry.get_artifact("sparse_attention")
schema = artifact.implementation.get_schema()
# Returns: {"type": "object", "properties": {...}, "required": [...]}
```

## Pipeline Processing

Chain artifacts together:

```python
# Define pipeline
pipeline = [
    {"artifact": "sparse_attention", "params": {"sparsity_ratio": 0.8}},
    {"artifact": "tensor_parallel", "params": {"num_splits": 2}}
]

# Execute pipeline
for step in pipeline:
    artifact_name = step["artifact"]
    params = step["params"]
    result = registry.execute_artifact(artifact_name, {**input_data, **params})
```

## Migration from Algorithms to Artifacts

The system has been updated to use "artifacts" instead of "algorithms":

| Old (Algorithm) | New (Artifact) |
|-----------------|----------------|
| `AlgorithmRegistry` | `ArtifactRegistry` |
| `register_algorithm()` | `register_artifact()` |
| `AlgorithmAPI` | `ArtifactAPI` |
| `algorithm_name` | `artifact_name` |

All existing functionality remains compatible with the new naming.

## Development Workflow

1. **Create artifact**: Implement base class with single `execute()` method
2. **Register artifact**: Use `registry.register_artifact()`
3. **Test locally**: Use CLI or direct execution
4. **Deploy**: Hot-reload via RESTful API

## Testing

Run the artifact system demonstration:

```bash
python examples/artifact_usage.py
```

## Example Commands

```bash
# Basic usage
python src/main.py --list-artifacts
python src/main.py --execute-artifact lru_cache --artifact-input '{"operation":"put","key":"test","value":"data"}'

# Complex example
python src/main.py --execute-artifact sparse_attention --artifact-input '{"attention_matrix":[[1,0.8,0.6],[0.8,1,0.7],[0.6,0.7,1]],"sparsity_ratio":0.6}'

# Development server
python src/main.py --server --mode restful --port 8000
```

The artifact system provides a clean, extensible foundation for horizontal scaling with zero-cost abstraction and RESTful API support.