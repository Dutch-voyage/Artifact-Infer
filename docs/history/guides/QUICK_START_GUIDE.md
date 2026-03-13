# Artifact System Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Basic Usage

```bash
# List all available artifacts
python src/main.py --list-artifacts

# Execute a simple artifact
python src/main.py --execute-artifact sparse_attention --artifact-input '{"attention_matrix": [[1,0.5],[0.5,1]], "sparsity_ratio": 0.5}'

# Start development server
python src/main.py --server --mode restful --port 8000
```

### 2. Python Quick Start

```python
import asyncio
from core.service_registry import registry
from core.artifact_base import ExecutionMode

# Execute artifact directly
result = registry.execute_artifact(
    "sparse_attention",
    {"attention_matrix": [[1.0, 0.8], [0.8, 1.0]], "sparsity_ratio": 0.3},
    ExecutionMode.ZERO_COST
)
print(f"Result: {result}")
```

### 3. Pipeline Example

```python
from examples.advanced_artifacts import ArtifactPipeline

# Create a pipeline
pipeline = ArtifactPipeline()
pipeline.add_step("memory_optimizer", {"operation": "analyze"})
pipeline.add_step("sparse_attention", {"sparsity_ratio": 0.5})

# Execute pipeline
result = asyncio.run(pipeline.execute({"model_size": 500_000_000}))
```

### 4. Custom Artifact

```python
from core.artifact_base import TransformerArtifact

class MyCustomArtifact(TransformerArtifact):
    def transform(self, data, **kwargs):
        return {"processed": data["input"] * 2}
    
    def get_schema(self):
        return {"type": "object", "properties": {"input": {"type": "number"}}, "required": ["input"]}

# Register and use
registry.register_artifact("my_custom", MyCustomArtifact(), "transform")
result = registry.execute_artifact("my_custom", {"input": 42}, ExecutionMode.ZERO_COST)
```

## 📚 Essential Resources

### Files to Check First
- `examples/artifact_usage.py` - Basic examples
- `examples/advanced_artifacts.py` - Advanced patterns
- `examples/artifact_cookbook.py` - Ready-to-use recipes
- `ARTIFACT_IO_DOCUMENTATION.md` - Complete IO documentation

### Key Commands
```bash
# Development
python src/main.py --list-artifacts
python src/main.py --execute-artifact lru_cache --artifact-input '{"operation":"stats"}'
python src/main.py --server --mode restful

# Examples
python examples/artifact_usage.py
python examples/advanced_artifacts.py
python examples/artifact_cookbook.py
```

### REST API Endpoints (when server is running)
- `GET /` - System info
- `GET /artifacts` - List artifacts
- `POST /artifacts/execute` - Execute artifact
- `POST /artifacts/{name}/execute` - Execute specific artifact

## 🎯 Common Use Cases

1. **Memory Optimization**: Use `memory_optimizer` + `sparse_attention`
2. **Multi-GPU**: Use `tensor_parallel` + `pipeline_parallel`
3. **Caching**: Use `lru_cache` for model weights
4. **Batch Processing**: Use patterns from `artifact_cookbook.py`

## 🔧 Troubleshooting

### Common Issues
1. **Import Error**: Add `sys.path.insert(0, str(Path(__file__).parent.parent / "src"))`
2. **Memory Error**: Use `memory_optimizer` artifact first
3. **Schema Error**: Check `artifact.get_schema()` for correct input format

### Debug Commands
```python
# Check artifact schema
artifact = registry.get_artifact("sparse_attention")
print(artifact.implementation.get_schema())

# List all artifacts
for interface, artifacts in registry.list_artifacts().items():
    print(f"{interface}: {list(artifacts.keys())}")
```

## 📖 Next Steps

1. Run the examples: `python examples/artifact_usage.py`
2. Read the full documentation: `ARTIFACT_IO_DOCUMENTATION.md`
3. Try the cookbook recipes: `python examples/artifact_cookbook.py`
4. Create your first custom artifact using the templates in `artifact_cookbook.py`