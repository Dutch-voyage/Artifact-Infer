# Micro-Service Architecture Simplification Specification

## Overview

This specification addresses the simplification demands from `simplify_demands.md` by proposing a **Unified RESTful-Zero-Cost Architecture** that eliminates unnecessary serialization overhead while maintaining horizontal scalability and RESTful API benefits for artifact development.

## Core Simplification Principles

### 1. Zero-Cost RESTful Abstraction

**Problem**: Traditional micro-services require serialization/deserialization overhead for all communications.

**Solution**: Implement a dual-mode architecture where:
- **Research Mode**: Full RESTful API with JSON serialization for artifact development
- **Production Mode**: Zero-cost direct function calls with no serialization overhead

```python
# Zero-cost mode (production)
result = tokenizer.tokenize("Hello world")  # Direct call

# RESTful mode (research/debugging)
result = await tokenizer.post_tokenize({"text": "Hello world"})  # RESTful API
```

### 2. Unified Interface Design

**Problem**: Multiple serialization formats and inconsistent APIs across services.

**Solution**: Single unified interface that supports both direct calls and RESTful patterns:

```python
class UnifiedService:
    def tokenize(self, text: str) -> TokenizedBatch:  # Zero-cost
    async def post_tokenize(self, data: dict) -> RESTfulResponse:  # RESTful
```

### 3. Horizontal Scaling Through RESTful Resources

**Problem**: Difficulty in adding new attention processing or parallelism patterns.

**Solution**: Treat artifacts as RESTful resources that can be dynamically loaded:

```python
# Adding new attention sparsification
POST /artifacts/attention/sparsity
{
    "code": "def sparse_attention(...): ...",
    "parameters": {"sparsity_ratio": 0.8}
}

# Adding new parallelism pattern
POST /artifacts/parallel/tensor_split
{
    "code": "def tensor_parallel_split(...): ...",
    "pattern": "model_parallel"
}
```

## Simplified Architecture Components

### 1. Unified Service Registry

**Single registry** for all services with dynamic loading:

```python
class ServiceRegistry:
    def register_artifact(self, name: str, code: str, interface: str):
        """Register new artifact as RESTful resource"""
        
    def get_service(self, name: str, mode: str = "zero_cost"):
        """Get service in zero-cost or RESTful mode"""
```

### 2. Python Object Pass-Through

**Efficient Python object passing** without serialization:

```python
# Instead of:
json_data = serialize(tensor)
send_over_network(json_data)
tensor = deserialize(json_data)

# Use:
tensor = service.process(tensor)  # Direct pass-through
```

### 3. RESTful API as Development Interface

**RESTful APIs for artifact development** with zero-cost toggle:

```python
# Development environment
env = DevelopmentEnvironment()
env.enable_restful_mode(True)  # Full RESTful API

# Production deployment
env.enable_restful_mode(False)  # Zero-cost direct calls
```

## Simplified Service Interfaces

### 1. Tokenizer Service
```python
class SimplifiedTokenizer:
    def tokenize(self, text: str) -> List[int]
    def detokenize(self, tokens: List[int]) -> str
    
    # RESTful equivalents for development
    async def post_tokenize(self, data: dict) -> RESTfulResponse
    async def get_detokenize(self, data: dict) -> RESTfulResponse
```

### 2. Model Service
```python
class SimplifiedModel:
    def forward(self, input_ids: List[int]) -> ModelOutput
    def generate(self, prompt: str, max_tokens: int) -> str
    
    # RESTful equivalents
    async def post_forward(self, data: dict) -> RESTfulResponse
    async def post_generate(self, data: dict) -> RESTfulResponse
```

### 3. Memory Service
```python
class SimplifiedMemory:
    def allocate(self, size: int, strategy: str) -> MemoryHandle
    def free(self, handle: MemoryHandle)
    
    # RESTful equivalents
    async def post_allocate(self, data: dict) -> RESTfulResponse
    async def delete_free(self, data: dict) -> RESTfulResponse
```

## Horizontal Scaling Patterns

### 1. artifact-as-Resource Pattern

**Adding new attention processing**:

```python
# Register new attention artifact
registry.register_artifact(
    name="sparse_attention",
    code="""
def sparse_attention(query, key, value, sparsity_mask):
    # Implementation for sparse attention
    return masked_attention
""",
    interface="attention"
)

# Use via RESTful API
POST /artifacts/sparse_attention/execute
{
    "query": tensor_data,
    "sparsity_ratio": 0.8
}
```

### 2. Parallelism-as-Resource Pattern

**Adding new parallelism**:

```python
# Register tensor parallelism
registry.register_artifact(
    name="tensor_parallel",
    code="""
def tensor_parallel_split(tensor, num_splits):
    # Split tensor across devices
    return split_tensors
""",
    interface="parallel"
)

# Use via RESTful API
POST /artifacts/tensor_parallel/split
{
    "tensor": tensor_data,
    "num_splits": 4
}
```

## Implementation Strategy

### Phase 1: Core Simplification
1. **Replace serialization** with Python object pass-through
2. **Unify interfaces** to support both direct calls and RESTful
3. **Implement dual-mode** (zero-cost + RESTful)

### Phase 2: Horizontal Scaling
1. **artifact registry** for dynamic loading
2. **RESTful resource endpoints** for each artifact
3. **Hot-swapping** capability for artifact updates

### Phase 3: Optimization
1. **Memory-mapped tensors** for zero-copy sharing
2. **Shared memory** for inter-process communication
3. **CUDA IPC** for GPU tensor sharing

## File Structure Simplification

```
src/
├── core/
│   ├── unified_service.py      # Unified zero-cost/RESTful interface
│   ├── service_registry.py     # artifact registration and discovery
│   └── zero_cost_transport.py  # Zero-overhead object passing
├── services/
│   ├── tokenizer_service.py    # Simplified tokenizer
│   ├── model_service.py       # Simplified model
│   └── memory_service.py      # Simplified memory
└── artifacts/
    ├── attention/
    │   ├── sparse_attention.py
    │   └── sliding_window.py
    └── parallel/
        ├── tensor_parallel.py
        └── pipeline_parallel.py
```

## Migration Path

### From Current to Simplified

1. **Identify serialization bottlenecks** in current micro-services
2. **Replace with unified interfaces** maintaining backward compatibility
3. **Gradually migrate** RESTful endpoints to zero-cost mode
4. **Add artifact registry** for horizontal scaling

### Backward Compatibility

- **Existing RESTful APIs** remain unchanged for external clients
- **Internal services** use zero-cost mode by default
- **Development mode** enables full RESTful debugging

## Performance Targets

- **Zero serialization overhead** in production mode
- **RESTful API latency** < 1ms for debugging
- **artifact registration** < 100ms for hot-swapping
- **Horizontal scaling** linear with number of artifacts

## Example Usage

### artifact Developer Workflow
```python
# 1. Register new attention artifact
registry.register_artifact(
    name="my_attention",
    code=my_attention_implementation,
    interface="attention"
)

# 2. Test via RESTful API
curl -X POST http://localhost:8000/artifacts/my_attention \
  -d '{"query": [[1,2,3]], "key": [[4,5,6]]}'

# 3. Deploy to production
registry.enable_zero_cost_mode("my_attention")
```

### System Integration
```python
# Zero-cost usage in production
result = model_service.forward(input_ids)

# RESTful usage for debugging
debug_result = await model_service.post_forward({"input_ids": input_ids})
```

## Summary

This specification provides a **unified architecture** that:
- **Eliminates serialization overhead** through zero-cost abstraction
- **Maintains RESTful benefits** for artifact development
- **Enables horizontal scaling** through artifact-as-resource pattern
- **Provides seamless migration** from current micro-service architecture

The key insight is that **RESTful APIs and zero-cost abstraction are not mutually exclusive** - we can have both through intelligent interface design and dual-mode operation.