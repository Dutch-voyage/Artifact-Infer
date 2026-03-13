# Artifact IO Documentation

This document provides comprehensive input/output examples and schemas for all built-in artifacts in the system.

## Quick Reference

| Artifact | Interface | Description | Complexity |
|----------|-----------|-------------|------------|
| `sparse_attention` | attention | Sparse attention with configurable sparsity | Medium |
| `sliding_window` | attention | Local attention with sliding window | Low |
| `tensor_parallel` | parallel | Tensor splitting for parallel processing | Medium |
| `pipeline_parallel` | parallel | Pipeline stage parallelism | High |
| `lru_cache` | memory | Least Recently Used cache | Low |
| `advanced_attention` | attention | Multi-pattern attention processing | High |
| `memory_optimizer` | memory | LLM memory analysis and optimization | High |

---

## Attention Artifacts

### 1. sparse_attention

**Description**: Applies sparsity to attention matrices by zeroing out low-magnitude connections.

**Input Schema**:
```json
{
  "attention_matrix": [[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]],
  "sparsity_ratio": 0.5
}
```

**Example Input**:
```json
{
  "attention_matrix": [
    [1.0, 0.8, 0.6, 0.4],
    [0.8, 1.0, 0.7, 0.5],
    [0.6, 0.7, 1.0, 0.8],
    [0.4, 0.5, 0.8, 1.0]
  ],
  "sparsity_ratio": 0.6
}
```

**Example Output**:
```json
{
  "processed_attention": [
    [1.0, 0.8, 0.0, 0.0],
    [0.8, 1.0, 0.7, 0.0],
    [0.0, 0.7, 1.0, 0.8],
    [0.0, 0.0, 0.8, 1.0]
  ],
  "sparsity_ratio": 0.6,
  "non_zero_count": 10,
  "total_elements": 16
}
```

**Use Cases**:
- Reduce memory usage in large attention matrices
- Speed up attention computation
- Implement sparse transformer variants

---

### 2. sliding_window

**Description**: Applies local attention using a sliding window, keeping only nearby token connections.

**Input Schema**:
```json
{
  "attention_matrix": [[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]],
  "window_size": 2
}
```

**Example Input**:
```json
{
  "attention_matrix": [
    [1.0, 0.8, 0.6, 0.4, 0.2],
    [0.8, 1.0, 0.7, 0.5, 0.3],
    [0.6, 0.7, 1.0, 0.8, 0.6],
    [0.4, 0.5, 0.8, 1.0, 0.7],
    [0.2, 0.3, 0.6, 0.7, 1.0]
  ],
  "window_size": 1
}
```

**Example Output**:
```json
{
  "processed_attention": [
    [1.0, 0.8, 0.0, 0.0, 0.0],
    [0.8, 1.0, 0.7, 0.0, 0.0],
    [0.0, 0.7, 1.0, 0.8, 0.0],
    [0.0, 0.0, 0.8, 1.0, 0.7],
    [0.0, 0.0, 0.0, 0.7, 1.0]
  ],
  "window_size": 1,
  "retained_connections": 13
}
```

**Use Cases**:
- Long sequence processing
- Local attention patterns
- Memory-efficient inference

---

### 3. advanced_attention

**Description**: Multi-pattern attention processing supporting sparse, sliding, global, local, and dilated patterns.

**Input Schema**:
```json
{
  "attention_matrix": [[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]],
  "pattern": "sparse",
  "sparsity_ratio": 0.5,
  "window_size": 2,
  "global_tokens": [0],
  "block_size": 4,
  "dilation": 2
}
```

**Example Input - Global Pattern**:
```json
{
  "attention_matrix": [
    [1.0, 0.1, 0.1],
    [0.1, 1.0, 0.1],
    [0.1, 0.1, 1.0]
  ],
  "pattern": "global",
  "global_tokens": [0]
}
```

**Example Output - Global Pattern**:
```json
{
  "processed_attention": [
    [1.0, 0.1, 0.1],
    [0.1, 1.0, 0.1],
    [0.1, 0.1, 1.0]
  ],
  "global_tokens": [0],
  "global_attention_positions": 3
}
```

**Example Input - Local Pattern**:
```json
{
  "attention_matrix": [
    [1.0, 0.8, 0.6, 0.4],
    [0.8, 1.0, 0.7, 0.5],
    [0.6, 0.7, 1.0, 0.8],
    [0.4, 0.5, 0.8, 1.0]
  ],
  "pattern": "local",
  "block_size": 2
}
```

**Example Output - Local Pattern**:
```json
{
  "processed_attention": [
    [1.0, 0.8, 0.0, 0.0],
    [0.8, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.8],
    [0.0, 0.0, 0.8, 1.0]
  ],
  "block_size": 2,
  "blocks": 2
}
```

**Use Cases**:
- Flexible attention patterns
- Research experiments
- Custom transformer variants

---

## Parallel Artifacts

### 4. tensor_parallel

**Description**: Splits tensors for parallel processing across multiple devices.

**Input Schema**:
```json
{
  "tensor": [1, 2, 3, 4, 5, 6],
  "num_splits": 2
}
```

**Example Input**:
```json
{
  "tensor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
  "num_splits": 4
}
```

**Example Output**:
```json
{
  "result": [
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0]
  ],
  "num_splits": 4,
  "split_size": 2,
  "original_shape": [8]
}
```

**Use Cases**:
- Multi-GPU training
- Large model inference
- Distributed computing

---

### 5. pipeline_parallel

**Description**: Implements pipeline parallelism for sequential model processing.

**Input Schema**:
```json
{
  "data": [1, 2, 3, 4, 5, 6],
  "num_stages": 3
}
```

**Example Input**:
```json
{
  "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 12],
  "num_stages": 3
}
```

**Example Output**:
```json
{
  "pipeline_stages": [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 12]
  ],
  "num_stages": 3,
  "stage_size": [4, 4, 2],
  "total_elements": 10
}
```

**Use Cases**:
- Large model training
- Sequential processing
- Memory optimization

---

## Memory Artifacts

### 6. lru_cache

**Description**: Least Recently Used cache for efficient memory management.

**Input Schema**:
```json
{
  "operation": "put",
  "key": "example_key",
  "value": "example_value",
  "max_size": 1000
}
```

**Supported Operations**:
- `put`: Store a key-value pair
- `get`: Retrieve a value by key
- `delete`: Remove a specific key
- `clear`: Clear all cache
- `stats`: Get cache statistics

**Example Input - Put Operation**:
```json
{
  "operation": "put",
  "key": "model_weights",
  "value": {"layer1": [1, 2, 3], "layer2": [4, 5, 6]}
}
```

**Example Output - Put Operation**:
```json
{
  "result": "stored",
  "key": "model_weights",
  "cache_size": 1
}
```

**Example Input - Stats Operation**:
```json
{
  "operation": "stats"
}
```

**Example Output - Stats Operation**:
```json
{
  "max_size": 1000,
  "current_size": 3,
  "hits": 5,
  "misses": 2,
  "hit_rate": 0.714,
  "keys": ["model_weights", "attention_cache", "embedding_cache"]
}
```

**Use Cases**:
- Model weight caching
- Attention matrix caching
- Embeddings caching

---

### 7. memory_optimizer

**Description**: LLM memory analysis and optimization recommendations.

**Input Schema**:
```json
{
  "operation": "analyze",
  "model_size": 500000000,
  "batch_size": 4,
  "sequence_length": 512,
  "current_usage": {},
  "target_memory": 8000000000,
  "model_name": "gpt2"
}
```

**Supported Operations**:
- `analyze`: Analyze memory usage
- `optimize`: Get optimization recommendations
- `clear_cache`: Clear specific cache
- `get_recommendations`: Get model-specific recommendations

**Example Input - Analyze Operation**:
```json
{
  "operation": "analyze",
  "model_size": 500000000,
  "batch_size": 8,
  "sequence_length": 1024
}
```

**Example Output - Analyze Operation**:
```json
{
  "model_size_mb": 476.837,
  "attention_memory_mb": 32.0,
  "activations_memory_mb": 24576.0,
  "total_memory_mb": 25084.837,
  "memory_breakdown": {
    "model": 500000000,
    "attention": 33554432,
    "activations": 25769803776
  }
}
```

**Example Input - Get Recommendations**:
```json
{
  "operation": "get_recommendations",
  "model_name": "gpt2-medium"
}
```

**Example Output - Get Recommendations**:
```json
{
  "model_name": "gpt2-medium",
  "estimated_model_size_mb": 1430.511,
  "recommended_batch_sizes": {
    "8GB_gpu": 4,
    "16GB_gpu": 8,
    "24GB_gpu": 16
  },
  "optimization_suggestions": [
    "Use attention sparsity for large sequences",
    "Implement gradient checkpointing",
    "Use mixed precision training"
  ]
}
```

**Use Cases**:
- Memory planning
- Batch size optimization
- Model deployment planning

---

## Pipeline Examples

### Basic Pipeline

```python
pipeline = [
    {"artifact": "memory_optimizer", "params": {"operation": "analyze", "model_size": 500000000}},
    {"artifact": "sparse_attention", "params": {"sparsity_ratio": 0.6}},
    {"artifact": "tensor_parallel", "params": {"num_splits": 2}}
]
```

### Advanced Pipeline with Conditional Execution

```python
def should_apply_sparsity(data):
    return data.get("total_memory_mb", 0) > 10000

pipeline = [
    {"artifact": "memory_optimizer", "params": {"operation": "analyze"}},
    {"artifact": "advanced_attention", "params": {"pattern": "sparse"}, "conditional": should_apply_sparsity},
    {"artifact": "lru_cache", "params": {"operation": "put", "key": "optimized_weights"}}
]
```

---

## Error Handling

### Common Errors and Solutions

1. **Invalid Input Schema**
   - Error: `KeyError: 'required_field'`
   - Solution: Check artifact schema with `registry.get_artifact(name).implementation.get_schema()`

2. **Invalid Operation**
   - Error: `ValueError: Unsupported operation`
   - Solution: Check supported operations in artifact documentation

3. **Memory Constraints**
   - Error: `MemoryError`
   - Solution: Use `memory_optimizer` artifact to get recommendations

4. **Shape Mismatches**
   - Error: `ValueError: Incompatible shapes`
   - Solution: Verify tensor dimensions match expected input format

---

## Performance Tips

1. **Zero-Cost Mode**: Use `ExecutionMode.ZERO_COST` for production
2. **Batch Processing**: Process multiple items together when possible
3. **Cache Reuse**: Use `lru_cache` for frequently accessed data
4. **Memory Analysis**: Always analyze memory usage before deployment
5. **Pipeline Optimization**: Chain artifacts to minimize data transfer

---

## Testing Artifacts

```python
# Quick test function
def test_artifact(name, input_data):
    try:
        result = registry.execute_artifact(name, input_data, ExecutionMode.ZERO_COST)
        print(f"✓ {name}: Success")
        return result
    except Exception as e:
        print(f"✗ {name}: {e}")
        return None

# Test all artifacts
test_artifact("sparse_attention", {"attention_matrix": [[1,0.5],[0.5,1]], "sparsity_ratio": 0.5})
test_artifact("lru_cache", {"operation": "stats"})
```