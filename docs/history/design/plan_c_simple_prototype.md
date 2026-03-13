# Plan C: Simple RESTful Prototype Design

## Overview
This document outlines a simple RESTful prototype design for the inference engine, keeping everything in a single process like SwiftLLM while providing clean modular interfaces and minimal communication overhead.

## Simple Modular Design

### Core Components (Single Process)

#### 1. Tokenizer Module (`src/tokenizer.py`)
```python
class SimpleTokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
```

#### 2. Model Module (`src/model.py`)
```python
class SimpleModel:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device = device
    
    def forward(self, input_ids: List[int]) -> torch.Tensor:
        return self.model(torch.tensor([input_ids]).to(self.device))
    
    def generate(self, prompt: str, max_tokens: int) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0])
```

#### 3. Scheduler Module (`src/scheduler.py`)
```python
class SimpleScheduler:
    def __init__(self, max_batch_size: int = 8):
        self.queue = deque()
        self.max_batch_size = max_batch_size
    
    def add_request(self, prompt: str, max_tokens: int) -> str:
        request_id = str(uuid.uuid4())[:8]
        self.queue.append({
            'id': request_id,
            'prompt': prompt,
            'max_tokens': max_tokens
        })
        return request_id
    
    def get_batch(self) -> List[Dict]:
        return [self.queue.popleft() for _ in range(min(len(self.queue), self.max_batch_size))]
```

## RESTful API Design

### Single FastAPI Server (`src/server.py`)
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
engine = SimpleEngine()  # All modules in one process

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerateRequest):
    return {"text": engine.generate(request.prompt, request.max_tokens)}

@app.post("/tokenize")
async def tokenize(text: str):
    return {"tokens": engine.tokenize(text)}

@app.get("/health")
async def health():
    return {"status": "ok"}
```

## Minimal Communication Pattern

### Direct Function Calls (Zero Overhead)
```python
class SimpleEngine:
    def __init__(self, model_path: str):
        self.tokenizer = SimpleTokenizer(model_path)
        self.model = SimpleModel(model_path)
        self.scheduler = SimpleScheduler()
    
    def generate(self, prompt: str, max_tokens: int) -> str:
        # Direct call - no serialization
        return self.model.generate(prompt, max_tokens)
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        # Simple batching - no queue
        return [self.model.generate(p, 50) for p in prompts]
```

## Minimal Logging

### Simple Logger
```python
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Only log essential info
logger.info(f"Generated {len(tokens)} tokens in {latency:.2f}s")
```

## SwiftLLM-Inspired Simplicity

### Single File Entry Point
```python
# main.py - Everything in one file for prototype
from transformers import AutoTokenizer, AutoModelForCausalLM

class SwiftLLM:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def __call__(self, prompt, max_tokens=50):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0])

# Usage
engine = SwiftLLM()
print(engine("Hello world"))
```

## RESTful Endpoints (Single Process)

### User APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate text from prompt |
| `/tokenize` | POST | Tokenize text |
| `/batch_generate` | POST | Batch generation |
| `/model_info` | GET | Model metadata |
| `/health` | GET | Health check |

### Algorithm/System Development APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/debug/attention_scores` | POST | Get attention scores for analysis |
| `/debug/latency_stats` | GET | Get timing statistics |
| `/debug/memory_usage` | GET | Memory usage breakdown |
| `/debug/profiling/start` | POST | Start profiling session |
| `/debug/profiling/stop` | POST | Stop profiling and get report |
| `/optimization/sparsity_mask` | POST | Apply sparsity mask to attention |
| `/optimization/batch_schedule` | POST | Custom batch scheduling |
| `/parallel/tensor_split` | POST | Split tensor for parallel computation |
| `/pipeline/step_execute` | POST | Execute specific pipeline step |
| `/cache/clear` | POST | Clear KV cache |
| `/cache/stats` | GET | Cache hit/miss statistics |

## Development Commands

```bash
# Install dependencies
pip install fastapi transformers torch uvicorn

# Run server
uvicorn server:app --reload --port 8000

# Test endpoints
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "Hello world", "max_tokens": 10}'
```

## Performance Characteristics

- **Startup**: ~5s (model loading)
- **First token**: ~100ms
- **Subsequent tokens**: ~20ms/token
- **Memory**: Model size + ~500MB overhead
- **CPU**: Single core for generation

## Algorithm Development Use Cases

### 1. Attention Analysis & Sparsification
```python
# RESTful API for attention research
POST /debug/attention_scores
{
  "prompt": "The cat sat on the mat",
  "layer_idx": 5,
  "head_idx": 3
}

Response:
{
  "attention_matrix": [[0.1, 0.8, 0.1], ...],
  "sparsity_ratio": 0.72,
  "entropy": 1.34
}

# Apply custom sparsity mask
POST /optimization/sparsity_mask
{
  "mask_pattern": "top_k",
  "k": 32,
  "threshold": 0.1
}
```

### 2. Pipeline Communication & Computation
```python
# Step-wise pipeline execution
POST /pipeline/step_execute
{
  "step": "embed",
  "input_ids": [101, 7592, 2088],
  "return_intermediate": true
}

# Async computation pipeline
POST /pipeline/schedule
{
  "steps": ["tokenize", "embed", "attention", "output"],
  "batch_size": 4,
  "overlap_comm": true
}
```

### 3. Parallel Tensor Computation
```python
# Split tensor for parallel processing
POST /parallel/tensor_split
{
  "tensor_shape": [4, 512, 768],
  "split_dim": 0,
  "num_splits": 2
}

# Execute parallel computation
POST /parallel/execute
{
  "splits": ["split_0", "split_1"],
  "operation": "matmul",
  "gather": true
}
```

### 4. System Monitoring & Optimization
```python
# Get detailed timing breakdown
GET /debug/latency_stats
Response:
{
  "tokenize_ms": 1.2,
  "embed_ms": 3.4,
  "attention_ms": 45.6,
  "output_ms": 2.1,
  "total_ms": 52.3
}

# Memory optimization feedback
GET /debug/memory_usage
{
  "model_weights_mb": 548,
  "kv_cache_mb": 12.3,
  "activation_mb": 45.6,
  "peak_memory_mb": 605.9
}
```

### 5. Custom Batch Scheduling
```python
# Algorithm developer's custom scheduler
POST /optimization/batch_schedule
{
  "algorithm": "priority_queue",
  "requests": [
    {"prompt": "Hello", "priority": 1, "max_tokens": 50},
    {"prompt": "World", "priority": 2, "max_tokens": 30}
  ],
  "max_batch_size": 4,
  "strategy": "minimize_latency"
}
```

## RESTful API Examples for Algorithm Engineers

### Case 1: Attention Sparsity Research
```bash
# Step 1: Get attention scores for analysis
curl -X POST http://localhost:8000/debug/attention_scores \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The quick brown fox", "layer_idx": 6}'

# Step 2: Apply learned sparsity pattern
curl -X POST http://localhost:8000/optimization/sparsity_mask \
  -H "Content-Type: application/json" \
  -d '{"sparsity_ratio": 0.8, "method": "top_k"}'

# Step 3: Measure impact on generation quality
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The quick brown fox", "max_tokens": 50}'
```

### Case 2: Pipeline Overlap Optimization
```bash
# Step 1: Profile current pipeline
curl -X POST http://localhost:8000/debug/profiling/start
curl -X POST http://localhost:8000/generate \
  -d '{"prompt": "Hello", "max_tokens": 100}'
curl -X POST http://localhost:8000/debug/profiling/stop

# Step 2: Test overlapping communication/computation
curl -X POST http://localhost:8000/pipeline/step_execute \
  -d '{"step": "attention", "overlap": true}'
```

### Case 3: Memory Optimization
```bash
# Step 1: Check memory usage pattern
curl http://localhost:8000/debug/memory_usage

# Step 2: Clear cache and remeasure
curl -X POST http://localhost:8000/cache/clear
curl http://localhost:8000/cache/stats
```

## Implementation Notes

### Minimal Overhead Design
- **Direct function calls** between modules (no serialization)
- **Optional logging** - disabled by default
- **Lightweight profiling** - only when requested
- **In-memory state** - no database persistence

### Development-Ready Features
- **Hot reload** - restart server with new algorithm
- **JSON responses** - easy parsing in any language
- **Error handling** - detailed error messages
- **Performance metrics** - built-in timing

### Quick Development Workflow
```bash
# 1. Start with simple server
python simple_server.py

# 2. Test your algorithm via REST
python test_algorithm.py  # Uses requests library

# 3. Iterate quickly
# Edit algorithm.py → server auto-reloads → retest via REST
```

## Communication Cost Optimization

### Direct Function Calls (Zero Overhead)
```python
# Simple direct calls - no serialization
class SimpleEngine:
    def process(self, prompt: str) -> str:
        tokens = self.tokenizer.encode(prompt)  # Direct call
        output = self.model.generate(tokens)    # Direct call
        return self.tokenizer.decode(output)    # Direct call
```

### Minimal Logging
```python
import time

class SimpleLogger:
    def __init__(self, enabled=False):
        self.enabled = enabled
    
    def log(self, message: str):
        if self.enabled:
            print(f"{time.time():.3f}: {message}")

logger = SimpleLogger(enabled=False)  # Disable by default
```

## File Structure
```
src/
├── simple_engine.py    # Main engine class
├── simple_server.py    # FastAPI server
└── simple_test.py      # Test script
```

## Quick Start

### 1. Install dependencies
```bash
pip install fastapi transformers torch uvicorn
```

### 2. Create simple server
```python
# simple_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# Load model once at startup
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

@app.post("/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer.encode(request.prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=request.max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": text}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Run and test
```bash
# Start server
python simple_server.py

# Test in another terminal
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 10}'
```