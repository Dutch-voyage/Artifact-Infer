# Implementation Plans: Prototype Inference Engine

Based on the roadmap.md analysis and repository evaluation, here are detailed implementation plans for building a prototype inference engine.

## Executive Summary

This document provides **5 distinct implementation plans** ranging from minimal educational (200 lines) to production-ready microservices (5k+ lines), with clear language choices, timelines, and migration paths.

---

## Plan A: SwiftLLM-Inspired Python-First (Recommended)

### **Overview**
- **Language Stack**: 85% Python + 10% Triton + 5% CUDA
- **Lines of Code**: 2,000-3,000 (proven via SwiftLLM)
- **Timeline**: 2-3 weeks
- **Team Size**: 1-2 developers
- **Performance**: vLLM-equivalent

### **Component Breakdown**

#### **1. Model Definition**
```python
# src/models/llama.py
class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        self.layers = nn.ModuleList([LlamaLayer(config) for _ in range(config.n_layers)])
    
    def forward(self, input_ids, past_key_values=None):
        # 400 lines total
```

#### **2. Tokenizer**
```python
# src/tokenizer/async_tokenizer.py
class AsyncTokenizer:
    async def encode(self, text: str) -> List[int]: ...
    async def decode(self, tokens: List[int]) -> str: ...
```

#### **3. Scheduler**
```python
# src/scheduler/continuous_batch.py
class ContinuousBatchScheduler:
    def step(self) -> List[Response]: ...
    def add_request(self, request: Request) -> None: ...
```

#### **4. Memory Allocator**
```python
# src/memory/paged_manager.py
class PagedMemoryManager:
    def allocate_blocks(self, seq_len: int) -> List[int]: ...
    def free_blocks(self, block_ids: List[int]) -> None: ...
```

#### **5. Computation Allocator**
```python
# src/kernels/attention.py (Triton)
@triton.jit
def attention_kernel(...): ...
```

### **File Structure**
```
swift-prototype/
├── src/
│   ├── models/
│   │   ├── llama.py           # 400 lines
│   │   └── config.py          # 100 lines
│   ├── scheduler/
│   │   └── batch_scheduler.py # 300 lines
│   ├── memory/
│   │   └── paged_manager.py   # 250 lines
│   ├── kernels/
│   │   ├── attention.py       # 200 lines
│   │   └── gemm.py            # 150 lines
│   ├── api/
│   │   └── server.py          # 200 lines
│   └── utils/
│       └── profiling.py       # 100 lines
├── tests/
│   ├── test_scheduler.py
│   └── test_memory.py
└── examples/
    └── offline.py
```

### **Quick Start Commands**
```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install torch transformers triton

# 2. Clone and setup
git clone https://github.com/interestingLSY/swiftLLM.git
cd swift-prototype
pip install -e .

# 3. Run basic test
python examples/offline.py --model-path /path/to/llama-7b
```

---

## Plan B: Rust-First High-Performance

### **Overview**
- **Language Stack**: 60% Rust + 30% Python + 10% CUDA
- **Lines of Code**: 3,000-4,000
- **Timeline**: 4-5 weeks
- **Performance**: SGLang-level

### **Component Implementation**

#### **1. Model Definition (Rust)**
```rust
// src/model/llama.rs
pub struct LlamaModel {
    layers: Vec<LlamaLayer>,
    embeddings: Embedding,
}

impl LlamaModel {
    pub fn forward(&mut self, input_ids: &[i32]) -> Tensor { ... }
}
```

#### **2. Scheduler (Rust)**
```rust
// src/scheduler/mod.rs
pub struct Scheduler {
    pending: VecDeque<Request>,
    running: Vec<Sequence>,
}

impl Scheduler {
    pub fn step(&mut self) -> Vec<ScheduledBatch> { ... }
}
```

#### **3. Memory Allocator (Rust + CUDA)**
```rust
// src/memory/pool.rs
pub struct MemoryPool {
    blocks: Vec<Block>,
    free_list: Vec<usize>,
}
```

### **Setup Commands**
```bash
# 1. Rust setup
cargo new rust-prototype
cd rust-prototype
cargo add torch --features python

# 2. CUDA bindings
cargo add cuda-runtime-sys
cargo add pyo3

# 3. Build and test
cargo build --release
cargo test
```

---

## Plan C: Hybrid Microservices

### **Overview**
- **Language Stack**: 70% Python + 25% Rust + 5% CUDA
- **Architecture**: RESTful microservices
- **Timeline**: 4-6 weeks
- **Scalability**: High (horizontal)

### **Service Architecture**

#### **Service 1: Model Service (Python)**
```python
# services/model/main.py
from fastapi import FastAPI
import torch

app = FastAPI()

@app.post("/v1/forward")
async def forward(request: ForwardRequest):
    return model.forward(request.input_ids)
```

#### **Service 2: Scheduler Service (Rust)**
```rust
// services/scheduler/main.rs
#[tokio::main]
async fn main() {
    let scheduler = Scheduler::new();
    // REST API endpoints...
}
```

#### **Service 3: Memory Service (Rust)**
```rust
// services/memory/main.rs
#[post("/v1/allocate")]
async fn allocate_memory(size: usize) -> AllocationResponse {
    // Paged memory management
}
```

### **Docker Compose Setup**
```yaml
# docker-compose.yml
version: '3.8'
services:
  model:
    build: ./services/model
    ports: ["8001:8000"]
  scheduler:
    build: ./services/scheduler
    ports: ["8002:8000"]
  memory:
    build: ./services/memory
    ports: ["8003:8000"]
```

---

## Plan D: Educational C++ Approach

### **Overview**
- **Language Stack**: 70% C++ + 20% Python + 10% CUDA
- **Lines of Code**: 1,000-1,500
- **Timeline**: 3-4 weeks
- **Purpose**: Educational understanding

### **Component Design**

#### **1. Minimal Model (C++)**
```cpp
// src/model/llama.h
class LlamaModel {
public:
    Tensor forward(const Tensor& input_ids);
private:
    std::vector<Layer> layers;
};
```

#### **2. Simple Scheduler (C++)**
```cpp
// src/scheduler/simple.h
class SimpleScheduler {
public:
    Batch get_next_batch();
private:
    std::queue<Request> requests;
};
```

### **Build System**
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(edu-inference)

find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

add_library(inference SHARED src/inference.cpp)
target_link_libraries(inference ${CUDA_LIBRARIES})
```

---

## Plan E: Triton-First Research Platform

### **Overview**
- **Language Stack**: 50% Python + 40% Triton + 10% CUDA
- **Focus**: Research and experimentation
- **Timeline**: 5-6 weeks
- **Features**: Autotuning, dynamic kernels

### **Advanced Features**

#### **1. Dynamic Kernel Generation**
```python
# src/kernels/generator.py
class KernelGenerator:
    def generate_attention(self, config: Config) -> TritonKernel:
        # Generate optimized kernels based on shapes
```

#### **2. Autotuning Infrastructure**
```python
# src/autotune/manager.py
class Autotuner:
    def tune_kernel(self, kernel: TritonKernel, shapes: List[Shape]) -> BestConfig:
        # Benchmark different configurations
```

---

## **Implementation Roadmap by Plan**

### **Phase 1: Foundation (Week 1-2)**
**Plan A (Recommended)**:
```bash
# Day 1-2: Setup
pip install torch transformers triton fastapi uvicorn
pip install pytest black mypy

# Day 3-5: Core model
python -c "from src.models.llama import LlamaModel; model = LlamaModel(config)"

# Day 6-7: Basic scheduler
python tests/test_scheduler.py
```

### **Phase 2: Memory Optimization (Week 2-3)**
```bash
# Add paged attention
git submodule add https://github.com/vllm-project/vllm.git vendor/vllm

# Implement memory manager
python src/memory/paged_manager.py
```

### **Phase 3: API Layer (Week 3-4)**
```bash
# FastAPI setup
pip install fastapi uvicorn[standard]
python -m uvicorn src.api.server:app --reload

# Testing
pytest tests/test_api.py -v
```

---

## **Testing Strategy by Plan**

### **Plan A Testing**:
```bash
# Unit tests
pytest tests/test_scheduler.py -v
pytest tests/test_memory.py -v

# Integration tests
python examples/benchmark.py --model llama-7b --dataset sharegpt

# Performance validation
python scripts/compare_vllm.py
```

### **Plan B Testing**:
```bash
# Rust tests
cargo test --release

# Python integration
python tests/integration_test.py

# Performance benchmarks
cargo bench
```

### **Plan C Testing**:
```bash
# Service testing
docker-compose up --build

# Load testing
locust -f tests/load_test.py --host http://localhost:8001
```

---

## **Resource Requirements**

| Plan | GPU Memory | RAM | Storage | Development Time |
|------|------------|-----|---------|------------------|
| **A** | 8GB+ | 16GB | 50GB | 2-3 weeks |
| **B** | 8GB+ | 16GB | 50GB | 4-5 weeks |
| **C** | 8GB+ | 32GB | 100GB | 4-6 weeks |
| **D** | 4GB+ | 8GB | 20GB | 3-4 weeks |
| **E** | 12GB+ | 16GB | 75GB | 5-6 weeks |

---

## **Migration Paths**

### **A → C → B (Recommended)**
1. **Start**: SwiftLLM foundation (Plan A)
2. **Scale**: Microservices (Plan C)
3. **Optimize**: Rust performance (Plan B)

### **Educational Path: D → A → C**
1. **Learn**: C++ basics (Plan D)
2. **Implement**: Python version (Plan A)
3. **Scale**: Microservices (Plan C)

### **Research Path: E → A → B**
1. **Experiment**: Triton kernels (Plan E)
2. **Validate**: Python implementation (Plan A)
3. **Production**: Rust optimization (Plan B)

---

## **Final Recommendation**

**Start with Plan A (SwiftLLM Foundation)**:

### **Why Plan A?**
- ✅ **Proven 2k-line codebase** with vLLM-equivalent performance
- ✅ **Immediate testability** with existing benchmarks
- ✅ **Clear migration path** to other plans
- ✅ **Strong community support** from SwiftLLM
- ✅ **Lowest risk** with highest learning value

### **Next Steps (Plan A)**:
1. **Week 1**: Clone SwiftLLM, run basic tests
2. **Week 2**: Add FastAPI layer and memory optimization
3. **Week 3**: Add comprehensive testing and documentation
4. **Week 4**: Performance benchmarking against vLLM

### **Quick Start**:
```bash
git clone https://github.com/interestingLSY/swiftLLM.git
cd swift-llm-prototype
pip install -r requirements.txt
python examples/offline.py --model-path /path/to/llama-7b
```

Ready to implement? Start with Plan A for immediate results and clear upgrade paths.