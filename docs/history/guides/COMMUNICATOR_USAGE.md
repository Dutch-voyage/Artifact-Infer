# Cross-Service Communicator Usage Guide

The communicator system provides three modes of cross-service communication:

## Communication Modes

### 1. Async I/O Mode (`CommunicationMode.ASYNC_IO`)
- **Use case**: I/O heavy operations
- **Implementation**: Uses asyncio and ThreadPoolExecutor
- **Performance**: Good for network-bound operations

### 2. Multiprocess Mode (`CommunicationMode.MULTIPROCESS`)
- **Use case**: Compute-heavy operations
- **Implementation**: Uses multiprocessing with ProcessPoolExecutor
- **Performance**: Utilizes multiple CPU cores

### 3. In-Process Mode (`CommunicationMode.IN_PROCESS`)
- **Use case**: Simple function calls
- **Implementation**: Direct function calls with partial function support
- **Performance**: Zero overhead

## Quick Start

### Basic Usage

```python
import asyncio
from src.services.service_manager import ServiceManager
from src.communication.communicator import CommunicationMode

async def main():
    # Create service manager with specific communication mode
    manager = ServiceManager(communication_mode=CommunicationMode.ASYNC_IO)
    
    # Register services
    from src.services.compute_manager import ComputeManagerService
    compute_service = ComputeManagerService()
    manager.register_service("compute", compute_service)
    
    # Call service methods
    result = await manager.call_service("compute", "load_model", model_name="gpt2")
    print(result.data)
    
    # Shutdown
    manager.shutdown()

asyncio.run(main())
```

### Mixed Mode Usage

```python
# Register different services with different communication modes
manager = ServiceManager()

# I/O heavy service uses async I/O
manager.register_service("tokenizer", TokenizerService(), CommunicationMode.ASYNC_IO)

# Compute heavy service uses multiprocessing
manager.register_service("compute", ComputeManagerService(), CommunicationMode.MULTIPROCESS)

# Simple service uses in-process
manager.register_service("memory", MemoryManagerService(), CommunicationMode.IN_PROCESS)
```

### Partial Functions

```python
# Register simple functions for in-process calls
def calculate_attention(query, key, value):
    return query @ key.T @ value

manager.register_partial_function("attention", calculate_attention)

# Call partial function directly
result = manager.call_partial_function("attention", q, k, v)
```

## Configuration

```python
from src.communication.communicator import CommunicationConfig

config = CommunicationConfig(
    mode=CommunicationMode.ASYNC_IO,
    max_workers=4,        # Max worker threads/processes
    timeout=30.0,         # Timeout in seconds
    max_retries=3,        # Max retry attempts
    retry_delay=0.1       # Initial retry delay
)
```

## Service Registration

### For Async I/O and In-Process
```python
service_instance = YourService()
manager.register_service("service_name", service_instance)
```

### For Multiprocess
```python
# Register service class (not instance)
manager.register_service("service_name", YourServiceClass)
```

## Error Handling

The communicator system includes built-in retry mechanisms and error handling:

```python
try:
    result = await manager.call_service("service", "method", **params)
    if result.status == "error":
        print(f"Error: {result.error}")
except Exception as e:
    print(f"Communication error: {e}")
```

## Examples

Run the provided examples:

```bash
python examples/communicator_usage.py
```

Run tests:

```bash
python -m pytest tests/test_communicator.py -v
```