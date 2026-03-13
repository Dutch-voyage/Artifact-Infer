### Artifact abstraction

You should define what is required fro the artifacts, to ensure they can 
1. be registered by the services, 
2. be called (RESTful and non-RESTful) by the services 
3. communicate to other artifacts with zero-cost

the artifacts mush have **only one API** to be called directly to execute, the communication function is to ensure directly link to other artifacts, that can be called via services, but not called internally of the artifacts' own function. 

### RESTful and non_RESTful api

as suggested in the simplification_specification.md
```python
class SimplifiedMemory:
    def allocate(self, size: int, strategy: str) -> MemoryHandle
    def free(self, handle: MemoryHandle)
    
    # RESTful equivalents
    async def post_allocate(self, data: dict) -> RESTfulResponse
    async def delete_free(self, data: dict) -> RESTfulResponse
```

For each artifacts

### Structure of service

You should move some of the implementation of services of components into separate files in src/services. 

Additionally, please design some of the basic function for each service to call artifacts effectively. 