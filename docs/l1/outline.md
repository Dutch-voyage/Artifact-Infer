### Cooperative Programming

#### Core Design: Register and Combination

When **"scaling"** the system (single gpu => single machine => cluster), the system developer might have some options in detailed implenmentations. 

Once the implementations are settled, developer from all levels will (be forced to) be faced straight with these dull details. 

Two occasions are not desired:
1. The developer only wants to optimize in a smaller scale (e.g. single gpu), whereas he has to look at (and run) codes that operate at larger scales.
2. The developer want to access resources "far away" (from a differnet system level, abstraction level or remote process), yet they does not know where to find them, even if they manage to find, they does not understand. 
3. No comprehensive tests have been prepared or generated for newly added unit, so the developer has to test as a whole.

The fundamental rule is `Your code can be copy and pasted to where it is needed, and it can work`. 

The register and combine (code cropping) is basically a support for such "copy and paste" operations. 

```python
class ExampleService(Service):
    def __init__(self):
        pass

    def register(self, artifact):
        artifact_bind_to_service(artifact, self)
        service_bind_to_artifact(self, artifact)
    
    def call(self):
        pass

    def run(self)
        for part in pipeline:
            self.call(self.artifact_table[part.artfact_name])

    def combine(self):
        for artifact in self.artifact_table:
            revise_and_insert_code(self.code, artifact.code)
        
        return self.code
```

#### Case 1: write one's own part as if seeing all
```python
class ExampleA(Artifact):
    def __init__(self):
        pass
    
    def step_of_A(self):
        do_something_with(self.ExampleB.resource) # while this is no "self.ExampleB"
```

if not, we would do something like:
```python 
class ExampleA():
    def step_of_A(self, service: Service): # hook
        do_something_with(service.ExampleB.resource) 
        # the resource must locate at the same process space
        # or the the service will involve distributed process manager like `Ray`

@ray.remote
class ExampleB():
    pass

class Service():
    def __init__(self):
        self.ExampleA = ExampleA()
        self.ExampleB = ExampleB.remote()

```

or (message queue)

```python 
class ExampleA():
    def __init__(self):
        self.to_ExampleB = zmq.Context(2).socket(zmq.REQ)
        self.from_ExampleB = zmq.Context(2).socket(zmq.REP)
        self.to_ExampleB.connext(endpoint1)
        self.from_ExampleB.connect(endpoint2)


    async def step_of_A(self):
        while True:
            request = Request_to_B(method_name)
            self.to_ExampleB.send_pyobj(request)
            ret = await self.from_ExampleB.recv_pyobj()


class ExampleB():
    def __init__(self):
        # same with ExampleA
        pass
    
    async def step_of_B(self):
        while True:
            # same with ExampleA
```

#### Case 2: lowering codes into different levels of abstraction

Let's take KV cache compression as an example.
In most occasions, one only write a callback function based on hf transformer. 


```python 
class CustomKVCache(Cache):
    def __init__(self):
        pass

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        # override the naive implementation
        save_kv_to_cache_tensor()
        pass
```

- P.S.
For huggingface Cache implementation, see
https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/cache_utils.py#L742
The above `update` interface will be called during inference of an LLM, see Qwen3 as an example: 
https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3/modeling_qwen3.py#L210


For more flexible or more complex Cache customization. 
https://github.com/Zefan-Cai/KVCache-Factory/tree/main
```python
class KVclusterCustom():
    def update_kv(self, key_states, query_states, value_states, ...):
        # you can revise the interface to add inputs as you want
        # for example the query_states can be involved here
        do_something()
        return key_compressed, value_compressed

def llama_attn_forward_custom(
    # These are fixed interface definition, since they will be called in the complete transformer interface
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    self.kv_cluster = KVclusterCustom()
    before_update_KV()
    key_compressed, value_compressed = self.kv_cluster.update_kv(key_states, query_states, value_states)
    past_key_value.update(key_compressed, value_compressed, self.layer_idx, cache_kwargs)
    after_update_KV()

# monkey patch
import transformers
transformers.models.llama.modelling_llama.LlamaAttention.forward = llama_attn_forward_custom

```
- P.S
For KV Cluster (the component that do compression), see https://github.com/Zefan-Cai/KVCache-Factory/blob/main/pyramidkv/pyramidkv_utils.py 
For attention patching, see https://github.com/Zefan-Cai/KVCache-Factory/blob/main/pyramidkv/llama_model.py
In short, this level of customization can involve more diverse inputs to do compression (not just what are defined in Cache.update()). Also, the computation of attention can be adjusted accordingly. Yet everything will be restrained within this scope. 

The above two example are (almost) the two shallowest modification regardiong KV cache compression (can be effective already). Yet the latter version (KVCache-factory style) can enable a lower (or higher?) level of customization. 

#### Case 3: Profiling
TODO

#### Discussion
How to merge imports ?

1. relative import from local projects
2. import from third party
3. import that is being used in the registered method
4. import that is not being used in the registered method
(should we import unstateful class member method into the integrated script?)
(should we patch stateful method, i.e. involving calling self into calling the service method?)
