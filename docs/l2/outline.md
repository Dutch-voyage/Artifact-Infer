### Adding a new function to an existing codebase

For an existing and **developing** framework (take inference engine as an example), analyzing its code structure is extremely painful. 


#### Case 1 vLLM v1

**vLLM** https://blog.vllm.ai/2025/01/27/v1-alpha-release.html has a major code refractoring after developing less than a year. 

https://github.com/vllm-project/vllm/issues/8779

> - Highlights of the new design:
> 
> - Driver process + SPMD workers
>   - When TP=n & PP=m, vLLM engine will have n*m + 1 processes in total.
>     - Corollary: even when using a single GPU, we will have 2 processes.
>   - The driver process will have the scheduler, memory manager, etc.
>   - The workers are stateful, maintaining most of the request states.
>     - The driver will only send the “diffs”
>       - New request: input token IDs & block tables & sampling params, etc.
>       - In-flight request: scheduled request IDs, new block IDs (no token IDs, sampling params, etc.)
>   - Clean up data structures like SeqGroupMetadata
> - Async single-step scheduling, instead of multi-step scheduling
>   - Scheduler will schedule the n+1-th step, while the worker is executing the n-th step.
>   - We will reuse the code from multi-step scheduling to incrementally update the model inputs.
>   - Needs a special care for PP, since the output token IDs from the last stage should be sent to the first stage.
> - De-tokenizer moves to the driver process
>   - Async de-tokenization can be regarded as part of async scheduling
> - Native support for different types of model states
>   - Regular KV cache, Mamba cache, encoder cache, etc.
>   - Dedicated memory manager & block table for each type of cache
> - Drop beam search from vLLM engine
>   - Provide a solution to emulate beam search outside vLLM engine
> - Prefix-caching as a first-class feature
>   - Implement parallel sampling via prefix caching
>   - Remove the concept of SequenceGroup
>   - Optimize prefix caching overheads
> - Remove/minimize PyObjectCache

**None of them are direct algorithmatic progress!**

Yet there are still algorithm-specific designs that affect **how to organize your code**. 

1. Scheduling 
   1. deciding which (group) of reqeusts to run (SLO, service-level objective) 
   2. preparing metadata (**can be very costly**) 
2. Sampling
   1. some sampling ideas are contraditory (beam search) 
   2. consistency 
3. Caching System 
   1. prefix caching (Casacade Attention) 
   2. caching metadata 

#### Case 2 Flashinfer backend
https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout
1. add attention artifacts
2. metadata adaptation
3. other refactoring (as free as possible)
