### ROADMAP

#### prototype design
##### components
1. model definition 
   1. (probably in pytorch, migrating from transformers, much like qwen3 in nano-vllm)
   2. other options, C++ native or Rust native (not seen much of necessity)
2. tokenizer
   1. api/interface defination, normal mode/streaming mode, depends on the language (for python we have asyncio, not know much of C++/Rust)
3. scheduler
   1. continous batching 
   2. RPC with tokenizer and model executor, which is cost in python/cpp/rust
4. memory allocator
   1. api definition
   2. python/triton/c++/rust ?
5. computation allocator
   1. include the necessary kernel designs for a defined model (i believe there are plenty of triton kernel in swiftLLM, can also refer to sglang's integration with flash-infer)
