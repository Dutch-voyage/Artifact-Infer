### types of architectures

1. online server/offline engine 
2. tokenizer 
3. scheduler 
4. memory manager 
5. computation manager 
6. please add here if anything is missed

### prototype-level choice for current architectures

1. online server/offline engine 
just refer to vllm/sglang, for very simple function transfer
2. tokeniser
huggingface tokenizer is enough, consider fast tokenizer which is also from transformer
3. scheduler
need to define endpoints
should manage resources like control flow/execution plan and profiled results
4. memory manager
leave room for hierachical plans
the current implementation can only consider gpu-only memory (for KV cache)
5. computation manager
is a execution model for now 
    1. simple implementation from transformers' causal LLM
    2. one predefined plan with RESTapi for alternative kernels

### why micro-services for this project

Modern LLM infra-engineer and algorithm designer often fail to meet each other's request. This project aims to bring people together, and treat everybody as frontend developper. The micro-service design must be zero-cost, in order to accurately profile different system design. The mircor-service mush also be as decoupled as possible, so that algorithm designer can touch almost any part of LLM serving, without looking into the whole system. The microservice will also provide very easy debug, with interface example data available for and module. The user can update their execution plan for any part of LLM inference and run the LLM easily as a whole after unit testing. 

Please consider the system design base on the above intuition.