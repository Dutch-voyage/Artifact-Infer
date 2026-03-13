### xylo-infer

> _xylo_: **_wood-made_**
> This repo aims to create fully-customized, yet not necessarilly performant. with clear apis/test cases that may (or may not) leave room for easiler implementation of new ideas within the scope of inference-engine.

There are many complicated/distinct (irrelavant even worse) technological scopes that can be included within the term "Inference Engine". 

To list a few:

Architecture-wise

- attention variants 
- KV management (sparsification, indexing/offloading)
- MoE
- quantization
- speculative-decoding/multi-token-prediction

Use scenario

- cloud serving
- RL rollout 
- edge computing

Optimization-level
- kernel-level optimization (fusion, communication overlapping)
- parallelization 
- hierachy management 
(memory: KV, compute: cpu-gpu, heterogenous system)
- request scheuling (agent workflow)

Is it possible that one can combine these topics together to build a realy hand-made inference engine on his/her own? Not very likely, unless you can cover tremendous work done by vllm/sglang/TensorRT groups (even if you can, fairly speaking, these matured open-source project cannot combine all topics well, as some of them are in essence contradicted). 

So why on-earth should one create another repo trying to cover all these problems together? The answer is rather straight-forward, these projects are easily made by AI coders, if separately, completely ignoring system-level design principles. This process is much like creating **some wood toys, with literally no practical value, yet looks real or at least somehow artistic** (you don't have to be aesthetically valuable to be artistic, as far as I understand). 

Though easily becoming rubbish, wood-mades have a very good quality --- they can be hand-crafted easily. This feature, I believe, is easily achieved by AIs, they are more patient collaborators than most humans. 