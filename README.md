# Artifact-Infer

Yet another inference engine friendly for beginners and users. 

## Design Ideas

*Build your inference engine like legos.*

DIY an inference engine from scratch or adapt from existing ones is difficult. Therefore a good start should make a beginer/developer feel ***"I can add code to some place that should appear in an existing inference engine"***. 

To achieve this, this repo is built upon two concepts: **Service** and **Artifact**. For a brief review, refer to document [l1](docs/l1/outline.md). 

## About documents

The rest of the documents should include a brief ideas of how each version implement some function. These documents may also include some interesting observation from some buggy feature. 

The versions may not be necessarily consecutive, we may need a tree structure for contents later. 

## Testing and developing

This repo is built upon `flash-attn`, `flashinfer-python`, `torch`, `transformers`. Please make sure these core packages does not contradict with each other. For the rest dependency, install them when needed. 

You can run evaluate script by `python -m eval.test_aime` for example. 

Evaluating inference engine is an important issue and still in development.

You should be able to develop on any version with its name listed under src/artifacts and src/services. The name suggest the version directly. In future, there may be steady release of some artifacts and services, which will be named without a version descriptor. 

