# Top-level eager imports were removed because they introduced an import cycle
# once LLMEngine started pulling in ModelRunner (which transitively imports MoeBackend,
# which lives under workshop.nanovllm_moe_orch.artifacts.moe_backend and itself reaches back
# into workshop.nanovllm_moe_orch.services.utils.context).
#
# Use the explicit paths instead:
#     from workshop.nanovllm_moe_orch.services.llm import LLM
#     from workshop.nanovllm_moe_orch.services.sampling_params import SamplingParams
