from src.services.nanovllm_v6 import LLM, SamplingParams

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from rich import print
from rich.console import Console

from transformers import AutoTokenizer

import os

def test_kl_by_sample(model_path="/home/yyx/models/Qwen3-4B"):
    logits_per_sample = np.load("./no_compress_logs/logits_for_kl.npy", allow_pickle=True)

    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        if_log_lse=False,
        if_compress_kvcache=True,
        compress_method="oMerge_filter",
        layer_budget=512,
        query_window_size=32,
        steps_between_cache_compressions=32,
        log_path="./oMerge_filter_logs",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # sampling_params = SamplingParams(temperature=0.6 ,top_k=20, top_p=0.95, max_tokens=8192)
    sampling_params = SamplingParams(temperature=-1, max_tokens=32)
    kls = []
    steps = []
    ifmatch = []
    console = Console()
    for sample in logits_per_sample:
        prompt_ids = sample["token_ids"]
        print(len(prompt_ids))
        outputs = llm.generate([prompt_ids], sampling_params, log_steps=[len(prompt_ids)])
        gt_logits = sample["logits"][:-1]
        gt_output_ids = sample["output_ids"]
        gen_logits = outputs[0]["logits"][-31:] # get the logits for the generated tokens
        
        prefix = prompt_ids[-50:] + outputs[0]["token_ids"][:-32]
        prefix_text = tokenizer.decode(prefix)
        original_token = tokenizer.decode(gt_output_ids)
        compressed_token = tokenizer.decode(outputs[0]["token_ids"][-32:])
        
        console.print(f"Uncompressed: {prefix_text}[bold blue]{original_token}[/bold blue] ")
        console.print(f"Compressed: {prefix_text}[bold red]{compressed_token}[/bold red] ")
        
        match_num = 0
        for gt_id, gen_id in zip(gt_output_ids, outputs[0]["token_ids"][-32:]):
            if gt_id == gen_id:
                match_num += 1
        ifmatch.append(match_num / 32)

        gt_logprobs = F.log_softmax(torch.tensor(gt_logits, device="cuda"), dim=-1)
        gen_logprobs = F.log_softmax(torch.tensor(gen_logits, device="cuda"), dim=-1)
        kl = F.kl_div(gen_logprobs, gt_logprobs, log_target=True)
        print(f"Step {len(prompt_ids)}: KL Divergence = {kl.item()}")
        kls.append(kl.cpu().numpy())
        steps.append(len(prompt_ids))

    path = "oMerge_filter_logs"
    figpath = "figs/kld"
    os.makedirs(path, exist_ok=True)
    os.makedirs(figpath, exist_ok=True)
    np.save(f"{path}/kl.npy", {"steps": steps, "kls": kls, "if_match": ifmatch})

    plt.plot(steps, np.log10(kls), label="oMerge_filter")
    plt.xlabel("Steps")
    plt.ylabel("log KL Divergence")
    plt.title("KL Divergence over Steps")
    plt.legend()
    
    plt.savefig(f"{figpath}/oMerge_filter_kl.png")

if __name__ == "__main__":
    test_kl_by_sample()
