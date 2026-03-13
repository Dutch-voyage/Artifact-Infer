import torch
import numpy as np
from src.services.nanovllm_v8 import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import datasets
from torch.utils.data import DataLoader, Dataset

from .utils import evaluate
import json
from pprint import pformat

import argparse


class Dataset_with_template(Dataset):
    def __init__(self, local_dir, data_source, tokenizer):
        self.dataframe = datasets.load_dataset(
            "parquet",
            data_files=os.path.join(local_dir, data_source + ".parquet"),
            split="train",
        )
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        row_dict = self.dataframe[idx]
        prompt = row_dict["prompt"]
        raw_prompt = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )
        # Filter out image-related columns that contain None
        row_dict = {k: v for k, v in row_dict.items() if k not in ["image", "has_image"]}
        row_dict["raw_prompt"] = raw_prompt
        return row_dict

    def __len__(self):
        return len(self.dataframe)

def generate_answer(
    local_dir="datasets",
    eval_path="new_eval_results",
    model_path="/home/yyx/models/Qwen3-4B",
    data_source="aime24",
    enforce_eager=False,
    compress_method="rkv",
    layer_budget=1024,
    layer_upper_budget=2048, 
    window_size=128,
    steps_between_cache_compressions=128,
    p_attn=0.9, 
    if_fake_compress=False, 
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        if_compress_kvcache=True,
        compress_method=compress_method,
        layer_budget=layer_budget + window_size,
        layer_upper_budget=layer_upper_budget + window_size, 
        window_size=window_size,
        steps_between_cache_compressions=steps_between_cache_compressions,
        p_attn=p_attn, 
        if_fake_compress=if_fake_compress,
    )
    sampling_params = SamplingParams(
        temperature=0.6, top_k=20, top_p=0.95, max_tokens=32768
    )
    dataset = Dataset_with_template(local_dir, data_source, tokenizer)

    batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    total_scores = 0.0
    total_generate_lengths = 0
    evaluate_result = {}

    for batch in dataloader:
        prompts = batch["raw_prompt"]
        ground_truth = batch["answer"]
        outputs = llm.generate(prompts, sampling_params)

        for idx in range(batch_size):
            output_ids = outputs[idx]["token_ids"]
            print(f"total output tokens {len(output_ids)}")

            score, ans = evaluate(outputs[idx]["text"], ground_truth[idx])
            total_scores += score

            generate_length = len(output_ids)
            total_generate_lengths += generate_length

            evaluate_result[idx] = {
                "score": int(score),
                "number_generated_tokens": int(generate_length),
                "ans": str(ans) if ans is not None else None,
                "ans_text": outputs[idx]["text"],
                "generated_tokens": [int(tok) for tok in output_ids],
            }

            all_text = (
                prompts[idx]
                + outputs[idx]["text"]
                + "\n\n"
                + "score: "
                + str(score)
                + "\n\n"
                + "generated_tokens: "
                + str(generate_length)
                + "\n\n"
                + "=" * 100
                + "\n\n"
            )
            # generated_text = outputs[0]["text"]
            with open(
                f"{eval_path}/{data_source}_compress_by_{compress_method}_layer_budget_{layer_budget}_window_size_{window_size}_steps_{steps_between_cache_compressions}.txt",
                "a",
            ) as f:
                f.write(all_text)

    evaluate_result["summary"] = {
        "compress_method": compress_method,
        "layer_budget": int(layer_budget),
        "window_size": int(window_size),
        "p_attn": float(p_attn), 
        "steps_between_cache_compressions": int(steps_between_cache_compressions),
        "total_score": float(total_scores),
        "total_generated_tokens": int(total_generate_lengths),
        "average_score": float(total_scores / len(dataset)),
        "average_generated_tokens": float(total_generate_lengths / len(dataset)),
    }

    # json.dump(evaluate_result, open(f"aime_{compress_method}_{layer_budget}_{window_size}_{steps_between_cache_compressions}.json", "w"))
    with open(
        f"{eval_path}/{data_source}_compress_by_{compress_method}_p_attn_{p_attn}_layer_budget_{layer_budget}_upper_{layer_upper_budget}_window_size_{window_size}_steps_{steps_between_cache_compressions}.json",
        "w",
    ) as f:
        json.dump(evaluate_result, f, indent=4)

    summary = "Evaluation completed." + "\n\n"
    summary += "scores sum: " + str(total_scores) + "\n\n"
    summary += f"Average score: {total_scores / len(dataset)}" + "\n\n"
    summary += "Total generated tokens: " + str(total_generate_lengths) + "\n\n"
    summary += (
        f"Average generated tokens: {total_generate_lengths / (len(dataset))}" + "\n\n"
    )
    with open(
        f"{eval_path}/{data_source}_{compress_method}_{layer_budget}_{layer_upper_budget}_{window_size}_{steps_between_cache_compressions}",
        "a",
    ) as f:
        f.write(summary)

    print("Evaluation completed.")
    print("scores sum:", total_scores)
    print(f"Average score: {total_scores / (len(dataset))}")
    print("Total generated tokens:", total_generate_lengths)
    print(f"Average generated tokens: {total_generate_lengths / (len(dataset))}")


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default="datasets")
    parser.add_argument("--eval_path", type=str, default="new_eval_results")
    parser.add_argument("--data_source", type=str, default="aime24")
    parser.add_argument("--model_path", type=str, default="/home/yyx/models/Qwen3-4B")
    parser.add_argument("--enforce_eager", type=str_to_bool, default=False)
    parser.add_argument(
        "--compress_method", type=str, default="vanilla_topp", choices=["rkv", "snapkv", "vanilla_topp"]
    )
    parser.add_argument("--layer_budget", type=int, default=1024)
    parser.add_argument("--layer_upper_budget", type=int, default=2048)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--steps_between_cache_compressions", type=int, default=128)
    
    parser.add_argument("--p_attn", type=float, default=0.9)
    
    parser.add_argument("--if_fake_compress", type=str_to_bool, default=False)
    args = parser.parse_args()

    generate_answer(
        local_dir=args.local_dir,
        eval_path=args.eval_path,
        data_source=args.data_source,
        model_path=args.model_path,
        enforce_eager=args.enforce_eager,
        compress_method=args.compress_method,
        layer_budget=args.layer_budget,
        layer_upper_budget=args.layer_budget * 2, 
        window_size=args.window_size,
        steps_between_cache_compressions=args.steps_between_cache_compressions,
        p_attn=args.p_attn, 
        if_fake_compress=args.if_fake_compress
    )


if __name__ == "__main__":
    main()
