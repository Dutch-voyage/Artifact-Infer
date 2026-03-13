import torch
import numpy as np
from src.services.nanovllm_v7 import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import datasets
from torch.utils.data import DataLoader, Dataset
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

        row_dict["raw_prompt"] = raw_prompt
        return row_dict

    def __len__(self):
        return len(self.dataframe)


def generate_answer(
    local_dir="./datasets",
    model_path="/home/yyx/models/Qwen3-4B",
    enforce_eager=True,
    tensor_parallel_size=1,
    if_log_compress=True,
    if_fake_compress=False,
    if_compress_kvcache=True,
    if_temperatured=False,
    lse_preserve_merge=False,
    compress_method="none",
    layer_budget=512,
    query_window_size=32,
    steps_between_cache_compressions=32,
    log_path="./test_logs",
    p_attn=0.90,
    attn_reduce_method="raw",
    temperature=-1,
    max_tokens=8192,
    data_source="aime24",
    sample_idx=5,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = Dataset_with_template(local_dir, data_source, tokenizer)
    torch.set_warn_always(False)
    
    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tensor_parallel_size,
        if_log_compress=if_log_compress,
        if_fake_compress=if_fake_compress,
        if_compress_kvcache=if_compress_kvcache,
        if_temperatured=if_temperatured,
        lse_preserve_merge=lse_preserve_merge,
        compress_method=compress_method,
        layer_budget=layer_budget,
        query_window_size=query_window_size,
        steps_between_cache_compressions=steps_between_cache_compressions,
        log_path=log_path,
        p_attn=p_attn,
        attn_reduce_method=attn_reduce_method,
    )
    
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    sample = dataset[sample_idx]
    prompt = sample["raw_prompt"]

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    outputs = llm.generate([prompt], sampling_params, )

    output_ids = outputs[0]["token_ids"]
    
    # log_steps=list(range(512 + 32 - len(input_ids), 2048 - 33, 32))
    # dump_record = []
    # for step in log_steps:
    #     dump_record.append({"token_ids": input_ids.tolist() + output_ids[: step], "output_ids": output_ids[step: step + 33], "logits": outputs[0]["logits"][len(input_ids) + step:len(input_ids) + step + 32]})
    # np.save(f"./no_compress_logs/logits_for_kl.npy", dump_record)
    
    print(f"total input tokens {len(input_ids)}")
    print(f"total output tokens {len(output_ids)}")
    # print(len(outputs[0]["logits"]))
    all_text = prompt + outputs[0]["text"]
    # generated_text = outputs[0]["text"]
    with open("aime_5_answer_test", "w") as f:
        f.write(all_text)


def main():
    parser = argparse.ArgumentParser(description="Test script for LLM inference with configurable parameters")

    # Model and data paths
    parser.add_argument("--local_dir", type=str, default="./datasets", help="Directory containing datasets")
    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument("--model_path", type=str, default="/home/yyx/models/Qwen3-4B", help="Path to the model")
    parser.add_argument("--data_source", type=str, default="aime24", help="Dataset source name")
    parser.add_argument("--sample_idx", type=int, default=5, help="Sample index to test")

    # LLM configuration
    parser.add_argument("--enforce_eager", type=str_to_bool, default=True, help="Enforce eager execution")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--if_log_compress", type=str_to_bool, default=True, help="Enable compression logging")
    parser.add_argument("--if_fake_compress", type=str_to_bool, default=True, help="Use fake compression for testing")
    parser.add_argument("--if_compress_kvcache", type=str_to_bool, default=True, help="Enable KV cache compression")
    parser.add_argument("--if_temperatured", type=str_to_bool, default=False, help="Use temperatured attention")
    parser.add_argument("--lse_preserve_merge", type=str_to_bool, default=False, help="Use LSE preserve merge method")
    parser.add_argument("--compress_method", type=str, default="none", help="Compression method")
    parser.add_argument("--layer_budget", type=int, default=512, help="Layer budget for compression")
    parser.add_argument("--query_window_size", type=int, default=32, help="Query window size")
    parser.add_argument("--steps_between_cache_compressions", type=int, default=32, help="Steps between cache compressions")
    parser.add_argument("--log_path", type=str, default="./test_test_logs", help="Path for log files")
    parser.add_argument("--p_attn", type=float, default=0.90, help="Attention percentile")
    parser.add_argument("--attn_reduce_method", type=str, default="raw", help="Attention reduction method")

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=-1, help="Sampling temperature (-1 for greedy)")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum number of output tokens")

    args = parser.parse_args()

    generate_answer(
        local_dir=args.local_dir,
        model_path=args.model_path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        if_log_compress=args.if_log_compress,
        if_fake_compress=args.if_fake_compress,
        if_temperatured=args.if_temperatured,
        if_compress_kvcache=args.if_compress_kvcache,
        lse_preserve_merge=args.lse_preserve_merge,
        compress_method=args.compress_method,
        layer_budget=args.layer_budget,
        query_window_size=args.query_window_size,
        steps_between_cache_compressions=args.steps_between_cache_compressions,
        log_path=args.log_path,
        p_attn=args.p_attn,
        attn_reduce_method=args.attn_reduce_method,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        data_source=args.data_source,
        sample_idx=args.sample_idx,
    )


if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    main()
