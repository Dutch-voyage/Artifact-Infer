import torch
import numpy as np
from src.services.nanovllm_vl import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import datasets
from torch.utils.data import DataLoader, Dataset
import ast
import re

class Dataset_with_template(Dataset):
    def __init__(self, local_dir, data_source, tokenizer):
        self.dataframe = datasets.load_dataset(
            "json",
            data_files=os.path.join(local_dir, data_source + ".json"),
            split="train",
        )
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        row_dict = self.dataframe[idx]
        question = row_dict["question"]

        options = ast.literal_eval(row_dict["options"])

        for i, option in enumerate(options):
            question += f"\nOption {chr(65 + i)}: {option}"
        
        prompt = question + "\nPlease select the correct answer from the options above."
        
        prompt = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }, 
                {"type": "image", "image_url": row_dict["image_paths"][0]}
            ]
        }]
        
        raw_prompt = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )
        row_dict["raw_prompt"] = raw_prompt
        return row_dict

    def __len__(self):
        return len(self.dataframe)


def generate_answer(local_dir="./datasets", model_path="/home/yyx/models/Qwen2.5-VL-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_source = "mmmu_json/Accounting/"
    dataset = Dataset_with_template(local_dir, data_source + "validation", tokenizer)


    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    # model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

    sample = dataset[5]
    prompt = sample["raw_prompt"]
    image_data = [os.path.join(local_dir, data_source, image_path) for image_path in sample["image_paths"] if image_path is not None]

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    outputs = llm.generate([prompt], sampling_params, image_data=[image_data])

    output_ids = outputs[0]["token_ids"]
    print(f"total output tokens {len(output_ids)}")

    all_text = prompt + outputs[0]["text"]
    # generated_text = outputs[0]["text"]
    with open("test_mm", "w") as f:
        f.write(all_text)


def main():
    generate_answer()


if __name__ == "__main__":
    main()
