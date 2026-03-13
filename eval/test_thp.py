import torch
import numpy as np
from src.services.nanovllm_v8 import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import datasets
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv
load_dotenv() 
from .utils import evaluate

MODEL_PATH = os.getenv("MODEL_PATH")

torch.set_printoptions(profile="full")

temperature = 0.6

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


def generate_answer(local_dir="datasets", model_path=f"{MODEL_PATH}/Qwen3-4B"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, 
              enforce_eager=False, 
              tensor_parallel_size=1, 
              if_compress_kvcache=True,
              if_fake_compress=False,
              lse_preserve_merge=False,
              compress_method="vanilla_topp",
              layer_budget=256,
              layer_upper_budget=512, 
              query_window_size=32,
              p_attn=0.90,
              steps_between_cache_compressions=128, 
              )
    
    sampling_params = SamplingParams(temperature=0.6 ,top_k=20, top_p=0.95, max_tokens=2048)
    
    # temperature < 0 for greedy_sampling
    # sampling_params = SamplingParams(temperature=-1, max_tokens=1024)
    # model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

    dataset = Dataset_with_template(local_dir, "aime24", tokenizer)

    prompts = [dataset[i]["raw_prompt"] for i in range(30)]
    # prompts = [dataset[25]["raw_prompt"]]
    
    outputs = llm.generate(prompts, sampling_params)
    input_ids_0 = tokenizer(prompts[0], return_tensors="pt").input_ids[0]
    output_ids_0 = tokenizer(outputs[0]["text"], return_tensors="pt").input_ids[0]
    input_ids_1 = tokenizer(prompts[-1], return_tensors="pt").input_ids[0]
    output_ids_1 = tokenizer(outputs[-1]["text"], return_tensors="pt").input_ids[0]
    
    print(f"req {0}:\n")
    print(f"total input tokens {len(input_ids_0)}")
    print(f"total output tokens {len(output_ids_0)}")
    
    print(f"req {-1}:\n")
    print(f"total input tokens {len(input_ids_1)}")
    print(f"total output tokens {len(output_ids_1)}")
    all_text = prompts[0] + outputs[0]["text"]  +"\n" + prompts[-1] + outputs[-1]["text"]
    # generated_text = outputs[0]["text"]
    with open("aime_5_answer_test", "w") as f:
        f.write(all_text)


def main():
    generate_answer()

if __name__ == "__main__":
    main()
