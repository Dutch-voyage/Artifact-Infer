import torch
import numpy as np
from src.services.nanovllm_v6 import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import datasets
from torch.utils.data import DataLoader, Dataset

from .utils import evaluate


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

        row_dict["raw_prompt"] = raw_prompt
        return row_dict

    def __len__(self):
        return len(self.dataframe)


def generate_answer(local_dir="datasets", model_path="/home/yyx/models/Qwen3-4B"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6 ,top_k=20, top_p=0.95, max_tokens=32768)
    dataset = Dataset_with_template(local_dir, "aime24", tokenizer)
    
    batch_size = 5
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
                "score": score,
                "number_generated_tokens": generate_length,
                "ans": ans, 
                "ans_text": outputs[idx]["text"], 
                "generated_tokens": output_ids, 
            }

            all_text = prompts[idx] + outputs[idx]["text"] + "\n\n" + "score: " + str(score) + "\n\n" + "generated_tokens: " + str(generate_length) + "\n\n" + "=" * 100 + "\n\n"
            # generated_text = outputs[0]["text"]
            with open("aime_baseline", "a") as f:
                f.write(all_text)
        
    evaluate_result["summary"] = {
        "total_score": total_scores,
        "total_generated_tokens": total_generate_lengths,
        "average_score": total_scores / len(dataset),
        "average_generated_tokens": total_generate_lengths / len(dataset)
    }
        
    summary = "Evaluation completed." + "\n\n"
    summary += "scores sum: " + str(total_scores) + "\n\n"
    summary += f"Average score: {total_scores / batch_size}" + "\n\n"
    summary += "Total generated tokens: " + str(total_generate_lengths) + "\n\n"
    summary += f"Average generated tokens: {total_generate_lengths / (batch_size * len(dataset))}" + "\n\n"
    with open("aime_baseline", "a") as f:
        f.write(summary)
    
    print("Evaluation completed.")
    print("scores sum:", total_scores)
    print(f"Average score: {total_scores / (batch_size * len(dataset))}")
    print("Total generated tokens:", total_generate_lengths)
    print(f"Average generated tokens: {total_generate_lengths / (batch_size * len(dataset))}")
    
    

def main():
    generate_answer()


if __name__ == "__main__":
    main()
