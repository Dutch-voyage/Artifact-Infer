import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import datasets
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import json


class DatasetWithTemplate(Dataset):
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


class HFEfficiencyEvaluator:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16,
        ).to(self.device)
        print("Model loaded successfully!")

    def measure_prefill_time(self, input_ids: torch.Tensor) -> float:
        """Measure time for prefill phase (processing input tokens)"""
        input_ids = input_ids.to(self.device)

        # Warm up
        with torch.no_grad():
            _ = self.model(input_ids.unsqueeze(0))

        # Measure prefill time
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(input_ids.unsqueeze(0))

        torch.cuda.synchronize()
        end_time = time.time()

        return end_time - start_time

    def measure_generation_time(self, input_ids: torch.Tensor, max_new_tokens: int = 512) -> Tuple[float, int]:
        """Measure time for generation phase and count generated tokens"""
        input_ids = input_ids.to(self.device)

        # Warm up with a small generation
        with torch.no_grad():
            _ = self.model.generate(
                input_ids.unsqueeze(0),
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Measure generation time
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids.unsqueeze(0),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        torch.cuda.synchronize()
        end_time = time.time()

        generated_tokens = outputs.shape[1] - input_ids.shape[0]

        return end_time - start_time, generated_tokens

    def evaluate_sample(self, prompt: str, max_new_tokens: int = 512) -> Dict:
        """Evaluate a single sample and return performance metrics"""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        num_input_tokens = len(input_ids)

        print(f"Input tokens: {num_input_tokens}")

        # Measure prefill time
        prefill_time = self.measure_prefill_time(input_ids)
        prefill_tokens_per_sec = num_input_tokens / prefill_time

        # Measure generation time
        generation_time, generated_tokens = self.measure_generation_time(input_ids, max_new_tokens)
        decode_tokens_per_sec = generated_tokens / generation_time

        # Generate text for verification
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids.unsqueeze(0).to(self.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "input_tokens": num_input_tokens,
            "generated_tokens": generated_tokens,
            "prefill_time": prefill_time,
            "generation_time": generation_time,
            "prefill_tokens_per_sec": prefill_tokens_per_sec,
            "decode_tokens_per_sec": decode_tokens_per_sec,
            "generated_text": generated_text
        }

    def evaluate_dataset(self, dataset: Dataset, num_samples: int = 5, max_new_tokens: int = 512) -> Dict:
        """Evaluate multiple samples from a dataset"""
        results = []

        for i in range(min(num_samples, len(dataset))):
            print(f"\nEvaluating sample {i+1}/{num_samples}...")
            sample = dataset[i]
            prompt = sample["raw_prompt"]

            result = self.evaluate_sample(prompt, max_new_tokens)
            result["sample_id"] = i
            results.append(result)

            print(f"Prefill: {result['prefill_tokens_per_sec']:.2f} tokens/s")
            print(f"Decode: {result['decode_tokens_per_sec']:.2f} tokens/s")

        # Calculate averages
        avg_prefill = np.mean([r["prefill_tokens_per_sec"] for r in results])
        avg_decode = np.mean([r["decode_tokens_per_sec"] for r in results])
        avg_input_tokens = np.mean([r["input_tokens"] for r in results])
        avg_generated_tokens = np.mean([r["generated_tokens"] for r in results])

        return {
            "individual_results": results,
            "average_prefill_tokens_per_sec": avg_prefill,
            "average_decode_tokens_per_sec": avg_decode,
            "average_input_tokens": avg_input_tokens,
            "average_generated_tokens": avg_generated_tokens
        }


def main():
    # Configuration
    model_path = "/home/yyx/models/Qwen3-4B"
    local_dir = "/home/yyx/efficient_inference/xylo-infer/datasets"
    data_source = "aime_2024"
    num_samples = 1
    max_new_tokens = 512

    print("HuggingFace Transformers Efficiency Evaluation")
    print("=" * 50)

    # Initialize evaluator
    evaluator = HFEfficiencyEvaluator(model_path)

    # Load dataset
    print(f"\nLoading dataset {data_source}...")
    dataset = DatasetWithTemplate(local_dir, data_source, evaluator.tokenizer)
    print(f"Dataset loaded with {len(dataset)} samples")

    # Run evaluation
    print(f"\nEvaluating {num_samples} samples with max_new_tokens={max_new_tokens}...")
    results = evaluator.evaluate_dataset(dataset, num_samples, max_new_tokens)

    # Print summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Average Prefill Speed: {results['average_prefill_tokens_per_sec']:.2f} tokens/s")
    print(f"Average Decode Speed: {results['average_decode_tokens_per_sec']:.2f} tokens/s")
    print(f"Average Input Length: {results['average_input_tokens']:.1f} tokens")
    print(f"Average Generated Length: {results['average_generated_tokens']:.1f} tokens")

    # Save detailed results
    output_file = "hf_efficiency_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")

    # Save a sample generation
    sample_result = results["individual_results"][0]
    with open("hf_sample_generation.txt", "w") as f:
        f.write(sample_result["generated_text"])
    print("Sample generation saved to hf_sample_generation.txt")


if __name__ == "__main__":
    main()