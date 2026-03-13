import datasets
from datasets import load_dataset
import os

def add_template(example):
    example['prompt'] = [{"role": "user", "content": example['problem'] + "Please reason step by step, and put your final answer within \\boxed{}."}]
    return example

def load_aime_dataset(local_dir, data_source):
    print(os.path.join(local_dir, data_source + ".json"))
    dataframe = datasets.load_dataset(
        "json",
        data_files=os.path.join(local_dir, data_source + ".json"),
        split="train",
    )
    
    # for item in dataframe:
    #     item['prompt'] = item['problem'] + "Please reason step by step, and put your final answer within \\boxed{}."

    # print(dataframe[0]['prompt'])

    dataframe = dataframe.map(add_template, remove_columns=["problem"])
    print(dataframe[0]['prompt'])

    dataframe.to_parquet(os.path.join(local_dir, data_source + ".parquet"))

    return dataframe

if __name__ == "__main__":
    local_dir = "./datasets"
    data_source = "aime24"
    load_aime_dataset(local_dir, data_source)