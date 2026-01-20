from __future__ import annotations

import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

class GSM8KDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        # load the dataset from the jsonl file
        with open(dataset_path, "r") as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

def gsm8k_collate_fn(batch, max_length: int = 1024):
    with open("cs336_alignment/prompts/alpaca_sft.prompt") as f:
        alpaca_sft_template = f.read()
    prompts = []
    responses = []
    for row in batch:
        question = row["question"]
        answer = row["answer"]
        prompt = alpaca_sft_template.format(instruction=question, response="")
        prompts.append(prompt)
        responses.append(answer)
    return prompts, responses

def get_gsm8k_data_loader(dataset_path: str, batch_size: int = 16, shuffle: bool = True, max_length: int = 1024):
    dataset = GSM8KDataset(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: gsm8k_collate_fn(batch, max_length=max_length))

if __name__ == "__main__":
    file_path = "data/gsm8k/train.jsonl"
    data_loader = get_gsm8k_data_loader(file_path, batch_size=512, shuffle=True)
    max_prompt_length, max_answer_length = 0, 0
    for batch in data_loader:
        print(len(batch))