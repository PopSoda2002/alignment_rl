import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional

class GSM8KDataset(Dataset):
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): Path to the gsm8k jsonl file (e.g., 'data/gsm8k/train.jsonl')
        """
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Returns:
            dict: {'question': str, 'answer': str}
        """
        return self.data[idx]

def get_gsm8k_dataloader(
    data_path: str, 
    batch_size: int = 4, 
    shuffle: bool = False, 
    num_workers: int = 0
) -> DataLoader:
    """
    Creates a DataLoader for the GSM8K dataset.
    
    Args:
        data_path (str): Path to the jsonl file.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes.
        
    Returns:
        DataLoader: The PyTorch DataLoader.
    """
    dataset = GSM8KDataset(data_path)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=None # Default collate works fine for list of dicts with strings
    )

if __name__ == "__main__":
    # Test the dataloader
    train_path = "data/gsm8k/train.jsonl"
    try:
        loader = get_gsm8k_dataloader(train_path, batch_size=2, shuffle=True)
        
        print(f"Total batches: {len(loader)}")
        for batch in loader:
            print("Batch keys:", batch.keys())
            print("First question in batch:", batch['question'][0])
            print("First answer in batch:", batch['answer'][0])
            break
    except FileNotFoundError:
        print(f"File not found: {train_path}. Please check the path.")

