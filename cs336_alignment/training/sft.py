from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from data_loader import get_gsm8k_data_loader
import torch.nn.functional as F
import os
from tqdm import tqdm
class SFTTrainer:
    def __init__(self, model, tokenizer, data_loader, sft_steps, learning_rate, weight_decay, device):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.sft_steps = sft_steps
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.device = device
        self.model = self.model.to(self.device)

    def train(self):
        print("SFT training started!")
        self.model.train()
        grad_accumulation_steps = 5
        for step in range(self.sft_steps):
            for i, data in tqdm(enumerate(self.data_loader)):
                sequences = data["sequences"].to(self.device)
                loss_mask = data["loss_mask"].to(self.device)
                inputs = sequences[:, :-1]
                labels = sequences[:, 1:]
                self.optimizer.zero_grad()
                outputs = self.model(inputs).logits
                loss = F.cross_entropy(
                    outputs.reshape(-1, outputs.size(-1)),
                    labels.reshape(-1), 
                    reduction="none"
                )
                mask = loss_mask
                if mask.shape == sequences.shape:
                    mask = mask[:, 1:]
                mask = mask.reshape(-1).to(loss.dtype)
                loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
                loss.backward()
                if (i + 1) % grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    print(f"Step {step} loss: {loss.item()}")
        output_dir = "checkpoints/sft/Qwen2.5-Math-1.5B"
        print(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"SFT training completed! Model saved to {output_dir}")

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("/root/models/Qwen2.5-Math-1.5B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained("/root/models/Qwen2.5-Math-1.5B")
    data_loader = get_gsm8k_data_loader("data/gsm8k/train.jsonl", batch_size=16, shuffle=True, max_length=1024)
    sft_trainer = SFTTrainer(model, tokenizer, data_loader, sft_steps=1, learning_rate=1e-5, weight_decay=1e-5, device="cuda")
    sft_trainer.train()