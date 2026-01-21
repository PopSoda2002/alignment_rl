from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from data_loader import get_gsm8k_data_loader
import torch.nn.functional as F
import os
from tqdm import tqdm
from vllm import SamplingParams

import wandb

from utils import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step
from cs336_alignment.evaluation.eval import init_vllm, load_policy_into_vllm_instance, evaluate_vllm_on_gsm8k

class SFTTrainer:
    def __init__(self, model, tokenizer, data_loader, sft_steps, learning_rate, weight_decay, device):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.sft_steps = sft_steps
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.device = device
        self.model = self.model.to(self.device)
        self.vllm_model = init_vllm(model_id="/root/models/Qwen2.5-Math-1.5B", device="cuda:6", seed=42)

    def train(self):
        print("SFT training started!")
        self.model.train()
        grad_accumulation_steps = 5
        print(f"Step 0 reward: {evaluate_vllm_on_gsm8k(self.vllm_model)}")
        wandb.init(project="sft", name="sft")
        for step in range(self.sft_steps):
            for i, (prompts, responses) in tqdm(enumerate(self.data_loader)):
                tokenized_data = tokenize_prompt_and_output(prompts, responses, self.tokenizer)
                input_ids = tokenized_data["input_ids"].to(self.device)
                labels = tokenized_data["labels"].to(self.device)
                response_mask = tokenized_data["response_mask"].to(self.device)
                response_log_probs = get_response_log_probs(self.model, input_ids, labels, return_token_entropy=True)["log_probs"]
                loss, _ = sft_microbatch_train_step(response_log_probs, response_mask, grad_accumulation_steps)
                if (i + 1) % grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # print(f"Step {step} loss: {loss.item()}")
            load_policy_into_vllm_instance(self.model, self.vllm_model)
            reward = evaluate_vllm_on_gsm8k(self.vllm_model)
            print(f"Step {step + 1} reward: {reward}")
            wandb.log({"reward": reward}, step=step + 1)
            
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
    sft_trainer = SFTTrainer(model, tokenizer, data_loader, sft_steps=10, learning_rate=1e-5, weight_decay=1e-5, device="cuda:7")
    sft_trainer.train()