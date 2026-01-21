import wandb
from tqdm import tqdm
import torch

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.evaluation.eval import evaluate_vllm_on_gsm8k, gsm8k_reward_fn, load_policy_into_vllm_instance, init_vllm
from cs336_alignment.training.utils import compute_policy_gradient_loss, tokenize_prompt_and_output, compute_group_normalized_rewards, grpo_microbatch_train_step, get_response_log_probs
from cs336_alignment.training.data_loader import get_gsm8k_data_loader

class GRPOTrainer:
    def __init__(self, model, tokenizer, data_loader, train_device, rollout_device, group_size, microbatch_size, grad_accumulation_steps, rollout_batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.train_device = train_device
        self.rollout_device = rollout_device
        self.group_size = group_size
        self.microbatch_size = microbatch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.rollout_batch_size = rollout_batch_size
        self.model = self.model.to(self.train_device)
        self.grpo_steps = 1
        self.vllm_model = init_vllm(model_id="/root/models/Qwen2.5-Math-1.5B", device=self.rollout_device, seed=42)

    def train(self):
        print("GRPO training started!")
        wandb.init(project="grpo", name="grpo")
        group_microbatch_size = self.microbatch_size * self.group_size
        for step in range(self.grpo_steps):
            prompts, responses = next(iter(self.data_loader))
            load_policy_into_vllm_instance(self.model, self.vllm_model)
            repeated_prompts = []
            repeated_ground_truths = []
            self.model.eval()
            for prompt, ground_truth in zip(prompts, responses):
                repeated_prompts.extend([prompt] * self.group_size)
                repeated_ground_truths.extend([ground_truth] * self.group_size)
            rollout_responses = self.rollout(repeated_prompts)
            tokenized_prompt_and_output = tokenize_prompt_and_output(repeated_prompts, rollout_responses, self.tokenizer)
            input_ids = tokenized_prompt_and_output["input_ids"].to(self.train_device)
            labels = tokenized_prompt_and_output["labels"].to(self.train_device)
            response_mask = tokenized_prompt_and_output["response_mask"].to(self.train_device)
            # with torch.no_grad():
            #     old_log_probs = get_response_log_probs(self.model, input_ids, labels)["log_probs"]
            advantages, raw_rewards, metadata = compute_group_normalized_rewards(gsm8k_reward_fn, rollout_responses, repeated_ground_truths, self.group_size, 0.1, True)
            advantages = advantages.unsqueeze(-1).to(self.train_device)
            raw_rewards = raw_rewards.unsqueeze(-1).to(self.train_device)
            self.model.train()
            for i in range(0, len(input_ids), group_microbatch_size):
                policy_log_probs = get_response_log_probs(self.model, input_ids[i:i+group_microbatch_size], labels[i:i+group_microbatch_size])["log_probs"]
                loss, metadata = grpo_microbatch_train_step(policy_log_probs, response_mask[i:i+group_microbatch_size], grad_accumulation_steps, "reinforce_with_baseline", raw_rewards[i:i+group_microbatch_size], advantages[i:i+group_microbatch_size], old_log_probs=None, cliprange=None)
                if (i + 1) % group_microbatch_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        reward = evaluate_vllm_on_gsm8k(self.vllm_model)
        print(f"Step {step + 1} reward: {reward}")
        wandb.log({"reward": reward}, step=step + 1)
    
    def rollout(self, prompts: list[str]) -> list[str]:
        return [rollout_response.outputs[0].text for rollout_response in self.vllm_model.generate(prompts, SamplingParams(temperature=1.0, top_p=1.0, max_tokens=2048))]

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("/root/models/Qwen2.5-Math-1.5B")
    tokenizer = AutoTokenizer.from_pretrained("/root/models/Qwen2.5-Math-1.5B")
    rollout_batch_size = 64
    microbatch_size = 2
    group_size = 2
    grad_accumulation_steps = 32
    data_loader = get_gsm8k_data_loader("data/gsm8k/train.jsonl", batch_size=rollout_batch_size, shuffle=True)
    train_device = "cuda:0"
    rollout_device = "cuda:1"
    grpo_trainer = GRPOTrainer(model, tokenizer, data_loader, train_device, rollout_device, group_size, microbatch_size, grad_accumulation_steps, rollout_batch_size)
    grpo_trainer.train()