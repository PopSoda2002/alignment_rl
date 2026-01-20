import re
import json
from typing import List, Optional, Callable

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from transformers import PreTrainedModel
import torch
from unittest.mock import patch

def get_r1_prompts(requests: List[str]) -> List[str]:
    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        r1_zero_template = f.read()
    return [r1_zero_template.format(question=request) for request in requests]

def extract_final_number(text: str) -> Optional[int]:
    numbers = re.findall(r"-?\d+\.?\d*", text)
    answer: Optional[int] = None
    try:
        answer = int(float(numbers[-1]))
    except Exception:
        pass
    return answer

def reward_fn(response: str, ground_truth: str) -> float:
    response_number = extract_final_number(response)
    ground_truth_number = extract_final_number(ground_truth)
    if response_number is None or ground_truth_number is None:
        return 0.0
    return 1.0 if response_number == ground_truth_number else 0.0

def evaluate_vllm(vllm_model: LLM, reward_fn: Callable[[str, str], float], requests: List[str], 
ground_truths: List[str], eval_sampling_params: SamplingParams) -> float:
    prompts = get_r1_prompts(requests)
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    reward = 0
    for i, (output, ground_truth) in enumerate(zip(outputs, ground_truths)):
        generated_text = output.outputs[0].text
        reward += reward_fn(generated_text, ground_truth)
        # if i % 200 == 0:
        #     print(f"request {i}: {prompts[i]=}, {generated_text=}, {ground_truth=}, {reward_fn(generated_text, ground_truth)=}")
    return round(float(reward / len(requests)), 4)

def evaluate_vllm_on_gsm8k(vllm_model: LLM) -> float:
    test_file_path = "data/gsm8k/test.jsonl"
    with open(test_file_path) as f:
        test_data = [json.loads(line) for line in f]
    requests = [example["question"] for example in test_data]
    ground_truths = [example["answer"] for example in test_data]
    eval_sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=2048, stop=["</answer>"], include_stop_str_in_output=True)
    reward = evaluate_vllm(vllm_model, reward_fn, requests, ground_truths, eval_sampling_params)
    return reward

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    13
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# # Setup wandb metrics
# wandb.define_metric("train_step") wandb.define_metric("eval_step") # the x‑axis for training
# # the x‑axis for evaluation
# # everything that starts with train/ is tied to train_step
# wandb.define_metric("train/*", step_metric="train_step")
# # everything that starts with eval/ is tied to eval_step
# wandb.define_metric("eval/*", step_metric="eval_step")

if __name__ == "__main__":
    test_file_path = "data/gsm8k/test.jsonl"
    with open(test_file_path) as f:
        test_data = [json.loads(line) for line in f]
    requests = [example["question"] for example in test_data]
    ground_truths = [example["answer"] for example in test_data]
    vllm_model = LLM(model="/root/models/Qwen2.5-Math-1.5B")
    eval_sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=2048, stop=["</answer>"], include_stop_str_in_output=True)
    reward = evaluate_vllm(vllm_model, reward_fn, requests, ground_truths, eval_sampling_params)
    print(f"Reward: {reward}")