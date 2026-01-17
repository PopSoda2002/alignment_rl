import re
import json
from typing import List, Optional, Callable

from vllm import LLM, SamplingParams

def get_r1_prompts(requests: List[str]) -> List[str]:
    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        r1_zero_template = f.read()
    return [r1_zero_template.format(question=request) for request in requests]

def extract_final_number(text: str) -> Optional[str]:
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if not numbers:
        return None
    return int(float(numbers[-1]))

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
        if i % 200 == 0:
            print(f"request {i}: {prompts[i]=}, {generated_text=}, {ground_truth=}, {reward_fn(generated_text, ground_truth)=}")
    return round(float(reward / len(requests)), 4)

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