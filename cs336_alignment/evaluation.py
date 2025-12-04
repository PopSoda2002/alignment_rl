from vllm import LLM, SamplingParams
from typing import Callable, List
import json
import re

# copy from verl/utils/reward_score/gsm8k.py
def parse_gsm8k_response(response: str) -> str:
    answer = re.findall("(\\-?[0-9\\.\\,]+)", response)
    if len(answer) == 0:
        # no reward is there is no answer
        return None
    final_answer = None
    invalid_str = ["", "."]
    for final_answer in reversed(answer):
        if final_answer not in invalid_str:
            break
    return final_answer

def math_reward_fn(response: str, ground_truth: str) -> float:
    answer = parse_gsm8k_response(response)
    ground_truth = parse_gsm8k_response(ground_truth)
    if answer is not None and ground_truth is not None and answer == ground_truth:
        return 1.0
    return 0.0


def evaluate_vllm(vllm_model : LLM, reward_fn: Callable[[str, str], float], prompts: List[str], ground_truths: List[str], eval_sampling_params: SamplingParams) -> float:
    prefix = "Please answer the question step by step. and take the final answer in the format of <answer>...</answer>."
    prompts = [prefix + prompt for prompt in prompts]
    responses = vllm_model.generate(prompts, eval_sampling_params)
    correct_count = 0
    for i, response in enumerate(responses):
        if i % 100 == 0:
            print(f"response: {response.outputs[0].text}")
            print(f"ground_truth: {ground_truths[i]}")
        reward = reward_fn(response.outputs[0].text, ground_truths[i])
        if reward == 1.0:
            correct_count += 1
    return correct_count / len(ground_truths)

if __name__ == "__main__":
    # load the test data from data/gsm8k/test.jsonl
    with open("data/gsm8k/test.jsonl", "r") as f:
        test_data = [json.loads(line) for line in f]
    print(f"Loaded {len(test_data)} test examples")
    prompts = [example["question"] for example in test_data]
    ground_truths = [example["answer"] for example in test_data]
    vllm_model = LLM(model="/root/Qwen2.5-Math-1.5B")
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=4096, stop=["</answer>"], include_stop_str_in_output=True)
    accuracy = evaluate_vllm(vllm_model, math_reward_fn, prompts, ground_truths, sampling_params)
    print(f"Accuracy: {accuracy}")
    print(f"Evaluated {len(prompts)} prompts")