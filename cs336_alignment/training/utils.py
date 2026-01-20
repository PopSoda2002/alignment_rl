import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from typing import Callable, Literal

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).
    """
    max_prompt_and_output_len = 0
    input_ids = []
    labels = []
    response_mask = []
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt_str)
        output_ids = tokenizer.encode(output_str)
        max_prompt_and_output_len = max(max_prompt_and_output_len, len(prompt_ids) + len(output_ids) - 1)
        single_input_ids = prompt_ids + output_ids + [tokenizer.eos_token_id]
        input_ids.append(single_input_ids[:-1])
        labels.append(single_input_ids[1:])
        single_response_mask = [False]*(len(prompt_ids)-1) + [True]*(len(single_input_ids)-len(prompt_ids) - 1)
        response_mask.append(single_response_mask)
    def pad_vector(vector: torch.Tensor, max_len: int, value: int) -> torch.Tensor:
        return torch.nn.functional.pad(vector, (0, max_len - vector.shape[0]), value=value)
    # Pad input_ids, labels, and response_mask to the same length
    input_ids = torch.stack([pad_vector(torch.tensor(input_id), max_prompt_and_output_len, tokenizer.pad_token_id) for input_id in input_ids], dim=0)
    labels = torch.stack([pad_vector(torch.tensor(label), max_prompt_and_output_len, tokenizer.pad_token_id) for label in labels], dim=0)
    response_mask = torch.stack([pad_vector(torch.tensor(mask, dtype=torch.bool), max_prompt_and_output_len, False) for mask in response_mask], dim=0)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

@torch.inference_mode()
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of the logits."""
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs), dim=-1)

@torch.inference_mode()
def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False) -> dict[str, torch.Tensor]:
    """Get the log-probs of the response given the prompt."""
    model = model.to(input_ids.device)
    model.eval()
    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    response_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {
            "log_probs": response_log_probs,
            "token_entropy": token_entropy,
        }
    return {"log_probs": response_log_probs, "token_entropy": None}

def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None, normalize_constant: float = 1.0) -> torch.Tensor:
    """Normalize the tensor along a dimension, considering only the elements with mask value 1."""
    return torch.sum(tensor * mask, dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss for a microbatch of data."""
    # Batch size * gradient_accumulation_steps * normalize_constant
    norm = gradient_accumulation_steps * normalize_constant * policy_log_probs.shape[0]
    loss = -(policy_log_probs * response_mask).sum() / norm
    loss.backward()
    return loss, {"loss": loss}

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]], rollout_responses: list[str], repeated_ground_truths: list[str], group_size: int, advantage_eps: float, normalize_by_std: bool):
    advantages = []
    raw_rewards = []
    metadata = {}
    for i in range(0, len(rollout_responses), group_size):
        group_rollout_responses = rollout_responses[i:i+group_size]
        group_repeated_ground_truths = repeated_ground_truths[i:i+group_size]
        group_rewards = []
        group_advantages = []
        for response, ground_truth in zip(group_rollout_responses, group_repeated_ground_truths):
            reward = reward_fn(response, ground_truth)["reward"]
            group_rewards.append(reward)
        raw_rewards.extend(group_rewards)
        group_rewards = torch.tensor(group_rewards)
        group_rewards_mean = group_rewards.mean()
        if normalize_by_std:
            group_rewards_std = group_rewards.std()
            group_advantages = (group_rewards - group_rewards_mean) / (group_rewards_std + advantage_eps)
        else:
            group_advantages = (group_rewards - group_rewards_mean)
        advantages.extend(group_advantages.tolist())
    metadata["raw_rewards_mean"] = torch.tensor(raw_rewards).mean()
    metadata["raw_rewards_std"] = torch.tensor(raw_rewards).std()
    return torch.tensor(advantages), torch.tensor(raw_rewards), metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the naive policy gradient loss."""
    loss = -(raw_rewards_or_advantages * policy_log_probs)
    return loss

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss."""
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clip_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    clip_loss = -torch.min(ratio * advantages, clip_ratio * advantages)
    return clip_loss, {"clip_loss": clip_loss}

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss."""
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    assert advantages is not None
    if loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    assert old_log_probs is not None
    assert cliprange is not None
    if loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension, considering only the elements with mask value 1."""
    value = tensor * mask
    if dim is None:
        return value.sum() / mask.sum()
    return value.sum(dim=dim) / mask.sum(dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch."""
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    loss.backward()
    return loss, metadata