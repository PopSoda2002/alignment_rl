import torch
from transformers import PreTrainedTokenizerBase

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