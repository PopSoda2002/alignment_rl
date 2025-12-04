from vllm import LLM, SamplingParams

prompts = [
    "Hello, how are you?",
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "What is the capital of Portugal?",
    "What is the capital of Greece?",
    "What is the capital of Turkey?",
    "What is the capital of Egypt?",
    "What is the capital of South Africa?",
]

sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["\n"]
)

llm = LLM(
    model="/root/Qwen2.5-Math-1.5B"
)

responses = llm.generate(prompts, sampling_params)

for output in responses:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"prompt: {prompt!r}, generated text: {generated_text!r}")