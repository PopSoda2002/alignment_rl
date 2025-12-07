import os
import torch
from typing import List, Dict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.nn.utils.rnn import pad_sequence

# 尝试导入数据加载器
try:
    from cs336_alignment.data_loader import GSM8KDataset
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from cs336_alignment.data_loader import GSM8KDataset

def train(model_path: str, dataset_path: str, batch_size: int, learning_rate: float, num_epochs: int):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Tokenizer & Model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.train()

    # 3. Load Prompt Template
    prompt_template_path = "cs336_alignment/prompts/alpaca_sft.prompt"
    if not os.path.exists(prompt_template_path):
        raise FileNotFoundError(f"Prompt file not found at {prompt_template_path}")
        
    with open(prompt_template_path, "r") as f:
        template_content = f.read()

    # 4. Prepare Dataset
    dataset = GSM8KDataset(dataset_path)
    
    # 5. Collate Function (Tokenization & Label Masking)
    def collate_fn(batch: List[Dict[str, str]]):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for item in batch:
            question = item['question']
            answer = item['answer']
            
            # 格式化完整文本
            full_text = template_content.format(instruction=question, response=answer) + tokenizer.eos_token
            
            # 格式化 Prompt 部分 (用于计算 Mask 长度)
            # 我们通过 split 找到 prompt 的部分
            prompt_part = template_content.split("{response}")[0].format(instruction=question)
            
            # Tokenize
            full_encoding = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
            prompt_encoding = tokenizer(prompt_part, return_tensors="pt", add_special_tokens=True)
            
            input_ids = full_encoding.input_ids[0]
            attention_mask = full_encoding.attention_mask[0]
            
            # 创建 Labels
            labels = input_ids.clone()
            
            # Mask 掉 Prompt 部分 (设为 -100)
            # 注意: 这里使用 prompt 编码后的长度作为切分点
            prompt_len = prompt_encoding.input_ids.shape[1]
            
            # 如果 tokenizer 自动添加了 BOS，prompt_len 和 full_encoding 都会包含它，直接用长度截断即可
            # 安全检查：确保 prompt_len 不超过总长度
            if prompt_len < len(labels):
                labels[:prompt_len] = -100
            else:
                # 异常情况，全 mask 掉
                labels[:] = -100
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        # Padding
        # 使用 pad_sequence 进行右填充 (batch_first=True)
        padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        
        return {
            "input_ids": padded_input_ids.to(device),
            "attention_mask": padded_attention_mask.to(device),
            "labels": padded_labels.to(device)
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 6. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        name="cosine", 
        optimizer=optimizer, 
        num_warmup_steps=int(0.1 * num_training_steps), 
        num_training_steps=num_training_steps
    )

    # 7. Training Loop
    print(f"Starting training for {num_epochs} epochs...")
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            global_step += 1
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.2e}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        save_dir = f"checkpoints/epoch_{epoch+1}"
        print(f"Saving checkpoint to {save_dir}...")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset_path", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()
    
    train(args.model_path, args.dataset_path, args.batch_size, args.learning_rate, args.num_epochs)
