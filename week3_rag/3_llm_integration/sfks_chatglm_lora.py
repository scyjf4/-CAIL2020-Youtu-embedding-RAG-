# -*- coding: utf-8 -*-
"""
使用 ChatGLM-6B + LoRA 微调司法考试问答
采用生成式方法，让模型直接输出答案选项

注意：ChatGLM-6B 需要约 13GB 显存，8GB 显存需要使用 4-bit 量化
"""
import json
import torch
import numpy as np
import random
import sys
from pathlib import Path
from typing import List, Dict

# ====== 配置 ======
SAMPLE_PER_FILE = 2000      # 采样数量
EPOCHS = 3                  # 生成模型不需要太多轮次
LEARNING_RATE = 1e-4        # LoRA 学习率
BATCH_SIZE = 1              # 小 batch，节省显存
GRAD_ACCUM = 8              # 梯度累积
MAX_SEQ_LENGTH = 512
USE_4BIT = True             # 使用 4-bit 量化以节省显存

# 模型选择
MODEL_NAME = "THUDM/chatglm3-6b"  # 或 "THUDM/chatglm-6b", "THUDM/chatglm2-6b"

print("\n" + "=" * 60)
print("ChatGLM-6B + LoRA 司法考试问答训练")
print(f"  模型: {MODEL_NAME}")
print(f"  4-bit 量化: {USE_4BIT}")
print(f"  采样数量: 每文件 {SAMPLE_PER_FILE} 条")
print("=" * 60)

# 检查依赖
try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
    )
    from datasets import Dataset
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
    from sklearn.metrics import accuracy_score
    print("  ✓ 依赖导入成功")
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    print("请运行: pip install transformers peft bitsandbytes accelerate")
    sys.exit(1)

# ====== 路径配置 ======
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # my_intern/
DATA_DIR = PROJECT_ROOT / "data" / "sfks"
OUTPUT_DIR = PROJECT_ROOT / "week3_rag" / "3_llm_integration" / "sfks_chatglm_lora_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_sfks_data(data_dir: Path, sample_per_file: int = 2000) -> List[Dict]:
    """加载司法考试数据"""
    all_data = []

    for filename in ["0_train.json", "1_train.json"]:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  ⚠ 文件不存在: {filepath}")
            continue

        print(f"  - 加载 {filename}...")
        file_data = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    answer = item.get("answer", [])

                    # 只用单选题
                    if not (isinstance(answer, list) and len(answer) == 1):
                        continue

                    options = item.get("option_list", {})
                    if not isinstance(options, dict) or len(options) != 4:
                        continue

                    statement = item.get("statement", "")
                    if not statement:
                        continue

                    file_data.append({
                        "question": statement,
                        "options": options,
                        "answer": answer[0],
                    })
                except json.JSONDecodeError:
                    continue

        # 采样
        if len(file_data) > sample_per_file:
            sampled = random.sample(file_data, sample_per_file)
            print(f"    从 {len(file_data)} 条中采样 {sample_per_file} 条")
        else:
            sampled = file_data
            print(f"    全部使用 {len(file_data)} 条")

        all_data.extend(sampled)

    random.shuffle(all_data)
    print(f"\n  ✓ 单选题总数: {len(all_data)} 条")

    return all_data


def format_prompt(item: Dict) -> str:
    """格式化为 ChatGLM 的 prompt"""
    question = item["question"]
    options = item["options"]

    prompt = f"""请回答以下司法考试选择题，只需回答选项字母（A、B、C或D）。

题目：{question}

选项：
A. {options.get('A', '')}
B. {options.get('B', '')}
C. {options.get('C', '')}
D. {options.get('D', '')}

答案："""

    return prompt


def create_dataset(data_list: List[Dict], tokenizer) -> Dataset:
    """创建训练数据集"""
    processed = []

    for item in data_list:
        prompt = format_prompt(item)
        answer = item["answer"]  # A, B, C, D

        # 完整的输入输出
        full_text = prompt + answer

        # tokenize
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors=None,
        )

        # 创建 labels（只在答案部分计算 loss）
        prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]
        labels = [-100] * len(prompt_ids) + tokenized["input_ids"][len(prompt_ids):]
        labels = labels[:MAX_SEQ_LENGTH]
        # padding 部分也设为 -100
        labels = labels + [-100] * (MAX_SEQ_LENGTH - len(labels))

        processed.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "answer": answer,
        })

    return Dataset.from_list(processed)


def main():
    print(f"\n[1/5] 加载模型: {MODEL_NAME}")

    # 量化配置
    if USE_4BIT:
        print("  使用 4-bit 量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quantization_config = None

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 加载模型
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
    )

    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    # 配置 LoRA
    print(f"\n[2/5] 配置 LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["query_key_value"],  # ChatGLM 的注意力层
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据
    print(f"\n[3/5] 加载数据...")
    data_list = load_sfks_data(DATA_DIR, sample_per_file=SAMPLE_PER_FILE)

    # 划分训练/验证集
    split_idx = int(len(data_list) * 0.9)
    train_data = data_list[:split_idx]
    eval_data = data_list[split_idx:]

    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(eval_data)} 条")

    train_dataset = create_dataset(train_data, tokenizer)
    eval_dataset = create_dataset(eval_data, tokenizer)

    # 训练
    print(f"\n[4/5] 开始训练...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,

        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,

        gradient_accumulation_steps=GRAD_ACCUM,
        fp16=True,
        seed=42,

        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 保存
    print(f"\n[5/5] 保存模型...")
    model.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # 评估
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    # 生成式评估
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in eval_data[:100]:  # 评估前 100 条
            prompt = format_prompt(item)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            predicted = response.strip().upper()[:1]  # 取第一个字符

            if predicted == item["answer"]:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"  评估准确率: {accuracy:.4f} ({correct}/{total})")

    # 保存结果
    results = {
        "model": MODEL_NAME,
        "use_4bit": USE_4BIT,
        "sample_per_file": SAMPLE_PER_FILE,
        "epochs": EPOCHS,
        "accuracy": accuracy,
    }
    with open(OUTPUT_DIR / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 模型保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
