# -*- coding: utf-8 -*-
"""
V5 版本：不使用 DAP，直接用原始 BERT
对比测试 DAP vs 原始 BERT 的效果

验证发现 DAP 预训练效果不明显，尝试直接使用原始 BERT
"""
import json
import torch
import numpy as np
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, List, Dict

# ====== 配置 ======
SAMPLE_PER_FILE = 4000  # 每个文件采样数量
EPOCHS = 8              # 训练轮次
LEARNING_RATE = 2e-5    # 学习率
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 384

# 关键：不使用 DAP，直接用原始 BERT
USE_DAP = False
PRETRAINED_MODEL = "hfl/chinese-bert-wwm-ext"  # 直接用原始模型

print("\n" + "=" * 60)
print("V5 BERT + LoRA 训练 (不使用 DAP)")
print(f"  基础模型: {PRETRAINED_MODEL}")
print(f"  使用 DAP: {USE_DAP}")
print(f"  数据量: 每个文件 {SAMPLE_PER_FILE} 条")
print(f"  训练轮次: {EPOCHS}")
print("=" * 60)

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForMultipleChoice,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
    from datasets import Dataset
    from peft import get_peft_model, LoraConfig
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    print("  ✓ 依赖导入成功")
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    sys.exit(1)

# ====== 路径配置 ======
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "sfks"
OUTPUT_DIR = SCRIPT_DIR.parent / "models" / "sfks_bert_lora_v5_no_dap"

# LoRA 参数
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_sfks_data(data_dir: Path, sample_per_file: int = 4000) -> List[Dict]:
    """加载司法考试数据"""
    all_data = []
    multi_count = 0

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

                    if not (isinstance(answer, list) and len(answer) == 1):
                        multi_count += 1
                        continue

                    options = item.get("option_list", {})
                    if not isinstance(options, dict) or len(options) != 4:
                        continue

                    statement = item.get("statement", "")
                    if not statement:
                        continue

                    file_data.append({
                        "id": item.get("id", ""),
                        "question": statement,
                        "option_list": options,
                        "answer": answer[0],
                        "subject": item.get("subject", "未分类"),
                    })
                except json.JSONDecodeError:
                    continue

        if sample_per_file and len(file_data) > sample_per_file:
            sampled = random.sample(file_data, sample_per_file)
            print(f"    从 {len(file_data)} 条中采样 {sample_per_file} 条")
        else:
            sampled = file_data
            print(f"    全部使用 {len(file_data)} 条")

        all_data.extend(sampled)

    random.shuffle(all_data)

    print(f"\n  ✓ 单选题总数: {len(all_data)} 条")
    print(f"  ✓ 多选题(已跳过): {multi_count} 条")

    answer_dist = {"A": 0, "B": 0, "C": 0, "D": 0}
    for item in all_data:
        ans = item["answer"]
        if ans in answer_dist:
            answer_dist[ans] += 1
    print(f"  ✓ 答案分布: {answer_dist}")

    return all_data


def create_dataset(data_list: List[Dict]) -> Dataset:
    processed = []
    for item in data_list:
        processed.append({
            "id": item["id"],
            "question": item["question"],
            "option_list": json.dumps(item["option_list"], ensure_ascii=False),
            "answer": item["answer"],
            "subject": item["subject"],
        })
    return Dataset.from_list(processed)


def preprocess_function(examples, tokenizer):
    questions = examples["question"]
    options_json = examples["option_list"]
    answers = examples["answer"]

    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    encoded_labels = []
    first_sentences = []
    second_sentences = []

    for i in range(len(questions)):
        q_text = questions[i]
        opts = json.loads(options_json[i])
        ans = answers[i]

        encoded_labels.append(label_map.get(ans, 0))

        for option_key in ["A", "B", "C", "D"]:
            first_sentences.append(q_text)
            opt_text = opts.get(option_key, "[未知]")
            second_sentences.append(f"{option_key}. {opt_text}")

    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

    result = {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized.items()}
    result["label"] = encoded_labels

    return result


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None

    def __call__(self, features):
        labels = [feature.pop("label") for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened = []
        for feature in features:
            for i in range(num_choices):
                flattened.append({k: v[i] for k, v in feature.items()})

        batch = self.tokenizer.pad(
            flattened,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_weighted": float(f1),
        "f1_macro": float(f1_macro),
    }


def main():
    print(f"\n[1/6] 模型配置...")
    print(f"  ✓ 使用原始 BERT (不使用 DAP): {PRETRAINED_MODEL}")

    print(f"\n[2/6] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    print(f"\n[3/6] 加载数据...")
    data_list = load_sfks_data(DATA_DIR, sample_per_file=SAMPLE_PER_FILE)
    dataset = create_dataset(data_list)

    print(f"\n[4/6] 加载模型并配置 LoRA...")
    base_model = AutoModelForMultipleChoice.from_pretrained(PRETRAINED_MODEL)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["query", "key", "value", "dense"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    print(f"\n[5/6] 预处理数据...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  - 训练集: {len(split['train'])} 条")
    print(f"  - 验证集: {len(split['test'])} 条")

    print(f"\n[6/6] 开始训练...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        label_names=["labels"],

        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,

        optim="adamw_torch",
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(),
        seed=42,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    print(f"\n✓ 训练完成!")
    model.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\n" + "=" * 60)
    print("最终评估结果")
    print("=" * 60)
    eval_results = trainer.evaluate()

    print(f"  准确率 (Accuracy):    {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"  精确率 (Precision):   {eval_results.get('eval_precision', 0):.4f}")
    print(f"  召回率 (Recall):      {eval_results.get('eval_recall', 0):.4f}")
    print(f"  F1 分数 (weighted):   {eval_results.get('eval_f1_weighted', 0):.4f}")
    print(f"  F1 分数 (macro):      {eval_results.get('eval_f1_macro', 0):.4f}")

    results = {
        "config": {
            "base_model": PRETRAINED_MODEL,
            "use_dap": USE_DAP,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "sample_per_file": SAMPLE_PER_FILE,
            "total_samples": len(data_list),
            "train_samples": len(split["train"]),
            "eval_samples": len(split["test"]),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
        "metrics": {
            "accuracy": eval_results.get("eval_accuracy", 0),
            "precision": eval_results.get("eval_precision", 0),
            "recall": eval_results.get("eval_recall", 0),
            "f1_weighted": eval_results.get("eval_f1_weighted", 0),
            "f1_macro": eval_results.get("eval_f1_macro", 0),
        }
    }

    with open(OUTPUT_DIR / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 模型保存至: {OUTPUT_DIR}")

    # 对比
    print("\n" + "=" * 60)
    print("与 DAP 版本对比:")
    print("  V4 (使用 DAP): F1 = 0.331")
    print(f"  V5 (不用 DAP): F1 = {eval_results.get('eval_f1_weighted', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
