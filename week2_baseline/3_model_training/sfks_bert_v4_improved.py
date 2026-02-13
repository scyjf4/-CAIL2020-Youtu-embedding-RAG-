# -*- coding: utf-8 -*-
"""
改进版 BERT + LoRA 微调 - 增加数据量 + 优化训练策略
目标：将 F1 从 0.17 提升至 0.4+
"""
import json
import torch
import numpy as np
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, List, Dict

# ====== 关键改进配置 ======
SAMPLE_PER_FILE = 4000  # 增加到 4000 条/文件，总共约 8000 条
EPOCHS = 8              # 增加训练轮次
LEARNING_RATE = 3e-5    # 稍微提高学习率
BATCH_SIZE = 8          # 减小 batch size 以适应更长序列
MAX_SEQ_LENGTH = 256    # 适当减少序列长度加速训练

print("\n" + "=" * 60)
print("改进版 BERT + LoRA 训练")
print(f"  数据量: 每个文件 {SAMPLE_PER_FILE} 条，约 {SAMPLE_PER_FILE * 2} 条总计")
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
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # my_intern/
DATA_DIR = PROJECT_ROOT / "data" / "sfks"
OUTPUT_DIR = SCRIPT_DIR.parent / "models" / "sfks_bert_lora_v4_improved"

# 模型配置 - 优先使用 DAP 模型
DAP_MODEL_PATH = SCRIPT_DIR.parent / "models" / "sfks_bert_dap" / "model"
BASE_MODEL = str(DAP_MODEL_PATH) if DAP_MODEL_PATH.exists() else "hfl/chinese-bert-wwm-ext"

# LoRA 参数 - 增加容量
LORA_RANK = 64       # 增加 rank
LORA_ALPHA = 128     # 增加 alpha
LORA_DROPOUT = 0.05  # 减少 dropout

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_sfks_data(data_dir: Path, sample_per_file: int = 4000) -> List[Dict]:
    """加载司法考试数据，使用更多数据"""
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

                    # 只用单选题
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

        # 采样
        if len(file_data) > sample_per_file:
            sampled = random.sample(file_data, sample_per_file)
            print(f"    从 {len(file_data)} 条中采样 {sample_per_file} 条")
        else:
            sampled = file_data
            print(f"    全部使用 {len(file_data)} 条")

        all_data.extend(sampled)

    # 打乱数据
    random.shuffle(all_data)

    print(f"\n  ✓ 单选题总数: {len(all_data)} 条")
    print(f"  ✓ 多选题(已跳过): {multi_count} 条")

    # 答案分布
    answer_dist = {"A": 0, "B": 0, "C": 0, "D": 0}
    for item in all_data:
        ans = item["answer"]
        if ans in answer_dist:
            answer_dist[ans] += 1
    print(f"  ✓ 答案分布: {answer_dist}")

    # 检查分布是否均衡
    values = list(answer_dist.values())
    if max(values) > 2 * min(values):
        print("  ⚠ 警告: 答案分布不均衡，可能影响训练效果")

    return all_data


def create_dataset(data_list: List[Dict]) -> Dataset:
    """转换为 HuggingFace Dataset"""
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
    """预处理数据 - 优化版"""
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

        # 为每个选项创建输入对
        for option_key in ["A", "B", "C", "D"]:
            # 问题作为第一个句子
            first_sentences.append(q_text)
            # 选项内容作为第二个句子，带上选项标签
            opt_text = opts.get(option_key, "[未知]")
            second_sentences.append(f"{option_key}. {opt_text}")

    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

    # 重组为 [batch_size, num_choices, seq_len] 格式
    result = {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized.items()}
    result["label"] = encoded_labels

    return result


@dataclass
class DataCollatorForMultipleChoice:
    """多选题数据整理器"""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None

    def __call__(self, features):
        labels = [feature.pop("label") for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        # 展平
        flattened = []
        for feature in features:
            for i in range(num_choices):
                flattened.append({k: v[i] for k, v in feature.items()})

        # padding
        batch = self.tokenizer.pad(
            flattened,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # 重组
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def compute_metrics(eval_pred):
    """计算详细指标"""
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
    print(f"\n[1/6] 检查模型...")
    if DAP_MODEL_PATH.exists():
        print(f"  ✓ 使用 DAP 预训练模型: {DAP_MODEL_PATH}")
    else:
        print(f"  使用原始 BERT: {BASE_MODEL}")

    # 加载 tokenizer
    print(f"\n[2/6] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 加载数据
    print(f"\n[3/6] 加载数据 (目标: {SAMPLE_PER_FILE * 2} 条)...")
    data_list = load_sfks_data(DATA_DIR, sample_per_file=SAMPLE_PER_FILE)
    dataset = create_dataset(data_list)

    # 加载模型
    print(f"\n[4/6] 加载模型并配置 LoRA...")
    base_model = AutoModelForMultipleChoice.from_pretrained(BASE_MODEL)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["query", "key", "value", "dense"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 预处理
    print(f"\n[5/6] 预处理数据...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    # 划分训练/验证集
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  - 训练集: {len(split['train'])} 条")
    print(f"  - 验证集: {len(split['test'])} 条")

    # 训练
    print(f"\n[6/6] 开始训练...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",  # 使用 F1 作为最佳模型指标
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
        gradient_accumulation_steps=4,  # 等效 batch_size = 32
        fp16=torch.cuda.is_available(),
        seed=42,

        # 防止过拟合
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

    # 保存
    print(f"\n✓ 训练完成!")
    model.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # 最终评估
    print("\n" + "=" * 60)
    print("最终评估结果")
    print("=" * 60)
    eval_results = trainer.evaluate()

    print(f"  准确率 (Accuracy):    {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"  精确率 (Precision):   {eval_results.get('eval_precision', 0):.4f}")
    print(f"  召回率 (Recall):      {eval_results.get('eval_recall', 0):.4f}")
    print(f"  F1 分数 (weighted):   {eval_results.get('eval_f1_weighted', 0):.4f}")
    print(f"  F1 分数 (macro):      {eval_results.get('eval_f1_macro', 0):.4f}")

    # 保存结果
    results = {
        "config": {
            "base_model": BASE_MODEL,
            "dap_used": DAP_MODEL_PATH.exists(),
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

    # 对比提示
    print("\n" + "=" * 60)
    print("与之前结果对比:")
    print("  之前 F1 (weighted): 0.1780")
    print(f"  当前 F1 (weighted): {eval_results.get('eval_f1_weighted', 0):.4f}")
    improvement = eval_results.get('eval_f1_weighted', 0) - 0.1780
    if improvement > 0:
        print(f"  ✓ 提升: +{improvement:.4f}")
    else:
        print(f"  ✗ 下降: {improvement:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
