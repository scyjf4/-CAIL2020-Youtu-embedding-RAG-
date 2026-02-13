# -*- coding: utf-8 -*-
"""
CAIL2020 司法考试 BERT + LoRA 微调脚本
基于司法考试向量库进行问答模型训练
"""
import json
import torch
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

print("\n[初始化] 检查依赖库版本...")

try:
    import transformers
    import datasets
    print(f"  ✓ transformers: {transformers.__version__}")
    print(f"  ✓ datasets: {datasets.__version__}")
    print(f"  ✓ torch: {torch.__version__}")
except ImportError as e:
    print(f"❌ 基础依赖缺失: {e}")
    sys.exit(1)

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForMultipleChoice,
        TrainingArguments,
        Trainer,
    )
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
    from datasets import Dataset
    print("  ✓ transformers 模块导入成功")
except ImportError as e:
    print(f"❌ Transformers 导入失败: {e}")
    sys.exit(1)

try:
    from peft import get_peft_model, LoraConfig
    print("  ✓ peft 模块导入成功")
except ImportError as e:
    print(f"❌ PEFT 导入失败: {e}")
    sys.exit(1)

print("✓ 所有依赖检查通过\n")

# ====== 配置参数 ======
MODEL_NAME = "hfl/chinese-bert-wwm-ext"
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "sfks"
OUTPUT_DIR = SCRIPT_DIR.parent / "models" / "sfks_bert_lora_output"
MAX_SEQ_LENGTH = 512

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

SAMPLE_SIZE = 3000
TEST_SPLIT = 0.1

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sfks_data(data_dir: Path, sample_size: int = None):
    """加载司法考试数据"""
    data_list = []

    for filename in ["0_train.json", "1_train.json"]:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  ⚠ 文件不存在: {filepath}")
            continue
        print(f"  - 加载 {filename}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        processed = {
                            "id": item.get("id", ""),
                            "question": item.get("statement", ""),  # sfks 用 statement
                            "option_list": item.get("option_list", {}),
                            "answer": item.get("answer", []),
                            "subject": item.get("subject", "未分类"),
                            "type": item.get("type", "0"),
                        }
                        if isinstance(processed["option_list"], dict):
                            data_list.append(processed)
                    except json.JSONDecodeError:
                        continue

    print(f"  ✓ 共加载 {len(data_list)} 条数据")

    if sample_size and sample_size < len(data_list):
        import random
        random.seed(42)
        data_list = random.sample(data_list, sample_size)
        print(f"  ✓ 采样后: {len(data_list)} 条")

    for item in data_list:
        item["option_list"] = json.dumps(item["option_list"], ensure_ascii=False)
        item["answer"] = json.dumps(item["answer"], ensure_ascii=False)

    return Dataset.from_list(data_list)


def preprocess_function(examples):
    """预处理数据"""
    questions = examples["question"]
    options_json = examples["option_list"]
    answers_json = examples["answer"]

    options_list = []
    for opt in options_json:
        if isinstance(opt, str):
            try:
                options_list.append(json.loads(opt))
            except json.JSONDecodeError:
                options_list.append({})
        else:
            options_list.append(opt if isinstance(opt, dict) else {})

    answers = []
    for ans in answers_json:
        if isinstance(ans, str):
            try:
                parsed = json.loads(ans)
                answers.append(parsed if isinstance(parsed, list) else [parsed])
            except json.JSONDecodeError:
                answers.append([ans])
        else:
            answers.append(ans if isinstance(ans, list) else [ans])

    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    encoded_labels = []
    first_sentences = []
    second_sentences = []

    for i in range(len(questions)):
        q_text = questions[i]
        opts = options_list[i]
        ans_label = answers[i]

        if isinstance(ans_label, list) and len(ans_label) > 0:
            label_idx = label_map.get(ans_label[0], 0)
        else:
            label_idx = 0

        encoded_labels.append(label_idx)

        for option_key in ["A", "B", "C", "D"]:
            first_sentences.append(q_text)
            opt_text = opts.get(option_key, "[未知选项]")
            second_sentences.append(opt_text)

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

    record = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    record["label"] = encoded_labels

    return record


@dataclass
class DataCollatorForMultipleChoice:
    """自定义 DataCollator，处理多选题的 Padding"""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]

        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": float(accuracy)}


# ====== 主程序 ======
if __name__ == "__main__":
    print("=" * 60)
    print("CAIL2020 司法考试 BERT + LoRA 微调")
    print("=" * 60)

    print(f"\n[1/6] 加载 tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"\n[2/6] 加载司法考试数据: {DATA_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"数据目录不存在: {DATA_DIR}")
    dataset = load_sfks_data(DATA_DIR, sample_size=SAMPLE_SIZE)

    print(f"\n[3/6] 加载模型: {MODEL_NAME}")
    base_model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)

    print(f"\n[4/6] 配置 LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["query", "value"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    print(f"\n[5/6] 预处理数据")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    split_datasets = tokenized_datasets.train_test_split(test_size=TEST_SPLIT, seed=42)

    print(f"  - 训练集: {len(split_datasets['train'])} 条")
    print(f"  - 验证集: {len(split_datasets['test'])} 条")

    print(f"\n[6/6] 设置训练参数并开始训练")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        logging_first_step=True,
        optim="adamw_torch",
        gradient_accumulation_steps=1,
        fp16=torch.cuda.is_available(),
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_datasets["train"],
        eval_dataset=split_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\n开始训练...\n")
    trainer.train()

    print(f"\n✓ 训练完成!")
    print(f"✓ 模型已保存至: {OUTPUT_DIR}")

    model.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
    print(f"✓ LoRA 适配器已保存至: {OUTPUT_DIR / 'lora_adapter'}")

    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"✓ Tokenizer 已保存")

    lora_config.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
    print(f"✓ LoRA 配置已保存\n")
