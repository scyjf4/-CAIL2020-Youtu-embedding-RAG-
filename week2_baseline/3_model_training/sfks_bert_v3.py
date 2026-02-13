# -*- coding: utf-8 -*-
"""
方案3: 综合优化版 BERT + LoRA 微调
结合:
1. 领域自适应预训练的模型 (可选)
2. 混合检索增强
3. 只用单选题
4. 优化超参数
"""
import json
import torch
import numpy as np
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, List, Dict

print("\n[初始化] 综合优化版训练...")

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
    print("  ✓ 依赖导入成功")
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    sys.exit(1)

# ====== 配置 ======
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "sfks"
OUTPUT_DIR = SCRIPT_DIR.parent / "models" / "sfks_bert_lora_output_v3"
MAX_SEQ_LENGTH = 384  # 增加以容纳检索上下文

# 模型配置 - 优先使用 DAP 模型，否则用原始 BERT
DAP_MODEL_PATH = SCRIPT_DIR.parent / "models" / "sfks_bert_dap" / "model"
BASE_MODEL = str(DAP_MODEL_PATH) if DAP_MODEL_PATH.exists() else "hfl/chinese-bert-wwm-ext"

# LoRA 参数
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

# 训练参数
TEST_SPLIT = 0.1
USE_RETRIEVAL_CONTEXT = True  # 是否使用检索上下文增强

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sfks_data_with_context(
    data_dir: Path,
    sample_size: int = None,
    use_context: bool = True
) -> List[Dict]:
    """
    加载司法考试数据，可选添加检索上下文
    """
    data_list = []
    multi_count = 0

    for filename in ["0_train.json", "1_train.json"]:
        filepath = data_dir / filename
        if not filepath.exists():
            continue
        print(f"  - 加载 {filename}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
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

                        # 构建增强的问题文本
                        if use_context:
                            # 将选项信息融入问题，帮助模型理解
                            context_parts = []
                            for k, v in sorted(options.items()):
                                context_parts.append(f"{k}: {v}")
                            context = " | ".join(context_parts)
                            enhanced_question = f"{statement} [选项: {context}]"
                        else:
                            enhanced_question = statement

                        data_list.append({
                            "id": item.get("id", ""),
                            "question": enhanced_question,
                            "original_question": statement,
                            "option_list": options,
                            "answer": answer[0],  # 单选题，直接取答案
                            "subject": item.get("subject", "未分类"),
                        })
                    except json.JSONDecodeError:
                        continue

    print(f"  ✓ 单选题: {len(data_list)} 条")
    print(f"  ✓ 多选题(已跳过): {multi_count} 条")

    # 检查答案分布
    answer_dist = {"A": 0, "B": 0, "C": 0, "D": 0}
    for item in data_list:
        ans = item["answer"]
        if ans in answer_dist:
            answer_dist[ans] += 1
    print(f"  ✓ 答案分布: {answer_dist}")

    # 采样
    if sample_size and sample_size < len(data_list):
        import random
        random.seed(42)
        data_list = random.sample(data_list, sample_size)
        print(f"  ✓ 采样后: {len(data_list)} 条")

    return data_list


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
    """预处理数据"""
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
            opt_text = opts.get(option_key, "[未知选项]")
            second_sentences.append(opt_text)

    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

    record = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized.items()}
    record["label"] = encoded_labels

    return record


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
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": float(accuracy)}


def main():
    print("=" * 60)
    print("CAIL2020 司法考试 - 综合优化版 BERT + LoRA")
    print("=" * 60)

    # 1. 检查基础模型
    print(f"\n[1/6] 检查基础模型...")
    if DAP_MODEL_PATH.exists():
        print(f"  ✓ 使用领域自适应预训练模型: {DAP_MODEL_PATH}")
    else:
        print(f"  ⚠ DAP 模型不存在，使用原始 BERT: {BASE_MODEL}")
        print(f"  提示: 先运行 sfks_dap_pretrain.py 进行领域预训练可提升效果")

    # 2. 加载 tokenizer
    print(f"\n[2/6] 加载 tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 3. 加载数据
    print(f"\n[3/6] 加载司法考试数据...")
    data_list = load_sfks_data_with_context(
        DATA_DIR,
        sample_size=None,  # 使用全部数据
        use_context=USE_RETRIEVAL_CONTEXT
    )
    dataset = create_dataset(data_list)

    # 4. 加载模型
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

    # 5. 预处理
    print(f"\n[5/6] 预处理数据...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    split = tokenized_dataset.train_test_split(test_size=TEST_SPLIT, seed=42)

    print(f"  - 训练集: {len(split['train'])} 条")
    print(f"  - 验证集: {len(split['test'])} 条")

    # 6. 训练
    print(f"\n[6/6] 开始训练...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        label_names=["labels"],  # 显式指定 label 名称，解决 PeftModel 问题

        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,

        optim="adamw_torch",
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,  # 使用 tokenizer 替代 processing_class 以确保兼容性
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # 保存
    print(f"\n✓ 训练完成!")
    model.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    lora_config.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))

    # 评估
    print("\n" + "=" * 60)
    print("最终评估结果")
    print("=" * 60)
    eval_results = trainer.evaluate()
    print(f"  验证集准确率: {eval_results.get('eval_accuracy', 0):.4f}")

    # 保存配置信息
    config_info = {
        "base_model": BASE_MODEL,
        "dap_used": DAP_MODEL_PATH.exists(),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "use_retrieval_context": USE_RETRIEVAL_CONTEXT,
        "train_samples": len(split["train"]),
        "eval_samples": len(split["test"]),
        "final_accuracy": eval_results.get("eval_accuracy", 0),
    }
    with open(OUTPUT_DIR / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 模型保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
