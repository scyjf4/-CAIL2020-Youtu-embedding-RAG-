# -*- coding: utf-8 -*-
"""
方案1: 领域自适应预训练 (Domain Adaptive Pre-training)
用 CAIL2020 法律文本对 BERT 进行 MLM 继续预训练
让模型更好地理解法律领域术语
"""
import json
import torch
import sys
from pathlib import Path
from tqdm import tqdm

print("\n[初始化] 领域自适应预训练...")

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset
    print("  ✓ 依赖导入成功")
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    sys.exit(1)

# ====== 配置 ======
MODEL_NAME = "hfl/chinese-bert-wwm-ext"
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "sfks"
OUTPUT_DIR = SCRIPT_DIR.parent / "models" / "sfks_bert_dap"  # Domain Adaptive Pretrained
MAX_SEQ_LENGTH = 256
MLM_PROBABILITY = 0.15

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_legal_texts(data_dir: Path) -> list:
    """
    从司法考试数据中提取法律文本用于 MLM 预训练
    提取: 题目 + 选项文本
    """
    texts = []

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
                        # 提取题目
                        statement = item.get("statement", "")
                        if statement:
                            texts.append(statement)

                        # 提取选项
                        options = item.get("option_list", {})
                        if isinstance(options, dict):
                            for opt_text in options.values():
                                if opt_text and len(opt_text) > 10:
                                    texts.append(opt_text)
                    except json.JSONDecodeError:
                        continue

    print(f"  ✓ 共提取 {len(texts)} 条法律文本")
    return texts


def main():
    print("=" * 60)
    print("CAIL2020 司法考试 - 领域自适应预训练 (DAP)")
    print("=" * 60)

    # 1. 加载 tokenizer
    print(f"\n[1/5] 加载 tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. 加载法律文本
    print(f"\n[2/5] 加载法律文本...")
    texts = load_legal_texts(DATA_DIR)

    # 3. 创建数据集
    print(f"\n[3/5] 创建 MLM 数据集...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )

    # 划分训练/验证集
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  - 训练集: {len(split['train'])} 条")
    print(f"  - 验证集: {len(split['test'])} 条")

    # 4. 加载模型
    print(f"\n[4/5] 加载 MLM 模型: {MODEL_NAME}")
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    # 数据整理器 (自动 mask)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    # 5. 训练
    print(f"\n[5/5] 开始领域自适应预训练...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        seed=42,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 保存
    print(f"\n✓ 领域自适应预训练完成!")
    model.save_pretrained(str(OUTPUT_DIR / "model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "model"))
    print(f"✓ 模型已保存至: {OUTPUT_DIR / 'model'}")
    print("\n后续步骤: 使用此模型作为 sfks_bert_v3.py 的基础模型")


if __name__ == "__main__":
    main()
