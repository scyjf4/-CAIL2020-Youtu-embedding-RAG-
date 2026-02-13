# -*- coding: utf-8 -*-
import json
import torch
import numpy as np
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union,Any

print("\n[初始化] 检查依赖库版本...")

# 版本检查
try:
    import transformers
    import datasets
    import torch

    print(f"  ✓ transformers: {transformers.__version__}")
    print(f"  ✓ datasets: {datasets.__version__}")
    print(f"  ✓ torch: {torch.__version__}")

except ImportError as e:
    print(f"❌ 基础依赖缺失: {e}")
    print("运行: pip install -r requirements_lora.txt")
    sys.exit(1)

# 导入 transformers 模块
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
    print("运行: pip install -r requirements_lora.txt")
    sys.exit(1)

# 导入 PEFT（可能存在版本兼容性问题）
try:
    from peft import get_peft_model, LoraConfig
    print("  ✓ peft 模块导入成功")
except ImportError as e:
    print(f"❌ PEFT 导入失败: {e}")
    print("\n解决方案:")
    print("  运行: python fix_dependencies.py")
    print("  或: pip install --upgrade peft accelerate")
    sys.exit(1)

print("✓ 所有依赖检查通过\n")

# ====== 配置参数 ======
MODEL_NAME = "hfl/chinese-bert-wwm-ext"
SCRIPT_DIR = Path(__file__).parent
# 数据文件在 2_retrieval/ 目录下
DATA_FILE = SCRIPT_DIR.parent / "2_retrieval" / "qa_train_retrieved_sample.jsonl"
OUTPUT_DIR = SCRIPT_DIR.parent / "models" / "cail_bert_lora_output"
MAX_SEQ_LENGTH = 512

# LoRA 参数（已优化）
LORA_RANK = 16          # ↑ 增加秩以提高模型容量（从 8 改为 16）
LORA_ALPHA = 32         # 对应调整 alpha
LORA_DROPOUT = 0.05

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _to_arrow_scalar(v: Any) -> Any:
    """Make values safe for pyarrow: keep scalars, stringify list/dict, decode bytes."""
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v

def _normalize_row(row: dict) -> dict:
    return {k: _to_arrow_scalar(v) for k, v in row.items()}

# 1. 加载处理好的 jsonl 数据
def load_data(file_path):
    """从 JSONL 文件加载数据，将嵌套列表字段转换为字符串"""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # 转换嵌套列表为 JSON 字符串，以便 PyArrow 可以处理
                if isinstance(item.get('options'), list):
                    item['options'] = json.dumps(item['options'], ensure_ascii=False)
                if isinstance(item.get('answer_option_label'), list):
                    item['answer_option_label'] = json.dumps(item['answer_option_label'], ensure_ascii=False)
                data_list.append(item)
    print(f"✓ 加载了 {len(data_list)} 条数据")
    data_list = [_normalize_row(r) for r in data_list]
    return Dataset.from_list(data_list)

# 2. 数据预处理函数
def preprocess_function(examples):
    """
    预处理数据，将问题与选项组合成 BERT 输入格式
    支持单选题（label为字符串）和多选题（label为列表）
    """
    questions = examples["question"]
    options_json = examples["options"]
    labels_json = examples["answer_option_label"]

    # 反序列化 JSON 字符串回原始数据
    options = []
    for opt in options_json:
        if isinstance(opt, str):
            try:
                options.append(json.loads(opt))
            except json.JSONDecodeError:
                # 如果不是有效的 JSON，当作已是列表处理
                options.append(opt)
        else:
            options.append(opt)

    labels = []
    for lbl in labels_json:
        if isinstance(lbl, str):
            try:
                parsed = json.loads(lbl)
                labels.append(parsed)
            except json.JSONDecodeError:
                # 如果 JSON 解析失败，说明本来就是普通字符串（如 "B"）
                labels.append(lbl)
        else:
            labels.append(lbl)

    # 映射选项标签到索引 A->0, B->1, C->2, D->3
    label_map = {k: i for i, k in enumerate(["A", "B", "C", "D"])}

    encoded_labels = []
    first_sentences = []
    second_sentences = []

    for i in range(len(questions)):
        q_text = questions[i]
        opts = options[i]
        ans_label = labels[i]

        # 处理单选题 (label 为字符串) 或多选题 (label 为列表)
        if isinstance(ans_label, list):
            # 多选题：取第一个正确答案（用于多选一）
            label_idx = label_map.get(ans_label[0], 0)
        else:
            # 单选题：直接映射
            label_idx = label_map.get(ans_label, 0)

        encoded_labels.append(label_idx)

        # 构建选项字典 {A: text_a, B: text_b, ...}
        current_options_dict = {}
        for opt in opts:
            # opt 可能是 dict（'label', 'text'）或其他格式
            if isinstance(opt, dict):
                current_options_dict[opt['label']] = opt['text']

        # 为每个问题创建 4 个 (question, option) 对
        for option_key in ["A", "B", "C", "D"]:
            first_sentences.append(q_text)
            opt_text = current_options_dict.get(option_key, "[未知选项]")
            second_sentences.append(opt_text)

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,  # 稍后在 DataCollator 中 padding
    )

    # 将平铺的数据重新组织为 (batch_size, num_choices, seq_len) 的形式
    # 例如 batch_size=2, num_choices=4，则 flatten 后有 8 条，现在变回 [2, 4, ...]
    record = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    record["label"] = encoded_labels

    return record


# 3. 自定义 DataCollator (处理 Padding)
@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]

        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        # Flatten: [batch, 4, seq] -> [batch*4, seq] 以便 padding
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

        # Un-flatten: [batch*4, seq] -> [batch, 4, seq]
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


# ====== 主程序 ======
if __name__ == "__main__":
    print("=" * 60)
    print("CAIL2020 法律问答 BERT + LoRA 微调")
    print("=" * 60)

    # 1. 加载 tokenizer
    print(f"\n[1/6] 加载 tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. 加载数据
    print(f"\n[2/6] 加载数据: {DATA_FILE}")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"数据文件不存在: {DATA_FILE}")
    dataset = load_data(DATA_FILE)

    # 3. 加载基础模型
    print(f"\n[3/6] 加载模型: {MODEL_NAME}")
    base_model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)

    # 4. 配置 LoRA
    print(f"\n[4/6] 配置 LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        # task_type 不指定也能工作（可选参数）
        # 多选题是序列分类任务，不需要显式指定
        target_modules=["query", "value"],  # 只在 attention 的 query 和 value 上应用 LoRA
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 5. 数据预处理和划分
    print(f"\n[5/6] 预处理数据")
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    split_datasets = tokenized_datasets.train_test_split(test_size=0.1, seed=42)

    print(f"  - 训练集: {len(split_datasets['train'])} 条")
    print(f"  - 验证集: {len(split_datasets['test'])} 条")

    # 6. 训练
    print(f"\n[6/6] 设置训练参数并开始训练")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,

        # ========== 优化的超参数 ==========
        learning_rate=1e-4,              # ↓ 降低学习率（LoRA 更敏感）
        per_device_train_batch_size=8,   # ↑ 增加批大小以获得更稳定的梯度
        per_device_eval_batch_size=8,
        num_train_epochs=10,             # ↑ 增加轮数（从 3 改为 10）
        warmup_steps=100,                # ✨ 新增：预热步数
        weight_decay=0.01,
        logging_steps=50,                # ↓ 更频繁地记录日志
        logging_first_step=True,

        # 优化器设置
        optim="adamw_torch",             # ✨ 新增：使用高效的 AdamW
        gradient_accumulation_steps=1,   # 梯度累积（如果 OOM 改为 2）

        # 混合精度和其他
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
        # compute_metrics 用于计算评估指标
        compute_metrics=lambda p: {
            "accuracy": float((np.argmax(p.predictions, axis=1) == p.label_ids).mean())
        }
    )

    print("\n开始训练...\n")
    trainer.train()

    # 保存模型
    print(f"\n✓ 训练完成!")
    print(f"✓ 模型已保存至: {OUTPUT_DIR}")

    # 保存 LoRA 权重
    model.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
    print(f"✓ LoRA 适配器已保存至: {OUTPUT_DIR / 'lora_adapter'}")

    # 保存 tokenizer
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"✓ Tokenizer 已保存")

    # 保存训练配置
    lora_config.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
    print(f"✓ LoRA 配置已保存\n")

