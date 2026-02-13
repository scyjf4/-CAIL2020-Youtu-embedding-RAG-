# -*- coding: utf-8 -*-
"""
综合优化版评估脚本 - 采样测试版
支持 v3_sample 模型评估
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from peft import PeftModel

# ====== 采样配置 ======
SAMPLE_SIZE = 200  # 评估时的采样数量（从测试数据中抽样）

SCRIPT_DIR = Path(__file__).parent
DAP_MODEL_PATH = SCRIPT_DIR.parent / "models" / "sfks_bert_dap" / "model"
BASE_MODEL = str(DAP_MODEL_PATH) if DAP_MODEL_PATH.exists() else "hfl/chinese-bert-wwm-ext"

# Sample 版本的模型路径
SAMPLE_MODEL_PATH = SCRIPT_DIR.parent / "models" / "sfks_bert_lora_output_v3_sample"


class SfksInferenceV3Sample:
    """司法考试问答推理类 V3 (采样版)"""

    def __init__(self, model_path=None, base_model=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = SAMPLE_MODEL_PATH
        self.model_path = Path(model_path)

        if base_model is None:
            base_model = BASE_MODEL

        print(f"[推理初始化] 设备: {self.device}")
        print(f"[1/3] 加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        print(f"[2/3] 加载基础模型: {base_model}")
        base_model_obj = AutoModelForMultipleChoice.from_pretrained(base_model)

        print(f"[3/3] 加载 LoRA 适配器...")
        lora_path = self.model_path / "lora_adapter"
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA 适配器不存在: {lora_path}\n请先运行 sfks_bert_v3_sample.py 训练模型")

        self.model = PeftModel.from_pretrained(base_model_obj, str(lora_path))
        self.model.to(self.device)
        self.model.eval()
        print("✓ 模型加载完成\n")

    def predict_single(self, question: str, options: Dict[str, str]):
        """单条预测"""
        option_keys = ["A", "B", "C", "D"]
        option_texts = [options.get(k, "[未知选项]") for k in option_keys]

        inputs = self.tokenizer(
            [question] * len(option_texts),
            option_texts,
            truncation=True,
            max_length=384,
            padding=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].view(1, len(option_texts), -1).to(self.device)
        attention_mask = inputs["attention_mask"].view(1, len(option_texts), -1).to(self.device)
        token_type_ids = inputs["token_type_ids"].view(1, len(option_texts), -1).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        logits = outputs.logits.squeeze(0).cpu().numpy()
        scores = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        predicted_idx = np.argmax(logits)
        predicted_label = option_keys[predicted_idx]
        confidence = scores[predicted_idx]

        return predicted_label, float(confidence)

    def predict_batch(self, questions: list, options_list: list):
        """批量预测"""
        results = []
        for q, opts in zip(questions, options_list):
            pred, conf = self.predict_single(q, opts)
            results.append((pred, conf))
        return results


def load_evaluation_data(data_dir: Path, sample_size: int = None):
    """
    加载评估数据 (只加载单选题)
    """
    import random
    random.seed(42)

    data = []
    multi_count = 0

    for filename in ["0_train.json", "1_train.json"]:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  ⚠ 文件不存在: {filepath}")
            continue

        print(f"  - 读取 {filename}...")
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
                        if isinstance(options, dict) and len(options) == 4:
                            data.append(item)
                    except json.JSONDecodeError:
                        continue

    print(f"  ✓ 单选题总数: {len(data)} 条")
    print(f"  ✓ 多选题(已跳过): {multi_count} 条")

    # 采样
    if sample_size and sample_size < len(data):
        data = random.sample(data, sample_size)
        print(f"  ✓ 采样后: {sample_size} 条")

    return data


def evaluate_model(data_dir: Path, model_path: Path = None, sample_size: int = SAMPLE_SIZE):
    """评估采样版模型"""
    print("=" * 60)
    print("CAIL2020 司法考试模型评估 (V3 采样测试版)")
    print("=" * 60)

    # 加载模型
    print("\n[1/3] 加载模型...")
    try:
        inference = SfksInferenceV3Sample(model_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None

    # 加载数据
    print(f"\n[2/3] 加载评估数据: {data_dir}")
    data = load_evaluation_data(data_dir, sample_size)

    if len(data) == 0:
        print("❌ 没有可用的评估数据")
        return None

    # 推理
    print(f"\n[3/3] 进行推理... (共 {len(data)} 条)")
    predictions = []
    ground_truths = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    # 记录一些错误样例
    error_samples = []

    for i, item in enumerate(data):
        if i % 50 == 0:
            print(f"  进度: {i}/{len(data)}")

        statement = item.get("statement", "")
        options = item.get("option_list", {})
        true_answer = item["answer"][0]

        # 构建增强问题 (与训练一致)
        context_parts = [f"{k}: {v}" for k, v in sorted(options.items())]
        enhanced_question = f"{statement} [选项: {' | '.join(context_parts)}]"

        pred_label, confidence = inference.predict_single(enhanced_question, options)
        predictions.append(pred_label)
        ground_truths.append(true_answer)

        # 记录错误样例（最多5个）
        if pred_label != true_answer and len(error_samples) < 5:
            error_samples.append({
                "question": statement[:100] + "..." if len(statement) > 100 else statement,
                "predicted": pred_label,
                "ground_truth": true_answer,
                "confidence": round(confidence, 4),
            })

    print(f"✓ 推理完成\n")

    # 计算指标
    pred_idx = [label_map.get(p, 0) for p in predictions]
    true_idx = [label_map.get(g, 0) for g in ground_truths]

    metrics = {
        "accuracy": accuracy_score(true_idx, pred_idx),
        "precision": precision_score(true_idx, pred_idx, average="weighted", zero_division=0),
        "recall": recall_score(true_idx, pred_idx, average="weighted", zero_division=0),
        "f1_weighted": f1_score(true_idx, pred_idx, average="weighted", zero_division=0),
        "f1_macro": f1_score(true_idx, pred_idx, average="macro", zero_division=0),
    }

    # 计算每个选项的准确率
    option_accuracy = {}
    for opt in ["A", "B", "C", "D"]:
        opt_indices = [i for i, g in enumerate(ground_truths) if g == opt]
        if opt_indices:
            correct = sum(1 for i in opt_indices if predictions[i] == ground_truths[i])
            option_accuracy[opt] = correct / len(opt_indices)
        else:
            option_accuracy[opt] = 0.0

    # 输出结果
    print("=" * 60)
    print("评估结果 (采样测试版)")
    print("=" * 60)
    print(f"\n样本数: {len(data)}")
    print(f"\n性能指标:")
    print(f"  准确率 (Accuracy):    {metrics['accuracy']:.4f}")
    print(f"  精确率 (Precision):   {metrics['precision']:.4f}")
    print(f"  召回率 (Recall):      {metrics['recall']:.4f}")
    print(f"  F1 分数 (weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  F1 分数 (macro):      {metrics['f1_macro']:.4f}")

    print(f"\n各选项准确率:")
    for opt, acc in option_accuracy.items():
        count = sum(1 for g in ground_truths if g == opt)
        print(f"  {opt}: {acc:.4f} ({count} 条)")

    print("\n分类报告:")
    print(classification_report(true_idx, pred_idx, target_names=["A", "B", "C", "D"], zero_division=0))

    if error_samples:
        print("\n错误样例分析 (前5个):")
        print("-" * 40)
        for i, err in enumerate(error_samples, 1):
            print(f"  [{i}] 问题: {err['question']}")
            print(f"      预测: {err['predicted']} (置信度: {err['confidence']})")
            print(f"      正确: {err['ground_truth']}")
            print()

    # 保存结果
    results = {
        "model_type": "v3_sample",
        "total_samples": len(data),
        "metrics": metrics,
        "option_accuracy": option_accuracy,
        "error_samples": error_samples,
    }
    output_file = SCRIPT_DIR / "sfks_evaluation_results_v3_sample.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ 结果已保存至: {output_file}\n")



    return metrics


def main():
    import sys
    data_dir = SCRIPT_DIR.parent.parent / "data" / "sfks"

    # 支持命令行参数
    sample_size = SAMPLE_SIZE
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
        except ValueError:
            data_dir = Path(sys.argv[1])

    if len(sys.argv) > 2:
        try:
            sample_size = int(sys.argv[2])
        except ValueError:
            pass

    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return

    if not SAMPLE_MODEL_PATH.exists():
        print(f"❌ 采样模型不存在: {SAMPLE_MODEL_PATH}")
        print("请先运行 sfks_bert_v3_sample.py 进行训练")
        return

    print(f"评估采样数量: {sample_size} 条")
    evaluate_model(data_dir, sample_size=sample_size)


if __name__ == "__main__":
    main()
