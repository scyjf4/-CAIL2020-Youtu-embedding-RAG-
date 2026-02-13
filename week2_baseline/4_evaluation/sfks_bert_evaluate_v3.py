# -*- coding: utf-8 -*-
"""
综合优化版评估脚本
支持 v3 模型评估
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from peft import PeftModel


SCRIPT_DIR = Path(__file__).parent
DAP_MODEL_PATH = SCRIPT_DIR.parent / "models" / "sfks_bert_dap" / "model"
BASE_MODEL = str(DAP_MODEL_PATH) if DAP_MODEL_PATH.exists() else "hfl/chinese-bert-wwm-ext"


class SfksInferenceV3:
    """司法考试问答推理类 V3"""

    def __init__(self, model_path=None, base_model=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = SCRIPT_DIR.parent / "models" / "sfks_bert_lora_output_v3"
        self.model_path = Path(model_path)

        if base_model is None:
            base_model = BASE_MODEL

        print(f"[推理初始化] 设备: {self.device}")
        print(f"[1/3] 加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        print(f"[2/3] 加载基础模型: {base_model}")
        base_model_obj = AutoModelForMultipleChoice.from_pretrained(base_model)

        print(f"[3/3] 加载 LoRA 适配器...")
        self.model = PeftModel.from_pretrained(base_model_obj, str(self.model_path / "lora_adapter"))
        self.model.to(self.device)
        self.model.eval()
        print("✓ 模型加载完成\n")

    def predict_single(self, question: str, options: Dict[str, str]):
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


def evaluate_model(data_dir: Path, model_path: Path = None, sample_size: int = None):
    """评估模型"""
    print("=" * 60)
    print("CAIL2020 司法考试模型评估 (V3)")
    print("=" * 60)

    # 加载模型
    print("\n加载模型...")
    inference = SfksInferenceV3(model_path)

    # 加载数据 (只评估单选题)
    print(f"加载数据: {data_dir}")
    data = []

    for filename in ["0_train.json", "1_train.json"]:
        filepath = data_dir / filename
        if not filepath.exists():
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        answer = item.get("answer", [])
                        if isinstance(answer, list) and len(answer) == 1:
                            options = item.get("option_list", {})
                            if isinstance(options, dict) and len(options) == 4:
                                data.append(item)
                    except json.JSONDecodeError:
                        continue

    # 采样
    if sample_size and sample_size < len(data):
        import random
        random.seed(42)
        data = random.sample(data, sample_size)

    print(f"✓ 加载了 {len(data)} 条单选题数据\n")

    # 推理
    print("进行推理...")
    predictions = []
    ground_truths = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    for i, item in enumerate(data):
        if i % 200 == 0:
            print(f"  进度: {i}/{len(data)}")

        statement = item.get("statement", "")
        options = item.get("option_list", {})

        # 构建增强问题 (与训练一致)
        context_parts = [f"{k}: {v}" for k, v in sorted(options.items())]
        enhanced_question = f"{statement} [选项: {' | '.join(context_parts)}]"

        pred_label, _ = inference.predict_single(enhanced_question, options)
        predictions.append(pred_label)
        ground_truths.append(item["answer"][0])

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

    # 输出结果
    print("=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"\n样本数: {len(data)}")
    print(f"\n性能指标:")
    print(f"  准确率 (Accuracy):    {metrics['accuracy']:.4f}")
    print(f"  精确率 (Precision):   {metrics['precision']:.4f}")
    print(f"  召回率 (Recall):      {metrics['recall']:.4f}")
    print(f"  F1 分数 (weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  F1 分数 (macro):      {metrics['f1_macro']:.4f}")

    print("\n分类报告:")
    print(classification_report(true_idx, pred_idx, target_names=["A", "B", "C", "D"], zero_division=0))

    # 保存结果
    results = {
        "total_samples": len(data),
        "metrics": metrics,
    }
    output_file = SCRIPT_DIR / "sfks_evaluation_results_v3.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ 结果已保存至: {output_file}\n")

    return metrics


def main():
    import sys
    data_dir = SCRIPT_DIR.parent.parent / "data" / "sfks"

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return

    evaluate_model(data_dir, sample_size=1000)


if __name__ == "__main__":
    main()
