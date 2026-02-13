# -*- coding: utf-8 -*-
"""
CAIL2020 司法考试 F1 分数计算脚本 (优化版)
只评估单选题，与训练数据一致
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from peft import PeftModel


class SfksInferenceV2:
    """司法考试问答推理类 V2"""

    def __init__(self, model_path=None, base_model="hfl/chinese-bert-wwm-ext", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = Path(__file__).parent / "sfks_bert_lora_output_v2"
        self.model_path = Path(model_path)

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
            max_length=512,
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

        return predicted_label, float(confidence), logits.tolist()


class SfksMetricsCalculatorV2:
    """司法考试模型评估类 V2"""

    def __init__(self):
        self.label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    def evaluate_from_file(self, data_dir: str, model_path: str = None, sample_size: int = None) -> Dict:
        print("加载模型...")
        inference = SfksInferenceV2(model_path)

        print(f"加载数据: {data_dir}")
        data_dir = Path(data_dir)
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
                            # 只评估单选题
                            if isinstance(answer, list) and len(answer) == 1:
                                if isinstance(item.get("option_list"), dict) and len(item["option_list"]) == 4:
                                    data.append(item)
                        except json.JSONDecodeError:
                            continue

        if sample_size and sample_size < len(data):
            import random
            random.seed(42)
            data = random.sample(data, sample_size)

        print(f"✓ 加载了 {len(data)} 条单选题数据\n")

        print("进行推理...")
        predictions = []
        ground_truths = []

        for i, item in enumerate(data):
            if i % 100 == 0:
                print(f"  处理进度: {i}/{len(data)}")

            question = item.get("statement", "")
            options = item.get("option_list", {})

            pred_label, _, _ = inference.predict_single(question, options)
            predictions.append(pred_label)

            true_label = item.get("answer", [])[0]
            ground_truths.append(true_label)

        print(f"✓ 推理完成\n")

        # 计算指标
        pred_idx = [self.label_map.get(p, 0) for p in predictions]
        true_idx = [self.label_map.get(g, 0) for g in ground_truths]

        metrics = {
            "accuracy": accuracy_score(true_idx, pred_idx),
            "precision": precision_score(true_idx, pred_idx, average="weighted", zero_division=0),
            "recall": recall_score(true_idx, pred_idx, average="weighted", zero_division=0),
            "f1_weighted": f1_score(true_idx, pred_idx, average="weighted", zero_division=0),
            "f1_macro": f1_score(true_idx, pred_idx, average="macro", zero_division=0),
        }

        # 分类报告
        print("\n分类报告:")
        print(classification_report(true_idx, pred_idx, target_names=["A", "B", "C", "D"], zero_division=0))

        return {
            "data_dir": str(data_dir),
            "model_path": str(model_path or "sfks_bert_lora_output_v2"),
            "total_samples": len(data),
            "metrics": metrics
        }


def print_metrics(results: Dict) -> None:
    print("=" * 60)
    print("CAIL2020 司法考试模型评估结果 (V2 - 仅单选题)")
    print("=" * 60)
    print(f"\n数据目录: {results['data_dir']}")
    print(f"模型路径: {results['model_path']}")
    print(f"样本总数: {results['total_samples']}\n")

    metrics = results['metrics']
    print("性能指标:")
    print(f"  准确率 (Accuracy):    {metrics['accuracy']:.4f}")
    print(f"  精确率 (Precision):   {metrics['precision']:.4f}")
    print(f"  召回率 (Recall):      {metrics['recall']:.4f}")
    print(f"  F1 分数 (weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  F1 分数 (macro):      {metrics['f1_macro']:.4f}")
    print()


def main():
    import sys
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent / "data" / "sfks"

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return

    calculator = SfksMetricsCalculatorV2()
    results = calculator.evaluate_from_file(str(data_dir), sample_size=1000)

    print_metrics(results)

    output_file = script_dir / "sfks_evaluation_results_v2.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ 结果已保存至: {output_file}\n")


if __name__ == "__main__":
    main()
