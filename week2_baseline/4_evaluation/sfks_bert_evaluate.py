# -*- coding: utf-8 -*-
"""
CAIL2020 司法考试 F1 分数计算脚本
支持单选题和多选题的性能评估
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sfks_bert_inference import SfksInference


class SfksMetricsCalculator:
    """司法考试模型评估类"""

    def __init__(self):
        self.label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.reverse_map = {v: k for k, v in self.label_map.items()}

    def evaluate_single_choice(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """评估单选题性能"""
        pred_idx = [self.label_map.get(p, 0) for p in predictions]
        true_idx = [self.label_map.get(g, 0) for g in ground_truths]

        metrics = {
            "accuracy": accuracy_score(true_idx, pred_idx),
            "precision": precision_score(true_idx, pred_idx, average="weighted", zero_division=0),
            "recall": recall_score(true_idx, pred_idx, average="weighted", zero_division=0),
            "f1_weighted": f1_score(true_idx, pred_idx, average="weighted", zero_division=0),
            "f1_macro": f1_score(true_idx, pred_idx, average="macro", zero_division=0),
        }

        return metrics

    def evaluate_multi_choice(
        self,
        predictions: List[List[str]],
        ground_truths: List[List[str]]
    ) -> Dict[str, float]:
        """评估多选题性能"""
        correct = 0
        for pred, truth in zip(predictions, ground_truths):
            if set(pred) == set(truth):
                correct += 1

        accuracy = correct / len(predictions) if predictions else 0

        return {
            "accuracy": accuracy,
            "correct_count": correct,
            "total_count": len(predictions),
        }

    def evaluate_from_file(
        self,
        data_dir: str,
        model_path: str = None,
        sample_size: int = None
    ) -> Dict:
        """从数据文件进行完整评估"""
        print("加载模型...")
        inference = SfksInference(model_path)

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
                            data.append(item)
                        except json.JSONDecodeError:
                            continue

        if sample_size and sample_size < len(data):
            import random
            random.seed(42)
            data = random.sample(data, sample_size)

        print(f"✓ 加载了 {len(data)} 条数据\n")

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

            true_label = item.get("answer", [])
            if isinstance(true_label, list) and len(true_label) > 0:
                ground_truths.append(true_label[0])
            elif isinstance(true_label, str):
                ground_truths.append(true_label)
            else:
                ground_truths.append("A")

        print(f"✓ 推理完成\n")

        print("计算指标...")
        metrics = self.evaluate_single_choice(predictions, ground_truths)

        results = {
            "data_dir": str(data_dir),
            "model_path": str(model_path or "sfks_bert_lora_output"),
            "total_samples": len(data),
            "metrics": metrics
        }

        return results


def print_metrics(results: Dict) -> None:
    """格式化输出评估结果"""
    print("=" * 60)
    print("CAIL2020 司法考试模型评估结果")
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
    """主函数"""
    import sys

    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent / "data" / "sfks"

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return

    calculator = SfksMetricsCalculator()
    results = calculator.evaluate_from_file(
        str(data_dir),
        sample_size=500
    )

    print_metrics(results)

    output_file = script_dir / "sfks_evaluation_results_v1.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ 结果已保存至: {output_file}\n")


if __name__ == "__main__":
    main()
