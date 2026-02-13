# -*- coding: utf-8 -*-

"""
CAIL2020 法律问答 F1 分数计算脚本
支持单选题和多选题的性能评估
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from ydlj_bert_inference import LawQAInference


class MetricsCalculator:
    def __init__(self):
        self.label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.reverse_map = {v: k for k, v in self.label_map.items()}

    def evaluate_single_choice(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        评估单选题性能

        Args:
            predictions: 预测的标签列表
            ground_truths: 真实标签列表

        Returns:
            包含各项指标的字典
        """
        # 转换为数字标签
        pred_idx = [self.label_map[p] for p in predictions]
        true_idx = [self.label_map[g] for g in ground_truths]

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
        """
        评估多选题性能（每个样本可能有多个正确答案）

        Args:
            predictions: 预测的标签列表 [[A, C], ...]
            ground_truths: 真实标签列表

        Returns:
            包含各项指标的字典
        """
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
        data_file: str,
        model_path: str = "cail_bert_lora_output"
    ) -> Dict:
        """
        从数据文件进行完整评估

        Args:
            data_file: 数据文件路径
            model_path: 已训练模型路径

        Returns:
            评估结果
        """
        # 加载模型
        print("加载模型...")
        inference = LawQAInference(model_path)

        # 加载数据
        print(f"加载数据: {data_file}")
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        print(f"✓ 加载了 {len(data)} 条数据\n")

        # 进行推理
        print("进行推理...")
        predictions = []
        ground_truths = []

        for item in data:
            # 获取预测
            pred_label, _, _ = inference.predict_single(
                item["question"],
                item["options"]
            )
            predictions.append(pred_label)

            # 获取真实标签
            true_label = item["answer_option_label"]
            if isinstance(true_label, list):
                ground_truths.append(true_label[0])  # 多选题取第一个
            else:
                ground_truths.append(true_label)

        print(f"✓ 推理完成\n")

        # 计算指标
        print("计算指标...")
        metrics = self.evaluate_single_choice(predictions, ground_truths)

        # 输出结果
        results = {
            "data_file": str(data_file),
            "model_path": str(model_path),
            "total_samples": len(data),
            "metrics": metrics
        }

        return results


def print_metrics(results: Dict) -> None:
    """格式化输出评估结果"""
    print("=" * 60)
    print("CAIL2020 法律问答模型评估结果")
    print("=" * 60)
    print(f"\n数据文件: {results['data_file']}")
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

    # 使用示例
    data_file = Path(__file__).parent / "qa_train_retrieved_sample.jsonl"

    if len(sys.argv) > 1:
        data_file = sys.argv[1]

    if not Path(data_file).exists():
        print(f"错误: 数据文件不存在: {data_file}")
        return

    # 进行评估
    calculator = MetricsCalculator()
    results = calculator.evaluate_from_file(str(data_file))

    # 输出结果
    print_metrics(results)

    # 保存结果
    output_file = Path(__file__).parent / "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ 结果已保存至: {output_file}\n")


if __name__ == "__main__":
    main()
