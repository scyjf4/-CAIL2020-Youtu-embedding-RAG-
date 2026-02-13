# -*- coding: utf-8 -*-

"""
CAIL2020 法律问答 BERT + LoRA 推理脚本
加载已训练的模型进行预测
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from peft import PeftModel
from typing import Dict, List, Tuple


class LawQAInference:
    def __init__(
        self,
        model_path: str = None,
        base_model: str = "hfl/chinese-bert-wwm-ext",
        device: str = None
    ):
        """
        初始化推理模型

        Args:
            model_path: LoRA 模型输出目录
            base_model: 基础模型名称
            device: 设备 ("cuda" / "cpu")
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 如果没有提供路径，使用默认路径
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "cail_bert_lora_output"
        self.model_path = Path(model_path)

        print(f"[推理初始化] 设备: {self.device}")

        # 加载 tokenizer
        print(f"[1/3] 加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        # 加载基础模型
        print(f"[2/3] 加载基础模型: {base_model}")
        base_model_obj = AutoModelForMultipleChoice.from_pretrained(base_model)

        # 加载 LoRA 适配器
        print(f"[3/3] 加载 LoRA 适配器...")
        self.model = PeftModel.from_pretrained(
            base_model_obj,
            str(self.model_path / "lora_adapter")
        )
        self.model.to(self.device)
        self.model.eval()

        print("✓ 模型加载完成\n")

    def predict_single(
        self,
        question: str,
        options: List[Dict[str, str]]
    ) -> Tuple[str, float, List[float]]:
        """
        预测单个问题的答案

        Args:
            question: 问题文本
            options: 选项列表 [{"label": "A", "text": "..."}]

        Returns:
            (predicted_label, confidence, all_logits)
        """
        # 构建输入
        option_texts = [opt['text'] for opt in options]
        option_labels = [opt['label'] for opt in options]

        # Tokenize
        inputs = self.tokenizer(
            [question] * len(option_texts),
            option_texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        )

        # 重新组织为 (1, num_choices, seq_len) 的形式
        input_ids = inputs["input_ids"].view(1, len(option_texts), -1).to(self.device)
        attention_mask = inputs["attention_mask"].view(1, len(option_texts), -1).to(self.device)
        token_type_ids = inputs["token_type_ids"].view(1, len(option_texts), -1).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        # 解析输出
        logits = outputs.logits.squeeze(0).cpu().numpy()
        scores = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        predicted_idx = np.argmax(logits)
        predicted_label = option_labels[predicted_idx]
        confidence = scores[predicted_idx]

        return predicted_label, float(confidence), logits.tolist()

    def predict_batch(
        self,
        data: List[Dict]
    ) -> List[Dict]:
        """
        批量预测

        Args:
            data: 数据列表，每个元素包含 "question" 和 "options"

        Returns:
            预测结果列表
        """
        results = []
        for item in data:
            pred_label, confidence, logits = self.predict_single(
                item["question"],
                item["options"]
            )
            results.append({
                "id": item.get("id", "unknown"),
                "question": item["question"],
                "predicted_label": pred_label,
                "confidence": confidence,
                "logits": logits
            })
        return results


def main():
    """演示推理用法"""

    # 初始化模型
    inference = LawQAInference()

    # 示例问题
    test_data = [
        {
            "question": "下列哪一项不是法律的渊源？",
            "options": [
                {"label": "A", "text": "宪法"},
                {"label": "B", "text": "法律"},
                {"label": "C", "text": "条例"},
                {"label": "D", "text": "判例法"}
            ]
        }
    ]

    # 进行推理
    print("开始推理...\n")
    results = inference.predict_batch(test_data)

    # 输出结果
    for result in results:
        print(f"问题: {result['question'][:50]}...")
        print(f"预测答案: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.4f}")
        print()


if __name__ == "__main__":
    main()
