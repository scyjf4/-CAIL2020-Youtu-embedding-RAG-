# -*- coding: utf-8 -*-
"""
CAIL2020 司法考试 BERT + LoRA 推理脚本
加载已训练的模型进行预测
"""
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from peft import PeftModel
from typing import Dict, List, Tuple


class SfksInference:
    """司法考试问答推理类"""

    def __init__(
        self,
        model_path: str = None,
        base_model: str = "hfl/chinese-bert-wwm-ext",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "sfks_bert_lora_output"
        self.model_path = Path(model_path)

        print(f"[推理初始化] 设备: {self.device}")

        print(f"[1/3] 加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        print(f"[2/3] 加载基础模型: {base_model}")
        base_model_obj = AutoModelForMultipleChoice.from_pretrained(base_model)

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
        options: Dict[str, str]
    ) -> Tuple[str, float, List[float]]:
        """
        预测单个问题的答案

        Args:
            question: 问题文本
            options: 选项字典 {"A": "...", "B": "...", "C": "...", "D": "..."}

        Returns:
            (predicted_label, confidence, all_logits)
        """
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
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        logits = outputs.logits.squeeze(0).cpu().numpy()
        scores = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        predicted_idx = np.argmax(logits)
        predicted_label = option_keys[predicted_idx]
        confidence = scores[predicted_idx]

        return predicted_label, float(confidence), logits.tolist()

    def predict_batch(self, data: List[Dict]) -> List[Dict]:
        """批量预测"""
        results = []
        for item in data:
            question = item.get("question") or item.get("statement", "")
            options = item.get("option_list", {})

            pred_label, confidence, logits = self.predict_single(question, options)
            results.append({
                "id": item.get("id", "unknown"),
                "question": question,
                "predicted_label": pred_label,
                "confidence": confidence,
                "logits": logits
            })
        return results


def main():
    """演示推理用法"""
    inference = SfksInference()

    test_data = [
        {
            "id": "test_1",
            "statement": "下列哪一项不是法律的渊源？",
            "option_list": {
                "A": "宪法",
                "B": "法律",
                "C": "条例",
                "D": "判例法"
            }
        },
        {
            "id": "test_2",
            "statement": "关于刑法的解释及其效力，下列说法正确的是？",
            "option_list": {
                "A": "立法解释具有最高法律效力",
                "B": "司法解释只能由最高人民法院作出",
                "C": "学理解释没有法律约束力",
                "D": "行政解释可以适用于所有刑事案件"
            }
        }
    ]

    print("开始推理...\n")
    results = inference.predict_batch(test_data)

    for result in results:
        print(f"问题: {result['question'][:60]}...")
        print(f"预测答案: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.4f}")
        print()


if __name__ == "__main__":
    main()
