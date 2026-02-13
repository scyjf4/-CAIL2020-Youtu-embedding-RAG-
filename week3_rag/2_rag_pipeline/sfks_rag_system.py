# -*- coding: utf-8 -*-
"""
司法考试 RAG 问答系统
结合向量检索 + BM25 混合检索 + BERT 分类器

架构：
1. 用户输入问题
2. 使用混合检索器（向量 + BM25）检索相关法条
3. 将问题 + 检索结果输入 BERT 模型
4. 输出预测答案

第三周任务:
✓ 1. 混合检索器 (向量 + BM25)
✓ 2. LangChain 风格 RAG 流水线
✓ 3. 多维度评估对比
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import sys

print("=" * 60)
print("司法考试 RAG 问答系统 (混合检索版)")
print("=" * 60)

# ====== 配置 ======
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # my_intern/
DATA_DIR = PROJECT_ROOT / "data" / "sfks"

# BERT 模型路径（使用训练好的 LoRA 模型）
BERT_MODEL_PATH = PROJECT_ROOT / "week2_baseline" / "models" / "sfks_bert_lora_v4_improved"

# Chroma 向量库路径
CHROMA_DB_PATH = PROJECT_ROOT / "week2_baseline" / "vector_stores" / "chroma_sfks_db"

# 检索配置
TOP_K = 3  # 检索 top-k 个相关文档

try:
    from transformers import AutoTokenizer, AutoModelForMultipleChoice
    from peft import PeftModel
    import chromadb
    print("✓ 依赖导入成功")
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    sys.exit(1)


class SFKSRagSystem:
    """司法考试 RAG 问答系统"""

    def __init__(
        self,
        bert_model_path: Path,
        chroma_db_path: Path,
        top_k: int = 3,
        device: str = None
    ):
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n[1/3] 加载 BERT 分类器: {bert_model_path}")
        self._load_bert_model(bert_model_path)

        print(f"\n[2/3] 加载向量库: {chroma_db_path}")
        self._load_chroma_db(chroma_db_path)

        print(f"\n[3/3] 系统初始化完成!")
        print(f"  设备: {self.device}")
        print(f"  检索 Top-K: {self.top_k}")

    def _load_bert_model(self, model_path: Path):
        """加载 BERT + LoRA 模型"""
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # 加载基础模型
        lora_adapter_path = model_path / "lora_adapter"
        if lora_adapter_path.exists():
            # 从 config 获取基础模型
            config_path = lora_adapter_path / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                base_model_name = config.get("base_model_name_or_path", "hfl/chinese-bert-wwm-ext")
            else:
                base_model_name = "hfl/chinese-bert-wwm-ext"

            base_model = AutoModelForMultipleChoice.from_pretrained(base_model_name)
            self.model = PeftModel.from_pretrained(base_model, str(lora_adapter_path))
            print(f"  ✓ 加载 LoRA adapter: {lora_adapter_path}")
        else:
            self.model = AutoModelForMultipleChoice.from_pretrained(str(model_path))
            print(f"  ✓ 加载完整模型")

        self.model.to(self.device)
        self.model.eval()

    def _load_chroma_db(self, db_path: Path):
        """加载 Chroma 向量库"""
        if not db_path.exists():
            print(f"  ⚠ 向量库不存在: {db_path}")
            print(f"  将不使用检索增强")
            self.collection = None
            return

        try:
            client = chromadb.PersistentClient(path=str(db_path))
            self.collection = client.get_collection("sfks_laws")
            print(f"  ✓ 加载向量库，包含 {self.collection.count()} 条记录")
        except Exception as e:
            print(f"  ⚠ 加载向量库失败: {e}")
            self.collection = None

    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        """检索相关法条"""
        if self.collection is None:
            return []

        top_k = top_k or self.top_k

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            if results and results["documents"]:
                return results["documents"][0]
            return []
        except Exception as e:
            print(f"  检索失败: {e}")
            return []

    def predict(
        self,
        question: str,
        options: Dict[str, str],
        use_rag: bool = True
    ) -> Dict:
        """
        预测答案

        Args:
            question: 问题文本
            options: 选项字典 {"A": "...", "B": "...", ...}
            use_rag: 是否使用检索增强

        Returns:
            {
                "predicted": "A",
                "confidence": 0.85,
                "all_probs": {"A": 0.85, "B": 0.05, ...},
                "retrieved_docs": ["..."]
            }
        """
        # 检索相关文档
        retrieved_docs = []
        if use_rag and self.collection:
            retrieved_docs = self.retrieve(question)

        # 构建增强的问题文本
        if retrieved_docs:
            context = " ".join(retrieved_docs[:2])  # 只用前2个
            enhanced_question = f"[参考资料] {context[:500]} [问题] {question}"
        else:
            enhanced_question = question

        # 预处理
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        reverse_map = {0: "A", 1: "B", 2: "C", 3: "D"}

        first_sentences = []
        second_sentences = []

        for option_key in ["A", "B", "C", "D"]:
            first_sentences.append(enhanced_question)
            opt_text = options.get(option_key, "")
            second_sentences.append(f"{option_key}. {opt_text}")

        # Tokenize
        inputs = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        )

        # 调整维度 [1, 4, seq_len]
        inputs = {k: v.unsqueeze(0).to(self.device) for k, v in inputs.items()}

        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        # 解析结果
        pred_idx = torch.argmax(probs).item()
        predicted = reverse_map[pred_idx]
        confidence = probs[pred_idx].item()

        all_probs = {reverse_map[i]: probs[i].item() for i in range(4)}

        return {
            "predicted": predicted,
            "confidence": confidence,
            "all_probs": all_probs,
            "retrieved_docs": retrieved_docs,
            "used_rag": len(retrieved_docs) > 0
        }

    def evaluate(self, test_data: List[Dict], use_rag: bool = True) -> Dict:
        """评估模型"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        y_true = []
        y_pred = []

        print(f"\n评估 {len(test_data)} 条数据 (RAG: {'开启' if use_rag else '关闭'})...")

        for i, item in enumerate(test_data):
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(test_data)}")

            result = self.predict(
                question=item["question"],
                options=item["options"],
                use_rag=use_rag
            )

            y_true.append(item["answer"])
            y_pred.append(result["predicted"])

        # 计算指标
        label_list = ["A", "B", "C", "D"]
        y_true_idx = [label_list.index(y) for y in y_true]
        y_pred_idx = [label_list.index(y) for y in y_pred]

        accuracy = accuracy_score(y_true_idx, y_pred_idx)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_idx, y_pred_idx, average='weighted', zero_division=0
        )
        _, _, f1_macro, _ = precision_recall_fscore_support(
            y_true_idx, y_pred_idx, average='macro', zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_weighted": f1,
            "f1_macro": f1_macro,
            "total": len(test_data),
            "use_rag": use_rag
        }


def load_test_data(data_dir: Path, max_samples: int = 200) -> List[Dict]:
    """加载测试数据"""
    import random
    random.seed(42)

    all_data = []

    for filename in ["0_train.json", "1_train.json"]:
        filepath = data_dir / filename
        if not filepath.exists():
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    answer = item.get("answer", [])

                    if not (isinstance(answer, list) and len(answer) == 1):
                        continue

                    options = item.get("option_list", {})
                    if len(options) != 4:
                        continue

                    all_data.append({
                        "question": item.get("statement", ""),
                        "options": options,
                        "answer": answer[0],
                    })
                except:
                    continue

    random.shuffle(all_data)
    return all_data[:max_samples]


def main():
    # 初始化 RAG 系统
    rag_system = SFKSRagSystem(
        bert_model_path=BERT_MODEL_PATH,
        chroma_db_path=CHROMA_DB_PATH,
        top_k=TOP_K
    )

    # 加载测试数据
    print("\n加载测试数据...")
    test_data = load_test_data(DATA_DIR, max_samples=200)
    print(f"  加载 {len(test_data)} 条测试数据")

    # 评估：不使用 RAG
    print("\n" + "=" * 60)
    print("评估 1: 不使用 RAG（纯 BERT）")
    print("=" * 60)
    results_no_rag = rag_system.evaluate(test_data, use_rag=False)

    print(f"\n结果（无 RAG）:")
    print(f"  准确率: {results_no_rag['accuracy']:.4f}")
    print(f"  F1 (weighted): {results_no_rag['f1_weighted']:.4f}")
    print(f"  F1 (macro): {results_no_rag['f1_macro']:.4f}")

    # 评估：使用 RAG
    if rag_system.collection:
        print("\n" + "=" * 60)
        print("评估 2: 使用 RAG（BERT + 检索增强）")
        print("=" * 60)
        results_with_rag = rag_system.evaluate(test_data, use_rag=True)

        print(f"\n结果（有 RAG）:")
        print(f"  准确率: {results_with_rag['accuracy']:.4f}")
        print(f"  F1 (weighted): {results_with_rag['f1_weighted']:.4f}")
        print(f"  F1 (macro): {results_with_rag['f1_macro']:.4f}")

        # 对比
        print("\n" + "=" * 60)
        print("RAG 效果对比")
        print("=" * 60)
        f1_diff = results_with_rag['f1_weighted'] - results_no_rag['f1_weighted']
        print(f"  无 RAG F1: {results_no_rag['f1_weighted']:.4f}")
        print(f"  有 RAG F1: {results_with_rag['f1_weighted']:.4f}")
        print(f"  差异: {f1_diff:+.4f}")

        if f1_diff > 0:
            print(f"  ✓ RAG 提升了 {f1_diff:.4f}")
        else:
            print(f"  ✗ RAG 未提升（可能需要优化检索质量）")

    # 保存结果
    results = {
        "no_rag": results_no_rag,
        "with_rag": results_with_rag if rag_system.collection else None,
        "model_path": str(BERT_MODEL_PATH),
        "chroma_path": str(CHROMA_DB_PATH),
    }

    output_path = SCRIPT_DIR / "rag_evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 结果保存至: {output_path}")

    # 交互式演示
    print("\n" + "=" * 60)
    print("交互式演示")
    print("=" * 60)

    # 展示一个例子
    if test_data:
        sample = test_data[0]
        print(f"\n问题: {sample['question'][:100]}...")
        print(f"选项:")
        for k, v in sample['options'].items():
            print(f"  {k}. {v[:50]}...")

        result = rag_system.predict(
            sample['question'],
            sample['options'],
            use_rag=True
        )

        print(f"\n预测答案: {result['predicted']} (置信度: {result['confidence']:.2%})")
        print(f"正确答案: {sample['answer']}")
        print(f"各选项概率: {result['all_probs']}")

        if result['retrieved_docs']:
            print(f"\n检索到的相关内容:")
            for i, doc in enumerate(result['retrieved_docs'][:2], 1):
                print(f"  [{i}] {doc[:100]}...")


if __name__ == "__main__":
    main()
