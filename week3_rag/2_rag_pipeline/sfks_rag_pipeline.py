# -*- coding: utf-8 -*-
"""
完整 RAG 问答流水线
基于 LangChain 框架，集成混合检索器与开源 LLM

架构:
1. 混合检索器 (向量 + BM25)
2. LangChain 框架整合
3. 支持多种 LLM 后端 (ChatGLM, Qwen, 本地模型)
4. 完整的评估指标

注意: 由于 8GB 显存限制，默认使用 BERT 分类器
      如果有更多显存，可以切换到 ChatGLM
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys
import time

print("=" * 60)
print("SFKS RAG 完整问答流水线")
print("=" * 60)

# ====== 依赖检查 ======
try:
    from transformers import AutoTokenizer, AutoModelForMultipleChoice
    from peft import PeftModel
    import chromadb
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠ transformers/peft 未安装，部分功能不可用")

try:
    from langchain.schema import Document
    from langchain.prompts import PromptTemplate
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    print("⚠ langchain 未安装，使用简化版本")

# ====== 配置 ======
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # my_intern/
DATA_DIR = PROJECT_ROOT / "data" / "sfks"
BERT_MODEL_PATH = PROJECT_ROOT / "week2_baseline" / "models" / "sfks_bert_lora_v4_improved"
CHROMA_DB_PATH = PROJECT_ROOT / "week2_baseline" / "vector_stores" / "chroma_sfks_db"


# ====== 检索器接口 ======
@dataclass
class RetrievalResult:
    """检索结果"""
    doc_id: str
    content: str
    score: float
    source: str  # "vector", "bm25", "hybrid"
    metadata: Dict = None


class BaseRetriever(ABC):
    """检索器基类"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        pass


class SimpleHybridRetriever(BaseRetriever):
    """简化版混合检索器（不依赖 jieba）"""

    def __init__(self, chroma_db_path: Path, alpha: float = 0.6):
        self.alpha = alpha
        self.documents = []
        self.doc_ids = []
        self.collection = None

        self._load_chroma(chroma_db_path)
        self._build_simple_index()

    def _load_chroma(self, db_path: Path):
        """加载 Chroma"""
        if not db_path.exists():
            print(f"  ⚠ Chroma 不存在: {db_path}")
            return

        try:
            client = chromadb.PersistentClient(path=str(db_path))
            # 尝试不同的 collection 名称
            for name in ["sfks_exams", "sfks_laws", "cail_cases"]:
                try:
                    self.collection = client.get_collection(name)
                    break
                except:
                    continue

            if self.collection:
                all_data = self.collection.get()
                self.doc_ids = all_data["ids"]
                self.documents = all_data["documents"]
                print(f"  ✓ 加载 {len(self.documents)} 条文档")
        except Exception as e:
            print(f"  ⚠ Chroma 加载失败: {e}")

    def _build_simple_index(self):
        """构建简单的关键词索引"""
        self.keyword_index = {}
        for i, doc in enumerate(self.documents):
            # 简单分词：按字符
            for char in doc:
                if '\u4e00' <= char <= '\u9fff':  # 中文
                    if char not in self.keyword_index:
                        self.keyword_index[char] = []
                    self.keyword_index[char].append(i)

    def _keyword_search(self, query: str, top_k: int = 10) -> List[tuple]:
        """简单关键词搜索"""
        scores = {}
        for char in query:
            if char in self.keyword_index:
                for doc_idx in self.keyword_index[char]:
                    scores[doc_idx] = scores.get(doc_idx, 0) + 1

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """混合检索"""
        results = []

        # 向量检索
        if self.collection:
            try:
                vector_results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k * 2, len(self.documents))
                )

                if vector_results["ids"]:
                    for i, (doc_id, doc, dist) in enumerate(zip(
                        vector_results["ids"][0],
                        vector_results["documents"][0],
                        vector_results["distances"][0]
                    )):
                        results.append(RetrievalResult(
                            doc_id=doc_id,
                            content=doc,
                            score=1.0 / (1.0 + dist),
                            source="hybrid"
                        ))
            except:
                pass

        # 如果没有结果，用关键词搜索
        if not results and self.documents:
            kw_results = self._keyword_search(query, top_k)
            for doc_idx, score in kw_results:
                results.append(RetrievalResult(
                    doc_id=self.doc_ids[doc_idx] if doc_idx < len(self.doc_ids) else f"doc_{doc_idx}",
                    content=self.documents[doc_idx],
                    score=score / len(query),
                    source="keyword"
                ))

        return results[:top_k]


# ====== LLM 接口 ======
class BaseLLM(ABC):
    """LLM 基类"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def predict_choice(
        self,
        question: str,
        options: Dict[str, str],
        context: str = ""
    ) -> Dict:
        pass


class BERTMultipleChoiceLLM(BaseLLM):
    """使用 BERT 做多选题（作为 LLM 的替代）"""

    def __init__(self, model_path: Path, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """加载 BERT + LoRA 模型"""
        print(f"  加载 BERT 模型: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        lora_path = model_path / "lora_adapter"
        if lora_path.exists():
            config_path = lora_path / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                base_model_name = config.get("base_model_name_or_path", "hfl/chinese-bert-wwm-ext")
            else:
                base_model_name = "hfl/chinese-bert-wwm-ext"

            base_model = AutoModelForMultipleChoice.from_pretrained(base_model_name)
            self.model = PeftModel.from_pretrained(base_model, str(lora_path))
        else:
            self.model = AutoModelForMultipleChoice.from_pretrained(str(model_path))

        self.model.to(self.device)
        self.model.eval()
        print(f"  ✓ 模型加载完成，设备: {self.device}")

    def generate(self, prompt: str) -> str:
        """BERT 不支持生成，返回空"""
        return ""

    def predict_choice(
        self,
        question: str,
        options: Dict[str, str],
        context: str = ""
    ) -> Dict:
        """预测选择题答案"""
        # 构建输入
        if context:
            full_question = f"[参考] {context[:400]} [问题] {question}"
        else:
            full_question = question

        first_sentences = []
        second_sentences = []

        for key in ["A", "B", "C", "D"]:
            first_sentences.append(full_question)
            second_sentences.append(f"{key}. {options.get(key, '')}")

        inputs = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.unsqueeze(0).to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_idx = torch.argmax(probs).item()
        reverse_map = {0: "A", 1: "B", 2: "C", 3: "D"}

        return {
            "predicted": reverse_map[pred_idx],
            "confidence": probs[pred_idx].item(),
            "all_probs": {reverse_map[i]: probs[i].item() for i in range(4)}
        }


# ====== RAG 流水线 ======
class RAGPipeline:
    """
    完整 RAG 问答流水线

    流程:
    1. 接收问题
    2. 混合检索相关文档
    3. 构建增强 prompt
    4. LLM 生成/预测答案
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        top_k: int = 3
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

        # Prompt 模板
        self.prompt_template = """你是一个专业的法律问答助手。请根据以下参考资料回答问题。

参考资料:
{context}

问题: {question}

选项:
{options}

请选择最正确的答案（A/B/C/D）:"""

    def answer(
        self,
        question: str,
        options: Dict[str, str],
        use_retrieval: bool = True
    ) -> Dict:
        """
        回答问题

        Args:
            question: 问题文本
            options: 选项 {"A": "...", ...}
            use_retrieval: 是否使用检索增强

        Returns:
            {
                "answer": "A",
                "confidence": 0.85,
                "retrieved_docs": [...],
                "latency_ms": 123
            }
        """
        start_time = time.time()

        # 1. 检索
        retrieved_docs = []
        context = ""
        if use_retrieval and self.retriever:
            results = self.retriever.retrieve(question, self.top_k)
            retrieved_docs = [r.content for r in results]
            context = " ".join(retrieved_docs[:2])  # 只用前2个

        # 2. 预测
        result = self.llm.predict_choice(question, options, context)

        latency = (time.time() - start_time) * 1000

        return {
            "answer": result["predicted"],
            "confidence": result["confidence"],
            "all_probs": result.get("all_probs", {}),
            "retrieved_docs": retrieved_docs,
            "use_retrieval": use_retrieval,
            "latency_ms": latency
        }


# ====== 评估系统 ======
class RAGEvaluator:
    """RAG 系统评估器"""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.metrics = {}

    def evaluate(
        self,
        test_data: List[Dict],
        use_retrieval: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        评估 RAG 系统

        Args:
            test_data: [{"question": str, "options": dict, "answer": str}, ...]
            use_retrieval: 是否使用检索
            verbose: 是否打印进度
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix
        )

        y_true = []
        y_pred = []
        latencies = []
        confidences = []

        if verbose:
            print(f"\n评估 {len(test_data)} 条数据...")

        for i, item in enumerate(test_data):
            if verbose and (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(test_data)}")

            result = self.pipeline.answer(
                question=item["question"],
                options=item["options"],
                use_retrieval=use_retrieval
            )

            y_true.append(item["answer"])
            y_pred.append(result["answer"])
            latencies.append(result["latency_ms"])
            confidences.append(result["confidence"])

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

        # 混淆矩阵
        cm = confusion_matrix(y_true_idx, y_pred_idx)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_weighted": float(f1),
            "f1_macro": float(f1_macro),
            "avg_latency_ms": float(np.mean(latencies)),
            "avg_confidence": float(np.mean(confidences)),
            "total_samples": len(test_data),
            "use_retrieval": use_retrieval,
            "confusion_matrix": cm.tolist()
        }

    def compare_with_baseline(
        self,
        test_data: List[Dict]
    ) -> Dict:
        """对比 RAG 与基线模型"""
        print("\n" + "=" * 60)
        print("RAG vs 基线模型对比评估")
        print("=" * 60)

        # 1. 基线（无检索）
        print("\n[1/2] 评估基线模型（无检索）...")
        baseline_results = self.evaluate(test_data, use_retrieval=False)

        # 2. RAG（有检索）
        print("\n[2/2] 评估 RAG 系统（有检索）...")
        rag_results = self.evaluate(test_data, use_retrieval=True)

        # 对比
        print("\n" + "=" * 60)
        print("评估结果对比")
        print("=" * 60)
        print(f"\n{'指标':<20} {'基线':<15} {'RAG':<15} {'差异':<15}")
        print("-" * 60)

        for metric in ["accuracy", "f1_weighted", "f1_macro", "avg_latency_ms"]:
            baseline_val = baseline_results[metric]
            rag_val = rag_results[metric]
            diff = rag_val - baseline_val

            if metric == "avg_latency_ms":
                print(f"{metric:<20} {baseline_val:<15.2f} {rag_val:<15.2f} {diff:+.2f}")
            else:
                print(f"{metric:<20} {baseline_val:<15.4f} {rag_val:<15.4f} {diff:+.4f}")

        improvement = rag_results["f1_weighted"] - baseline_results["f1_weighted"]
        if improvement > 0:
            print(f"\n✓ RAG 提升了 F1: +{improvement:.4f}")
        else:
            print(f"\n✗ RAG 未提升 F1: {improvement:.4f}")

        return {
            "baseline": baseline_results,
            "rag": rag_results,
            "improvement": {
                "accuracy": rag_results["accuracy"] - baseline_results["accuracy"],
                "f1_weighted": improvement,
                "f1_macro": rag_results["f1_macro"] - baseline_results["f1_macro"],
            }
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
    print("\n[1/4] 初始化混合检索器...")
    retriever = SimpleHybridRetriever(CHROMA_DB_PATH)

    print("\n[2/4] 初始化 LLM (BERT 分类器)...")
    if BERT_MODEL_PATH.exists():
        llm = BERTMultipleChoiceLLM(BERT_MODEL_PATH)
    else:
        print(f"  ⚠ 模型不存在: {BERT_MODEL_PATH}")
        print("  请先运行 sfks_bert_v4_improved.py 训练模型")
        return

    print("\n[3/4] 构建 RAG 流水线...")
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        top_k=3
    )

    print("\n[4/4] 加载测试数据...")
    test_data = load_test_data(DATA_DIR, max_samples=200)
    print(f"  加载 {len(test_data)} 条测试数据")

    # 评估
    evaluator = RAGEvaluator(pipeline)
    comparison = evaluator.compare_with_baseline(test_data)

    # 保存结果
    output_path = SCRIPT_DIR / "rag_pipeline_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        # 移除不可序列化的字段
        comparison["baseline"].pop("confusion_matrix", None)
        comparison["rag"].pop("confusion_matrix", None)
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 评估结果保存至: {output_path}")

    # 交互式演示
    print("\n" + "=" * 60)
    print("交互式演示")
    print("=" * 60)

    if test_data:
        sample = test_data[0]
        print(f"\n问题: {sample['question'][:100]}...")
        print("选项:")
        for k, v in sample['options'].items():
            print(f"  {k}. {v[:50]}...")

        result = pipeline.answer(
            sample['question'],
            sample['options'],
            use_retrieval=True
        )

        print(f"\n预测答案: {result['answer']} (置信度: {result['confidence']:.2%})")
        print(f"正确答案: {sample['answer']}")
        print(f"延迟: {result['latency_ms']:.2f} ms")

        if result['retrieved_docs']:
            print(f"\n检索到 {len(result['retrieved_docs'])} 条相关文档:")
            for i, doc in enumerate(result['retrieved_docs'][:2], 1):
                print(f"  [{i}] {doc[:80]}...")


if __name__ == "__main__":
    main()
