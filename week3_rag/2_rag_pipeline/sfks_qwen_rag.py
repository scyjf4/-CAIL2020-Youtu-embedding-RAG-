# -*- coding: utf-8 -*-
"""
CAIL2020 司法考试 RAG 系统 - 基于 Qwen2-7B
使用 Int4 量化版本，适配 8GB 显存

架构:
1. 混合检索器 (向量 + BM25)
2. Qwen2-7B-Instruct-Int4 作为生成模型
3. 完整的问答评估

依赖安装:
pip install transformers>=4.37.0 accelerate bitsandbytes
pip install auto-gptq optimum  # 用于 GPTQ 量化
"""
import os
import json
import torch
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
import re
from collections import Counter
import math

print("=" * 70)
print("CAIL2020 司法考试 RAG 系统 - Qwen2-7B")
print("=" * 70)

# ====== 环境检查 ======
def check_gpu():
    """检查GPU状态"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_mem = (torch.cuda.get_device_properties(0).total_memory -
                   torch.cuda.memory_allocated(0)) / 1024**3
        print(f"  GPU: {gpu_name}")
        print(f"  总显存: {total_mem:.1f} GB")
        print(f"  可用显存: {free_mem:.1f} GB")
        return True
    else:
        print("  ⚠ 未检测到 GPU，将使用 CPU（会很慢）")
        return False

print("\n[环境检查]")
HAS_GPU = check_gpu()

# ====== 配置 ======
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # my_intern/
DATA_DIR = PROJECT_ROOT / "data" / "sfks"
CHROMA_DB_PATH = PROJECT_ROOT / "week2_baseline" / "vector_stores" / "chroma_sfks_db"

# Qwen 模型配置 - 选择适合8GB显存的版本
QWEN_MODELS = {
    "qwen2-7b-int4": "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",  # ~5GB 显存
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",  # ~3GB 显存，无需量化
    "qwen1.5-4b": "Qwen/Qwen1.5-4B-Chat",  # ~4GB 显存
    "qwen2-0.5b": "Qwen/Qwen2-0.5B-Instruct",  # ~1GB 显存，轻量测试
}

# 默认使用 1.5B 版本（更稳定，不需要额外量化依赖）
DEFAULT_MODEL = "qwen2-1.5b"


# ====== BM25 检索器 ======
class BM25Retriever:
    """BM25 关键词检索器"""

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_count = len(documents)

        if self.doc_count == 0:
            print("  ⚠ BM25: 文档列表为空")
            self.tokenized_docs = []
            self.avg_doc_len = 0
            self.idf = {}
            return

        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.avg_doc_len = sum(len(doc) for doc in self.tokenized_docs) / max(self.doc_count, 1)
        self.idf = self._compute_idf()
        print(f"  ✓ BM25 索引: {self.doc_count} 文档")

    def _tokenize(self, text: str) -> List[str]:
        """简单中文分词"""
        text = re.sub(r'[^\w\s]', '', text)
        tokens = []
        current_word = ""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                if current_word:
                    tokens.append(current_word.lower())
                    current_word = ""
                tokens.append(char)
            elif char.isalnum():
                current_word += char
            else:
                if current_word:
                    tokens.append(current_word.lower())
                    current_word = ""
        if current_word:
            tokens.append(current_word.lower())
        return tokens

    def _compute_idf(self) -> Dict[str, float]:
        df = Counter()
        for doc in self.tokenized_docs:
            for term in set(doc):
                df[term] += 1
        return {
            term: math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.documents:
            return []
        query_tokens = self._tokenize(query)
        scores = []
        for i, doc_tokens in enumerate(self.tokenized_docs):
            tf = Counter(doc_tokens)
            doc_len = len(doc_tokens)
            score = 0.0
            for term in query_tokens:
                if term in self.idf:
                    term_freq = tf.get(term, 0)
                    numerator = term_freq * (self.k1 + 1)
                    denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1))
                    score += self.idf[term] * numerator / denominator
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ====== 混合检索器 ======
class HybridRetriever:
    """混合检索器：向量 + BM25"""

    def __init__(self, chroma_db_path: Path, alpha: float = 0.6):
        """
        alpha: 向量检索权重 (0-1)，1-alpha 为 BM25 权重
        """
        self.alpha = alpha
        self.documents = []
        self.doc_ids = []
        self.collection = None
        self.bm25 = None

        self._load_data(chroma_db_path)

    def _load_data(self, db_path: Path):
        """加载检索数据"""
        print("\n[加载检索数据]")

        # 1. 尝试加载 Chroma
        if db_path.exists():
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(db_path))

                for name in ["sfks_exams", "sfks_laws", "cail_cases"]:
                    try:
                        self.collection = client.get_collection(name)
                        result = self.collection.get(include=["documents"])
                        self.documents = result.get("documents", [])
                        self.doc_ids = result.get("ids", [])
                        print(f"  ✓ Chroma 加载成功: {name}, {len(self.documents)} 条")
                        break
                    except:
                        continue
            except Exception as e:
                print(f"  ⚠ Chroma 加载失败: {e}")

        # 2. 如果 Chroma 为空，尝试从原始数据加载
        if not self.documents:
            print("  ⚠ Chroma 为空，尝试从原始数据加载...")
            self._load_from_raw_data()

        # 3. 构建 BM25 索引
        if self.documents:
            self.bm25 = BM25Retriever(self.documents)
        else:
            print("  ⚠ 无可用文档，检索功能受限")

    def _load_from_raw_data(self):
        """从原始JSON文件加载"""
        for json_file in ["0_train.json", "1_train.json"]:
            file_path = DATA_DIR / json_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for item in data[:500]:  # 限制数量
                        statement = item.get("statement", "")
                        options = item.get("option_list", {})
                        doc = f"{statement} " + " ".join(options.values())
                        self.documents.append(doc)
                        self.doc_ids.append(str(len(self.doc_ids)))
                    print(f"  ✓ 从 {json_file} 加载 {len(data[:500])} 条")
                except Exception as e:
                    print(f"  ⚠ 加载 {json_file} 失败: {e}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """混合检索"""
        results = []

        # 向量检索
        vector_results = {}
        if self.collection:
            try:
                res = self.collection.query(query_texts=[query], n_results=top_k)
                for i, (doc_id, doc, dist) in enumerate(zip(
                    res.get("ids", [[]])[0],
                    res.get("documents", [[]])[0],
                    res.get("distances", [[]])[0]
                )):
                    score = 1 / (1 + dist)  # 距离转相似度
                    vector_results[doc_id] = {
                        "content": doc,
                        "vector_score": score,
                        "bm25_score": 0.0
                    }
            except Exception as e:
                print(f"  ⚠ 向量检索失败: {e}")

        # BM25 检索
        bm25_results = {}
        if self.bm25:
            bm25_hits = self.bm25.search(query, top_k=top_k)
            max_bm25 = max([s for _, s in bm25_hits], default=1)
            for idx, score in bm25_hits:
                doc_id = self.doc_ids[idx] if idx < len(self.doc_ids) else str(idx)
                norm_score = score / max(max_bm25, 0.001)
                if doc_id in vector_results:
                    vector_results[doc_id]["bm25_score"] = norm_score
                else:
                    bm25_results[doc_id] = {
                        "content": self.documents[idx],
                        "vector_score": 0.0,
                        "bm25_score": norm_score
                    }

        # 合并结果
        all_results = {**vector_results, **bm25_results}
        for doc_id, info in all_results.items():
            hybrid_score = (self.alpha * info["vector_score"] +
                          (1 - self.alpha) * info["bm25_score"])
            results.append({
                "doc_id": doc_id,
                "content": info["content"],
                "score": hybrid_score,
                "vector_score": info["vector_score"],
                "bm25_score": info["bm25_score"]
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ====== Qwen 模型加载 ======
class QwenRAG:
    """Qwen RAG 问答系统"""

    def __init__(self, model_name: str = DEFAULT_MODEL, retriever: HybridRetriever = None):
        self.model_name = model_name
        self.model_id = QWEN_MODELS.get(model_name, QWEN_MODELS[DEFAULT_MODEL])
        self.retriever = retriever
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载 Qwen 模型"""
        print(f"\n[加载模型] {self.model_id}")
        print(f"  设备: {self.device}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # 清理显存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 加载 tokenizer
            print("  加载 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )

            # 加载模型 - 根据版本选择不同策略
            print("  加载模型（可能需要几分钟下载）...")

            if "int4" in self.model_name.lower() or "gptq" in self.model_id.lower():
                # GPTQ 量化模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
            else:
                # 非量化模型，使用 bfloat16
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    low_cpu_mem_usage=True,
                )

            self.model.eval()

            # 显示显存使用
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1024**3
                print(f"  ✓ 模型加载成功，显存占用: {mem_used:.2f} GB")

            return True

        except Exception as e:
            print(f"  ✗ 模型加载失败: {e}")
            print("\n  尝试解决方案:")
            print("  1. pip install transformers>=4.37.0 accelerate")
            print("  2. 如果是 GPTQ 模型: pip install auto-gptq optimum")
            print("  3. 尝试更小的模型: qwen2-0.5b 或 qwen2-1.5b")
            return False

    def build_prompt(self, question: str, options: Dict[str, str],
                    context: List[Dict] = None) -> str:
        """构建提示词"""
        # 格式化选项
        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])

        # 格式化上下文
        context_text = ""
        if context:
            context_text = "\n\n参考资料:\n"
            for i, ctx in enumerate(context[:3], 1):
                content = ctx.get("content", "")[:300]  # 限制长度
                context_text += f"{i}. {content}\n"

        # 构建完整提示
        prompt = f"""你是一个专业的法律考试助手。请根据以下法律问题和选项，选择最正确的答案。

问题: {question}

选项:
{options_text}
{context_text}
请仔细分析问题，只输出正确答案的选项字母（A、B、C或D），不要输出其他内容。

答案: """

        return prompt

    @torch.no_grad()
    def generate_answer(self, question: str, options: Dict[str, str],
                       use_retrieval: bool = True) -> Dict:
        """生成答案"""
        if self.model is None:
            return {"error": "模型未加载", "answer": None}

        # 检索相关上下文
        context = []
        if use_retrieval and self.retriever:
            context = self.retriever.retrieve(question, top_k=3)

        # 构建提示
        prompt = self.build_prompt(question, options, context)

        # 生成
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # 解码
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # 提取答案
            answer = None
            for char in generated.upper():
                if char in "ABCD":
                    answer = char
                    break

            return {
                "answer": answer,
                "raw_output": generated,
                "context_used": len(context) > 0,
                "context": context[:2] if context else []
            }

        except Exception as e:
            return {"error": str(e), "answer": None}


# ====== 评估函数 ======
def evaluate_rag(rag: QwenRAG, test_data: List[Dict],
                use_retrieval: bool = True) -> Dict:
    """评估 RAG 系统"""
    print(f"\n[评估] 共 {len(test_data)} 条测试数据")
    print(f"  使用检索: {'是' if use_retrieval else '否'}")

    correct = 0
    total = 0
    results = []

    for i, item in enumerate(test_data):
        question = item.get("statement", "")
        options = item.get("option_list", {})
        answer_raw = item.get("answer", "")

        # 处理答案格式
        if isinstance(answer_raw, list):
            if len(answer_raw) != 1:
                continue  # 跳过多选题
            true_answer = answer_raw[0]
        else:
            true_answer = answer_raw

        # 生成答案
        result = rag.generate_answer(question, options, use_retrieval=use_retrieval)
        pred_answer = result.get("answer")

        # 统计
        is_correct = pred_answer == true_answer
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question[:50],
            "true": true_answer,
            "pred": pred_answer,
            "correct": is_correct
        })

        # 打印进度
        if (i + 1) % 10 == 0:
            acc = correct / total * 100
            print(f"  进度: {i+1}/{len(test_data)}, 当前准确率: {acc:.1f}%")

    # 计算指标
    accuracy = correct / max(total, 1)

    print(f"\n[评估结果]")
    print(f"  总样本: {total}")
    print(f"  正确: {correct}")
    print(f"  准确率: {accuracy*100:.2f}%")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "use_retrieval": use_retrieval,
        "details": results[:10]  # 前10条详情
    }


# ====== 主函数 ======
def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="SFKS Qwen RAG")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       choices=list(QWEN_MODELS.keys()),
                       help="Qwen 模型版本")
    parser.add_argument("--samples", type=int, default=50,
                       help="测试样本数")
    parser.add_argument("--no-retrieval", action="store_true",
                       help="禁用检索增强")
    args = parser.parse_args()

    # 1. 初始化检索器
    print("\n" + "=" * 70)
    print("[1/3] 初始化混合检索器")
    print("=" * 70)
    retriever = HybridRetriever(CHROMA_DB_PATH)

    # 2. 加载 Qwen 模型
    print("\n" + "=" * 70)
    print(f"[2/3] 加载 Qwen 模型: {args.model}")
    print("=" * 70)
    rag = QwenRAG(model_name=args.model, retriever=retriever)

    if not rag.load_model():
        print("\n✗ 模型加载失败，退出")
        return

    # 3. 加载测试数据
    print("\n" + "=" * 70)
    print("[3/3] 加载测试数据并评估")
    print("=" * 70)

    test_data = []
    for json_file in ["0_train.json", "1_train.json"]:
        file_path = DATA_DIR / json_file
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 只取单选题
            single_choice = [
                item for item in data
                if isinstance(item.get("answer"), list) and len(item["answer"]) == 1
            ]
            test_data.extend(single_choice)
            print(f"  ✓ 加载 {json_file}: {len(single_choice)} 条单选题")

    if not test_data:
        print("  ✗ 未找到测试数据")
        return

    # 限制样本数
    test_data = test_data[:args.samples]
    print(f"  使用 {len(test_data)} 条进行测试")

    # 4. 评估
    use_retrieval = not args.no_retrieval
    results = evaluate_rag(rag, test_data, use_retrieval=use_retrieval)

    # 5. 保存结果
    output_file = SCRIPT_DIR / "qwen_rag_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 结果保存至: {output_file}")

    # 对比测试（有检索 vs 无检索）
    if use_retrieval and len(test_data) <= 30:
        print("\n" + "=" * 70)
        print("[对比测试] 无检索增强")
        print("=" * 70)
        results_no_rag = evaluate_rag(rag, test_data, use_retrieval=False)

        print("\n" + "=" * 70)
        print("[对比结果]")
        print("=" * 70)
        print(f"  有检索 (RAG): {results['accuracy']*100:.2f}%")
        print(f"  无检索:       {results_no_rag['accuracy']*100:.2f}%")
        diff = (results['accuracy'] - results_no_rag['accuracy']) * 100
        print(f"  RAG 提升:     {diff:+.2f}%")


if __name__ == "__main__":
    main()
