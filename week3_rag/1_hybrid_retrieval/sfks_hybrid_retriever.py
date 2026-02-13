# -*- coding: utf-8 -*-
"""
方案2: 混合检索 (向量 + BM25)
结合 Youtu-Embedding 语义检索和 BM25 关键词检索
提升相关上下文召回率
"""
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
import chromadb

# ====== 配置 ======
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # my_intern/
CHROMA_DB_PATH = PROJECT_ROOT / "week2_baseline" / "vector_stores" / "chroma_sfks_db"


class BM25Retriever:
    """BM25 关键词检索器"""

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_count = len(documents)

        # 分词并建立索引
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.avg_doc_len = sum(len(doc) for doc in self.tokenized_docs) / self.doc_count

        # 计算 IDF
        self.idf = self._compute_idf()

        print(f"  ✓ BM25 索引构建完成: {self.doc_count} 文档")

    def _tokenize(self, text: str) -> List[str]:
        """简单中文分词 (基于字符)"""
        # 移除标点，按字符分割
        text = re.sub(r'[^\w\s]', '', text)
        # 对于中文，按字符分割；对于英文，按空格分割
        tokens = []
        current_word = ""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符
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
        """计算每个词的 IDF"""
        df = Counter()
        for doc in self.tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] += 1

        idf = {}
        for term, freq in df.items():
            idf[term] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def score(self, query: str, doc_idx: int) -> float:
        """计算单个文档的 BM25 分数"""
        query_tokens = self._tokenize(query)
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = len(doc_tokens)

        tf = Counter(doc_tokens)
        score = 0.0

        for term in query_tokens:
            if term not in self.idf:
                continue
            term_freq = tf.get(term, 0)
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += self.idf[term] * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """搜索并返回 top_k 结果"""
        scores = [(i, self.score(query, i)) for i in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    """混合检索器: 向量检索 + BM25"""

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: 向量检索权重 (1-alpha 为 BM25 权重)
        """
        self.alpha = alpha
        self.vector_client = None
        self.vector_collection = None
        self.bm25 = None
        self.documents = []
        self.ids = []
        self.metadatas = []

    def load_from_chroma(self, db_path: Path, collection_name: str = "sfks_exams"):
        """从 Chroma 加载向量库并构建 BM25 索引"""
        print(f"\n[加载混合检索器]")
        print(f"  - Chroma 路径: {db_path}")

        # 加载 Chroma
        self.vector_client = chromadb.PersistentClient(path=str(db_path))
        self.vector_collection = self.vector_client.get_collection(
            name=collection_name,
            embedding_function=None
        )

        # 获取所有文档
        all_data = self.vector_collection.get(include=["documents", "metadatas"])
        self.ids = all_data["ids"]
        self.documents = all_data["documents"]
        self.metadatas = all_data["metadatas"]

        print(f"  ✓ 加载 {len(self.documents)} 条文档")

        # 构建 BM25 索引
        print(f"  - 构建 BM25 索引...")
        self.bm25 = BM25Retriever(self.documents)

        print(f"  ✓ 混合检索器初始化完成")

    def search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        """
        混合检索

        Args:
            query: 查询文本
            query_embedding: 查询向量
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        # 1. 向量检索
        vector_results = self.vector_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 2, len(self.documents))  # 多取一些用于融合
        )

        vector_ids = vector_results["ids"][0]
        vector_distances = vector_results["distances"][0]

        # 归一化向量分数 (距离转相似度)
        max_dist = max(vector_distances) if vector_distances else 1
        vector_scores = {
            vid: 1 - (dist / max_dist)
            for vid, dist in zip(vector_ids, vector_distances)
        }

        # 2. BM25 检索
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        # 归一化 BM25 分数
        max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1
        bm25_scores = {
            self.ids[idx]: score / max_bm25 if max_bm25 > 0 else 0
            for idx, score in bm25_results
        }

        # 3. 融合分数
        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        combined_scores = {}

        for doc_id in all_ids:
            v_score = vector_scores.get(doc_id, 0)
            b_score = bm25_scores.get(doc_id, 0)
            combined_scores[doc_id] = self.alpha * v_score + (1 - self.alpha) * b_score

        # 4. 排序并返回
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]

        results = []
        for doc_id in sorted_ids:
            idx = self.ids.index(doc_id)
            results.append({
                "id": doc_id,
                "document": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": combined_scores[doc_id],
                "vector_score": vector_scores.get(doc_id, 0),
                "bm25_score": bm25_scores.get(doc_id, 0),
            })

        return results


def test_hybrid_retriever():
    """测试混合检索器"""
    print("=" * 60)
    print("混合检索测试 (向量 + BM25)")
    print("=" * 60)

    # 初始化
    retriever = HybridRetriever(alpha=0.6)  # 60% 向量, 40% BM25
    retriever.load_from_chroma(CHROMA_DB_PATH)

    # 测试查询 (需要 Youtu 模型生成向量)
    print("\n[测试查询]")

    # 这里用一个简单的随机向量测试结构
    # 实际使用时应该用 Youtu 模型生成
    dummy_embedding = np.random.randn(2048).tolist()

    test_queries = [
        "合同无效的情形有哪些？",
        "正当防卫的构成要件",
        "公司法人代表的责任",
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        results = retriever.search(query, dummy_embedding, top_k=3)

        for i, res in enumerate(results):
            print(f"  [{i+1}] 综合分数: {res['score']:.4f} "
                  f"(向量: {res['vector_score']:.4f}, BM25: {res['bm25_score']:.4f})")
            print(f"      问题: {res['metadata'].get('question', '')[:60]}...")


if __name__ == "__main__":
    test_hybrid_retriever()
