# -*- coding: utf-8 -*-
"""Small-sample runner for `retrieve_for_sfks.py`.

This script keeps the exact same retrieval + augmentation logic, but only
processes a tiny number of Q/A pairs for quick smoke testing.

Outputs:
- qa_train_retrieved_sample.jsonl (in this directory by default)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tqdm import tqdm
import chromadb

# 使用相对导入
from ydlj_query import _YoutuEmbeddingClient


BASE_DIR = Path(__file__).resolve().parent


def retrieve_and_augment_sample(
    sample_size: int = 2000,
    input_file: str | os.PathLike = BASE_DIR / "qa_train.jsonl",
    output_file: str | os.PathLike = BASE_DIR / "qa_train_retrieved_sample.jsonl",
    top_k: int = 3,
    chroma_dir: str | os.PathLike = BASE_DIR.parent / "vector_stores" / "chroma_ydlj_db",
    collection_name: str = "cail_cases",
):
    input_file = Path(input_file)
    output_file = Path(output_file)
    chroma_dir = Path(chroma_dir)

    print(f"正在加载问题文件: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # 只取前 sample_size 条做验证
    data = data[:sample_size]
    questions = [item["question"] for item in data]

    print(f"共加载 {len(questions)} 条问题（sample={sample_size}），准备进行检索...")

    encoder = _YoutuEmbeddingClient()

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name=collection_name, embedding_function=None)
    print(f"成功连接向量库 '{collection_name}'，包含 {collection.count()} 条文书。")

    # 小样本就不分 batch 也行，但为了和主脚本一致，保持 batch 逻辑
    batch_size = 16
    all_retrieved_docs: list[list[str]] = []

    for i in tqdm(range(0, len(questions), batch_size), desc="Retrieving(sample)"):
        batch_qs = questions[i : i + batch_size]
        q_embs = encoder.encode_queries(batch_qs)
        results = collection.query(query_embeddings=q_embs.tolist(), n_results=top_k)
        all_retrieved_docs.extend(results["documents"])

    print(f"检索完成，正在生成增强数据: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for item, retrieved_texts in zip(data, all_retrieved_docs):
            docs_text = "\n".join([f"《参考文书{j + 1}》: {txt}" for j, txt in enumerate(retrieved_texts)])
            original_context = item["context"]
            item["context"] = f"【参考资料】\n{docs_text}\n\n【题目与选项】\n{original_context}"
            item["retrieved_docs"] = retrieved_texts
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"处理完成！输出文件: {output_file}")


if __name__ == "__main__":
    retrieve_and_augment_sample(sample_size=2000)
