# -*- coding: utf-8 -*-
"""
CAIL2020 司法考试数据集 Chroma 向量库构建脚本
将 sfks_train.py 生成的向量导入 Chroma 持久化数据库
"""
import json
from pathlib import Path
import numpy as np
import chromadb


def main():
    print("=" * 70)
    print("司法考试向量库构建（Chroma）")
    print("=" * 70)

    BASE_DIR = Path(__file__).resolve().parent

    # 1. 读取向量和元信息
    print("\n[1/3] 加载向量数据...")

    # 向量文件在 vector_stores/sfks_vectors/ 目录下
    embeddings_path = BASE_DIR.parent / "vector_stores" / "sfks_vectors" / "sfks_embeddings.npy"
    meta_path = BASE_DIR.parent / "vector_stores" / "sfks_vectors" / "sfks_meta.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"找不到向量文件: {embeddings_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到元数据文件: {meta_path}")

    embeddings = np.load(embeddings_path)
    with meta_path.open("r", encoding="utf-8") as f:
        metas = json.load(f)

    print(f"✓ 向量矩阵: {embeddings.shape}")
    print(f"✓ 元数据: {len(metas)} 条")

    assert len(embeddings) == len(metas), f"向量与元数据数量不匹配: embeddings={len(embeddings)}, metas={len(metas)}"

    # 2. 创建 Chroma 客户端
    print("\n[2/3] 初始化 Chroma 数据库...")
    chroma_path = BASE_DIR.parent / "vector_stores" / "chroma_sfks_db"
    client = chromadb.PersistentClient(path=str(chroma_path))

    # 删除旧 collection（避免重复）
    try:
        client.delete_collection(name="sfks_exams")
        print("✓ 已删除旧 collection")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="sfks_exams",
        embedding_function=None,  # 我们已有向量，不需要自动生成
    )

    # 3. 准备数据
    print("\n[3/3] 写入向量库...")
    ids = []
    documents = []
    metadatas = []

    for i, m in enumerate(metas):
        # 构建文档文本（问题）
        item_id = str(m.get("id", str(i)))  # 确保是字符串
        question = str(m.get("question", ""))

        # 获取答案列表
        answer = m.get("answer", [])
        if isinstance(answer, list):
            answer_str = ",".join(str(a) for a in answer)
        else:
            answer_str = str(answer)

        # 元数据（Chroma 要求所有值都是 str/int/float/bool）
        flat_meta = {
            "id": item_id,
            "question": question[:500],  # 限制长度
            "answer": answer_str,
            "subject": str(m.get("subject", "未分类")),
            "type": str(m.get("type", "0")),
        }

        ids.append(item_id)
        documents.append(question)  # 用问题作为文档
        metadatas.append(flat_meta)

    print(f"  - 准备写入 {len(ids)} 条数据")

    # 4. 批量写入
    batch_size = 500
    total = len(ids)
    for i in range(0, total, batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_docs = documents[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        batch_embs = embeddings[i : i + batch_size].tolist()

        try:
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embs,
            )
            print(f"  - 已写入 {min(i + batch_size, total)}/{total}")
        except Exception as e:
            print(f"  ✗ 写入失败 (batch {i//batch_size + 1}): {e}")
            raise

    print(f"\n✓ 向量库构建完成!")
    print(f"✓ 总文档数: {collection.count()}")
    print(f"✓ 存储路径: {chroma_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
