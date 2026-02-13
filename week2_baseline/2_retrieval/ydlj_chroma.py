# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import chromadb


BASE_DIR = Path(__file__).resolve().parent

# 1. 读取向量和元信息（这些是用长文档 passage 编码出来的）
# 向量文件在 vector_stores/ 目录下
embeddings = np.load(BASE_DIR.parent / "vector_stores" / "cail_train_embeddings.npy")  # shape: [N, dim]
with (BASE_DIR.parent / "vector_stores" / "cail_train_meta.json").open("r", encoding="utf-8") as f:
    metas = json.load(f)  # 长度 N 的 list

assert len(embeddings) == len(metas), f"embeddings={len(embeddings)} metas={len(metas)}"


def build_documents_from_train_json(train_json_path: Path) -> dict[int, str]:
    """从 CAIL2020 ydlj/data/train.json 重建每条文书的长文本。

    返回一个 dict：id -> doc_text

    约定（与 youtu-embedding/training/CAIL2020/train/ydlj_train.py 的 load_cail_passages 保持一致）：
    - context[0][0] 为 title
    - context[0][1] 为 sentence list
    - doc_text = title + "\n" + "\n".join(sentences)
    """
    with train_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_text: dict[int, str] = {}
    for item in data:
        cid = item.get("_id")
        context = item.get("context", [])
        if cid is None or not context:
            continue
        try:
            title = context[0][0]
            sentences = context[0][1] if len(context[0]) > 1 else []
        except Exception:
            continue
        doc_text = str(title) + "\n" + "\n".join(map(str, sentences))
        # 有些 _id 可能不是 int，这里尽量转成 int
        try:
            cid_int = int(cid)
        except Exception:
            continue
        id_to_text[cid_int] = doc_text

    return id_to_text


# 2. 从原始训练集重建长文档文本（否则你现在库里只有 title/短句）
# repo_root: .../intern
repo_root = BASE_DIR.parents[2]  # 指向 intern/ 目录
train_json = repo_root / "CAIL2020" / "ydlj" / "data" / "train.json"
if not train_json.exists():
    raise FileNotFoundError(
        f"找不到原始 train.json: {train_json}\n"
        f"需要它来重建长文本 documents（提升检索质量）。"
    )

id_to_doc = build_documents_from_train_json(train_json)

# NOTE: embeddings / metas 的 id 是 0..N-1（见现有 cail_train_meta.json 示例），
# 所以我们用 meta['id'] 去 train.json 里找对应文书全文。

max_chars = 2000  # 过长会影响后续拼接 prompt 的长度；可按需要调整


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...（截断）"


ids = []
documents = []
metadatas = []

missing = 0
for m in metas:
    mid = m.get("id")
    if mid is None:
        continue
    try:
        mid_int = int(mid)
    except Exception:
        continue

    full_text = id_to_doc.get(mid_int)
    if not full_text:
        # 回退到 title（至少别是空）
        full_text = m.get("title", "")
        missing += 1

    ids.append(str(mid_int))
    documents.append(clip(full_text, max_chars))
    metadatas.append(m)

print(f"准备写入 Chroma: {len(ids)} 条 documents，其中 {missing} 条未在 train.json 找到全文（已回退为 title）。")

# 3. 创建 Chroma 客户端和 collection（持久化到磁盘）
chroma_path = BASE_DIR.parent / "vector_stores" / "chroma_ydlj_db"
client = chromadb.PersistentClient(path=str(chroma_path))

# 为避免重复 add 导致 count 翻倍，先删除旧 collection 再重建
try:
    client.delete_collection(name="cail_cases")
    print("已删除旧 collection: cail_cases")
except Exception:
    pass

collection = client.get_or_create_collection(
    name="cail_cases",
    # 注意：我们已经有自己的 embeddings，就不要再让 chroma 调用 embedding_function 了
    embedding_function=None,
)

# 4. 批量写入（可按 batch 分块避免一次性太大）
batch_size = 500
for i in range(0, len(ids), batch_size):
    batch_ids = ids[i : i + batch_size]
    batch_docs = documents[i : i + batch_size]
    batch_metas = metadatas[i : i + batch_size]
    batch_embs = embeddings[i : i + batch_size].tolist()  # Chroma 需要 list[list[float]]

    collection.add(
        ids=batch_ids,
        documents=batch_docs,
        metadatas=batch_metas,
        embeddings=batch_embs,
    )

print("Chroma 向量库构建完成。")
print(f"collection.count() = {collection.count()}")
