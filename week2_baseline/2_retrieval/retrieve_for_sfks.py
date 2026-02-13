# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

from tqdm import tqdm
import chromadb

# 使用相对导入，而不是绝对路径
from ydlj_query import _YoutuEmbeddingClient


# 脚本所在目录：所有相对路径都统一从这里解析，避免因为运行目录不同导致 FileNotFoundError
BASE_DIR = Path(__file__).resolve().parent


def retrieve_and_augment(
        input_file: str | os.PathLike = BASE_DIR / "qa_train.jsonl",
        output_file: str | os.PathLike = BASE_DIR / "qa_train_retrieved.jsonl",
        top_k: int = 3,
        chroma_dir: str | os.PathLike = BASE_DIR.parent / "vector_stores" / "chroma_ydlj_db",
        collection_name: str = "cail_cases",
):
    input_file = Path(input_file)
    output_file = Path(output_file)
    chroma_dir = Path(chroma_dir)

    print(f"正在加载问题文件: {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(
            f"找不到输入文件: {input_file}\n"
            f"当前脚本目录: {BASE_DIR}\n"
            f"请确认 qa_train.jsonl 是否在该目录下，或传入正确的 input_file 参数。"
        )

    with open(input_file, 'r', encoding='utf-8') as f:
        # jsonl 格式逐行读取
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    questions = [item['question'] for item in data]
    print(f"共加载 {len(questions)} 条问题，准备进行检索...")

    # 1. 初始化 Embedding 模型 (复用你现有的 Youtu 客户端)
    # 注意：确保你的 ydlj_query.py 里路径配置正确，或者在这里手动指定 model_dir
    encoder = _YoutuEmbeddingClient()

    # 2. 连接 Chroma 向量库（路径统一从脚本目录解析）
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"找不到 Chroma 持久化目录: {chroma_dir}\n"
            f"请先在 {BASE_DIR} 下生成 chroma_ydlj_db（例如运行 ydlj_chroma.py）。"
        )

    client = chromadb.PersistentClient(path=str(chroma_dir))
    # 明确禁用 embedding_function，避免 chroma 试图内部再算 embedding
    collection = client.get_collection(name=collection_name, embedding_function=None)

    print(f"成功连接向量库 '{collection_name}'，包含 {collection.count()} 条文书。")

    # 3. 批量检索 (Batch Retrieval)
    batch_size = 32  # 根据显存调整
    all_retrieved_docs = []

    for i in tqdm(range(0, len(questions), batch_size), desc="Retrieving"):
        batch_qs = questions[i: i + batch_size]

        # 3.1 编码 Queries
        # 调用 ydlj_query.py 中的 encode_queries 方法
        q_embs = encoder.encode_queries(batch_qs)
        q_embs_list = q_embs.tolist()

        # 3.2 在 Chroma 中检索
        results = collection.query(
            query_embeddings=q_embs_list,
            n_results=top_k
        )

        # results['documents'] 是 list of list，对应 batch 里的每个 query
        batch_docs = results['documents']
        all_retrieved_docs.extend(batch_docs)

    # 4. 组装数据并保存
    print(f"检索完成，正在生成增强数据: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for item, retrieved_texts in zip(data, all_retrieved_docs):
            # === 核心逻辑：把检索到的文书拼接到 Context 中 ===

            # 格式：[参考资料] 文书1... 文书2... [题目] 原始Context
            docs_text = "\n".join([f"《参考文书{j + 1}》: {txt}" for j, txt in enumerate(retrieved_texts)])

            original_context = item['context']  # 这里原本是：题干 + 选项

            # 拼接新的 Prompt/Context
            new_context = f"【参考资料】\n{docs_text}\n\n【题目与选项】\n{original_context}"

            # 更新字段
            item['context'] = new_context
            # 保留检索到的原文以便 debug
            item['retrieved_docs'] = retrieved_texts

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"处理完成！输出文件: {output_file}")


if __name__ == "__main__":
    # 确保你的路径是正确的（这里也用脚本目录来判断）
    chroma_path = BASE_DIR.parent / "vector_stores" / "chroma_ydlj_db"
    if not chroma_path.exists():
        print(f"警告：找不到 {chroma_path} 目录，请检查向量库路径！")

    retrieve_and_augment()