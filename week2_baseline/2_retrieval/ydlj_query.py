# -*- coding: utf-8 -*-
import chromadb
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

# 1. 重新加载 Chroma 持久化向量库
# 路径相对于当前脚本，指向 week2_baseline/vector_stores/chroma_ydlj_db
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = BASE_DIR.parent / "vector_stores" / "chroma_ydlj_db"
client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = client.get_collection(name="cail_cases", embedding_function=None)


class _YoutuEmbeddingClient:
    """Thin wrapper around the local Youtu-Embedding checkpoint."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_dir: str | os.PathLike | None = None,
        max_length: int = 1024,
    ) -> None:
        if hasattr(self, "_is_initialized"):
            return
        repo_root = Path(__file__).resolve().parents[3]
        default_model_dir = repo_root / "youtu-model"
        model_dir = Path(model_dir or os.getenv("YOUTU_MODEL_DIR", default_model_dir)).expanduser().resolve()
        if not model_dir.exists():
            raise FileNotFoundError(f"Youtu-Embedding checkpoint not found at {model_dir}")

        self.device = torch.device("cuda")
        self.max_length = max_length
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right", trust_remote_code=True)
        self.query_instruction = "Instruction: Given a search query, retrieve passages that answer the question \nQuery:"
        self.doc_instruction = ""
        self.query_instruction_len = self._instruction_token_length(self.query_instruction)
        self.doc_instruction_len = self._instruction_token_length(self.doc_instruction)
        self._is_initialized = True

    def _instruction_token_length(self, instruction: str) -> int:
        if not instruction:
            return 0
        tokens = self.tokenizer(
            instruction,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )["input_ids"]
        return len(tokens)

    def _mean_pool(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)

    @torch.no_grad()
    def _encode(self, sentences: Sequence[str], instruction_len: int) -> torch.Tensor:
        inputs = self.tokenizer(
            list(sentences),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            add_special_tokens=True,
        ).to(self.device)
        if instruction_len:
            inputs["attention_mask"][:, :instruction_len] = 0
        outputs = self.model(**inputs)
        embeddings = self._mean_pool(outputs[0], inputs["attention_mask"])
        return embeddings.detach().cpu()

    def encode_queries(self, queries: Iterable[str]) -> torch.Tensor:
        queries = list(queries)
        prefixed = [f"{self.query_instruction}{query}" for query in queries]
        return self._encode(prefixed, self.query_instruction_len)

    def encode_documents(self, docs: Iterable[str]) -> torch.Tensor:
        docs = list(docs)
        prefixed = [f"{self.doc_instruction}{doc}" for doc in docs]
        return self._encode(prefixed, self.doc_instruction_len)


def encode_with_youtu(texts: Sequence[str], mode: str = "query") -> torch.Tensor:
    """Encode queries or documents with the local Youtu model."""
    encoder = _YoutuEmbeddingClient()
    if mode == "query":
        return encoder.encode_queries(texts)
    if mode == "document":
        return encoder.encode_documents(texts)
    raise ValueError("mode must be 'query' or 'document'")


def search_cases(query_text, top_k=5):
    # 2.1 把 query 编码成向量
    query_emb = encode_with_youtu([query_text], mode="query")  # shape: [1, dim]

    # 2.2 调用 Chroma 查询
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=top_k
    )
    return results

if __name__ == "__main__":
    q = "交通事故赔偿责任如何分配？"
    res = search_cases(q, top_k=5)
    print(res)
