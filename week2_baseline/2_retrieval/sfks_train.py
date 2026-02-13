# -*- coding: utf-8 -*-
"""
CAIL2020 司法考试数据集向量化脚本
功能：将司法考试问答数据转换为Youtu-Embedding向量并保存
数据流：
  司法考试 0_train.json → 问题+选项文本 → Youtu向量编码 → 保存为 .npy + .json
"""
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer


class LLMEmbeddingModel:
    """本地Youtu-Embedding模型封装"""
    def __init__(self,
                 model_name_or_path,
                 batch_size=128,
                 max_length=1024,
                 gpu_id=0):
        """初始化本地embedding模型"""
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right", trust_remote_code=True)

        # 设备选择: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size

        query_instruction = "Given a search query, retrieve passages that answer the question"
        if query_instruction:
            self.query_instruction = f"Instruction: {query_instruction} \nQuery:"
        else:
            self.query_instruction = "Query:"
        self.doc_instruction = ""

        print(f"✓ 模型已加载: {model_name_or_path}")
        print(f"✓ 计算设备: {self.device}")

    def mean_pooling(self, hidden_state, attention_mask):
        """均值池化计算sentence embedding"""
        s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        embedding = s / d
        return embedding

    @torch.no_grad()
    def encode(self, sentences_batch, instruction):
        """对文本批次进行编码"""
        inputs = self.tokenizer(
            sentences_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            add_special_tokens=True,
        )

        # 移到指定设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs[0]

            instruction_tokens = self.tokenizer(
                instruction,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )["input_ids"]

            if len(np.shape(np.array(instruction_tokens))) == 1:
                inputs["attention_mask"][:, :len(instruction_tokens)] = 0
            else:
                instruction_length = [len(item) for item in instruction_tokens]
                assert len(instruction) == len(sentences_batch)
                for idx in range(len(instruction_length)):
                    inputs["attention_mask"][idx, :instruction_length[idx]] = 0

            embeddings = self.mean_pooling(last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        return embeddings

    def encode_queries(self, queries):
        """对查询文本编码"""
        queries = queries if isinstance(queries, list) else [queries]
        queries = [f"{self.query_instruction}{query}" for query in queries]
        return self.encode(queries, self.query_instruction)

    def encode_passages(self, passages):
        """对文档文本编码"""
        passages = passages if isinstance(passages, list) else [passages]
        passages = [f"{self.doc_instruction}{passage}" for passage in passages]
        return self.encode(passages, self.doc_instruction)


def load_sfks_data(jsonl_file: Path, sample_size: int = None):
    """
    从JSONL文件加载司法考试数据（0_train.json格式）

    数据格式示例：
    {
      "answer": ["B"],
      "id": "1_4269",
      "option_list": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "statement": "题目内容...",
      "subject": "国际经济法",
      "type": "0"
    }

    Args:
        jsonl_file: JSONL格式的数据文件路径
        sample_size: 采样数量，None表示加载全部
    Returns:
        questions: [str, ...]  问题列表
        passages: [str, ...]   问题+选项组合列表
        meta: [dict, ...]      元数据列表
    """
    questions = []
    passages = []
    meta = []
    count = 0

    with jsonl_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            # 提取关键字段（司法考试数据格式）
            item_id = item.get("id", "")
            question = item.get("statement", "")  # ✅ 修复：司法考试用 "statement" 字段
            option_list = item.get("option_list", {})  # ✅ 修复：dict 格式 {"A": "...", "B": "..."}
            answer = item.get("answer", [])  # ✅ 修复：list 格式 ["B"]
            subject = item.get("subject", "未分类")
            item_type = item.get("type", "0")

            if not question or not option_list:
                continue

            # 构造 passage：问题 + 所有选项
            # 格式: "问题：[题目内容]\n选项：\nA: [选项A]\nB: [选项B]\n..."
            options_text = "\n".join([
                f"{key}: {value}"
                for key, value in sorted(option_list.items())
            ])
            passage = f"问题：{question}\n选项：\n{options_text}"

            # 构造元数据
            metadata = {
                "id": item_id,
                "question": question,
                "answer": answer if isinstance(answer, list) else [answer],
                "subject": subject,
                "type": item_type,
            }

            questions.append(question)
            passages.append(passage)
            meta.append(metadata)

            count += 1
            if sample_size and count >= sample_size:
                break

    return questions, passages, meta


def encode_passages_in_batches(model: LLMEmbeddingModel, passages, batch_size: int = 32):
    """
    用模型按批次编码所有passages
    Args:
        model: LLMEmbeddingModel实例
        passages: 文本列表
        batch_size: 批大小
    Returns:
        np.ndarray: shape [N, dim]
    """
    all_embeddings = []
    for i in tqdm(range(0, len(passages), batch_size), desc="编码passages"):
        batch = passages[i : i + batch_size]
        emb = model.encode_passages(batch)  # torch.Tensor [B, dim]
        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings


def main():
    print("=" * 70)
    print("CAIL2020 司法考试数据集向量化")
    print("=" * 70)

    # 1. 路径配置
    project_root = Path(__file__).parent  # retrieval_data 目录

    # ✅ 修复：使用司法考试原始数据集
    data_file = project_root.parent.parent / "data" / "sfks" / "0_train.json"

    # youtu 本地模型路径
    youtu_model_dir = Path("/home/scyjf4/work/intern/youtu-model").expanduser().resolve()

    # 输出目录
    output_dir = project_root / "sfks_vectors"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 加载司法考试数据 - 前2000个（可调整）
    print(f"\n[1/4] 加载司法考试数据: {data_file}")
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    questions, passages, meta = load_sfks_data(data_file, sample_size=4000)
    print(f"✓ 加载了 {len(passages)} 条司法考试问答对")

    if not passages:
        raise ValueError("没有加载到有效的司法考试数据")

    # 打印示例
    print("\n数据示例（第1条）:")
    print(f"  问题: {questions[0][:60]}...")
    print(f"  科目: {meta[0]['subject']}")
    print(f"  答案: {meta[0]['answer']}")

    # 3. 加载 Youtu-Embedding 模型
    print(f"\n[2/4] 加载Youtu-Embedding模型: {youtu_model_dir}")
    if not youtu_model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {youtu_model_dir}")

    model = LLMEmbeddingModel(
        model_name_or_path=str(youtu_model_dir),
        batch_size=16,
        max_length=512,
        gpu_id=0,
    )

    # 4. 批量编码
    print(f"\n[3/4] 编码问题+选项组合")
    embeddings = encode_passages_in_batches(model, passages, batch_size=32)
    print(f"✓ 得到向量矩阵形状: {embeddings.shape}")

    # 5. 保存向量与元信息
    print(f"\n[4/4] 保存结果")
    emb_path = output_dir / "sfks_embeddings.npy"
    meta_path = output_dir / "sfks_meta.json"
    questions_path = output_dir / "sfks_questions.json"

    np.save(emb_path, embeddings)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with questions_path.open("w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f, ensure_ascii=False, indent=2)

    print(f"✓ 向量已保存: {emb_path}")
    print(f"✓ 元数据已保存: {meta_path}")
    print(f"✓ 问题已保存: {questions_path}")

    print("\n" + "=" * 70)
    print(f"总结:")
    print(f"  - 处理样本数: {len(passages)}")
    print(f"  - 向量维度: {embeddings.shape[1]}")
    print(f"  - 输出目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
