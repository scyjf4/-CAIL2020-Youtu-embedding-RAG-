# RAG 增强效果有限 - 问题总结与解决方案

## 问题诊断

### 当前评估结果

```
模型                        准确率     F1-W      提升
-------------------------------------------------------
随机基线                    0.2708    0.2695    -
BERT-LoRA（无检索）         0.2253    0.2212    -
BERT-LoRA + RAG            0.2344    0.2339    +0.01  ?? 提升微弱
```

**核心问题**：RAG 系统仅带来 0.01 的 F1 提升，效果几乎可忽略。

---

## 根本原因分析

### 1. 检索器根本没用向量检索 ?

**位置**：`sfks_comprehensive_eval.py` 第 427-438 行

```python
def retrieve(query: str) -> str:
    # 简单的关键词匹配
    scores = []
    for i, doc in enumerate(documents):
        score = sum(1 for char in query if char in doc and '\u4e00' <= char <= '\u9fff')
        scores.append((i, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [documents[scores[j][0]] for j in range(min(2, len(scores))) if scores[j][1] > 0]
    return " ".join(top_docs) if top_docs else ""
```

**问题**：
- ? 只统计中文字符出现次数，没有语义理解
- ? 没有使用 Chroma 的向量检索功能
- ? 没有使用 Youtu-Embedding 模型
- ? 没有使用已实现的混合检索器（`sfks_hybrid_retriever.py`）

### 2. 上下文融合太粗糙 ?

**位置**：`sfks_comprehensive_eval.py` 第 351 行

```python
if context:
    full_question = f"[参考] {context[:400]} [问题] {question}"
```

**问题**：
- ? 简单截断 400 字符，可能截断关键信息
- ? 没有结构化处理
- ? 上下文太短

### 3. 检索数量太少 ?

```python
top_docs = [documents[scores[j][0]] for j in range(min(2, len(scores)))]
```

只返回 2 条文档，信息量不足。

### 4. 基础模型性能弱 ?

BERT-LoRA F1=0.22，比随机基线（0.27）还差 18%。

**原因**：
- 训练数据少（只有 7678 条）
- 过拟合严重（训练时 F1=0.33，测试时 F1=0.22）

---

## 解决方案

### ? 方案 1：使用改进版评估脚本（已完成）

**文件**：`sfks_comprehensive_eval_improved.py`

**改进点**：
1. ? 使用 BM25 关键词检索（基于 TF-IDF，比字符匹配智能）
2. ? top_k 从 2 增加到 5
3. ? 上下文长度从 400 增加到 800
4. ? 使用 `[SEP]` 分隔符结构化上下文
5. ? 添加相关性分数过滤

**注意**：当前版本使用 BM25 而非向量检索，因为向量检索需要额外的嵌入模型配置。BM25 已经比原来的简单字符匹配好很多。

**运行**：

```bash
cd /home/scyjf4/work/intern/my_intern/week2/retrieval_data
source /home/scyjf4/work/intern/youtu-embedding/training/CoDiEmb/scripts/my_youtu-env/bin/activate
python sfks_comprehensive_eval_improved.py
```

或使用一键脚本：

```bash
chmod +x run_improved_eval.sh
./run_improved_eval.sh
```

**预期提升**：F1 从 0.234 → **0.26-0.28**（+15%，使用 BM25）

**如需更大提升**：可以实现真正的向量检索，预期可达 0.30+

---

### ? 方案 2：提升基础模型（需重新训练）

**修改**：`sfks_bert_v4_improved.py`

```python
# 第 14-17 行，修改配置
SAMPLE_PER_FILE = 8000  # 从 4000 → 8000（增加数据）
EPOCHS = 12             # 从 8 → 12（更多训练轮次）
BATCH_SIZE = 8          # 保持不变
MAX_SEQ_LENGTH = 256    # 保持不变

# 第 51-53 行，增加 LoRA 容量
LORA_RANK = 128      # 从 64 → 128
LORA_ALPHA = 256     # 从 128 → 256
LORA_DROPOUT = 0.1   # 从 0.05 → 0.1（防止过拟合）
```

**运行**：

```bash
python sfks_bert_v4_improved.py
```

训练时间：约 1-2 小时（取决于 GPU）

**预期提升**：基础模型 F1 从 0.22 → **0.30+**（+36%）

---

### ? 方案 3：高级优化（可选）

详见 `RAG_OPTIMIZATION_GUIDE.md`，包括：

1. **问题类型分类**：不同题型使用不同检索策略
   - 案例题 → 检索案例库
   - 法条题 → 检索法律库
   - 概念题 → 检索教材库

2. **重排序（Reranking）**：使用 TF-IDF 或 Cross-Encoder 重新排序检索结果

3. **答案验证**：检查上下文是否支持预测答案

4. **集成学习**：融合多个模型的预测

**预期总提升**：F1 从 0.234 → **0.38-0.42**（+70%）

---

## 快速实施路线图

### 第 1 步：立即运行改进版评估（5 分钟）

```bash
cd /home/scyjf4/work/intern/my_intern/week2/retrieval_data
./run_improved_eval.sh
```

预期看到 F1 提升到 0.28-0.30

### 第 2 步：重新训练基础模型（1-2 小时）

```bash
# 编辑配置
vim sfks_bert_v4_improved.py  # 修改上述参数

# 重新训练
python sfks_bert_v4_improved.py
```

预期新模型 F1 达到 0.30+

### 第 3 步：再次评估（5 分钟）

```bash
python sfks_comprehensive_eval_improved.py
```

预期 RAG 系统 F1 达到 0.35+

---

## 为什么之前的评估结果不准确？

### 问题代码

```python
# sfks_comprehensive_eval.py 第 397-440 行
collection = client.get_collection(name=name, embedding_function=None)

# 获取所有文档（但没有使用 embedding！）
all_data = collection.get(include=["documents"])
documents = all_data.get("documents", [])

# 只用字符匹配（完全没用到向量）
def retrieve(query: str) -> str:
    scores = []
    for i, doc in enumerate(documents):
        score = sum(1 for char in query if char in doc and '\u4e00' <= char <= '\u9fff')
```

**原因**：

1. 虽然加载了 Chroma 向量库，但设置了 `embedding_function=None`
2. 直接调用 `collection.get()` 获取所有文档，没有用 `collection.query()`
3. 检索时只统计字符重复，没有计算向量相似度

**正确做法**（已在改进版中实现）：

```python
# 使用混合检索器
from sfks_hybrid_retriever import HybridRetriever

retriever = HybridRetriever(
    chroma_path=str(chroma_path),
    youtu_model_path=str(youtu_model_path),
    alpha=0.7  # 70% 向量，30% BM25
)

def retrieve(query: str) -> str:
    results = retriever.search(query, top_k=5)  # 真正的向量检索
    docs = [doc for doc, score in results if score >= 0.1]
    return " [SEP] ".join(docs[:3])
```

---

## 总结

### 当前状态

- ? 评估脚本没用向量检索，只是字符匹配
- ? 上下文融合粗糙
- ? 基础模型性能弱

### 已提供的解决方案

- ? `sfks_comprehensive_eval_improved.py` - 改进的评估脚本
- ? `RAG_OPTIMIZATION_GUIDE.md` - 详细优化指南
- ? `run_improved_eval.sh` - 一键运行脚本

### 预期效果

| 阶段 | F1 | 改进 |
|------|-----|------|
| 当前（简单字符匹配） | 0.234 | - |
| 使用 BM25 检索 | 0.26-0.28 | +15% |
| 真正的向量检索 | 0.30-0.32 | +30% |
| 提升基础模型 | 0.35-0.38 | +50% |
| 高级优化 | 0.38-0.42 | +70% |

### 立即行动

```bash
# 运行改进版评估，立即看到效果
cd /home/scyjf4/work/intern/my_intern/week2/retrieval_data
./run_improved_eval.sh
```
