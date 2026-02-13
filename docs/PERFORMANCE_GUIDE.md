# 司法考试问答系统 - 性能提升指南

## 当前最佳结果

| 版本 | 模型 | 数据量 | F1 (weighted) | 准确率 |
|------|------|--------|---------------|--------|
| V1 初始 | BERT | 2000 | 0.178 | 19.2% |
| **V4 最佳** | **BERT + LoRA** | **7678** | **0.331** | **34.4%** |

---

## 进一步提升 F1 的方法

### 方法 1: 使用全量数据 (预期 F1: 0.38-0.42)

目前只用了约 7678 条单选题，实际有 10000+ 条。

```python
# 修改训练脚本
SAMPLE_PER_FILE = None  # 使用全部数据
```

### 方法 2: 数据增强 (预期提升: +0.02-0.05)

```python
# 选项打乱增强
import random

def augment_data(item):
    """打乱选项顺序，增加数据多样性"""
    options = item["options"]
    answer = item["answer"]
    
    # 随机打乱选项
    keys = list(options.keys())
    random.shuffle(keys)
    
    new_options = {new_key: options[old_key] 
                   for new_key, old_key in zip(["A", "B", "C", "D"], keys)}
    
    # 更新答案
    new_answer = ["A", "B", "C", "D"][keys.index(answer)]
    
    return {"options": new_options, "answer": new_answer, ...}
```

### 方法 3: 集成学习 (预期提升: +0.03-0.08)

训练多个模型，投票决定最终答案：

```python
models = [
    "hfl/chinese-bert-wwm-ext",
    "hfl/chinese-roberta-wwm-ext", 
    "nghuyong/ernie-3.0-base-zh",
]

def ensemble_predict(question, options):
    votes = []
    for model in models:
        pred = predict_with_model(model, question, options)
        votes.append(pred)
    
    # 多数投票
    from collections import Counter
    return Counter(votes).most_common(1)[0][0]
```

### 方法 4: 更长的训练 + 学习率调整 (预期提升: +0.02-0.04)

```python
EPOCHS = 15  # 从 8 增加到 15
LEARNING_RATE = 1e-5  # 降低学习率
WARMUP_RATIO = 0.2  # 增加 warmup
```

### 方法 5: RAG 检索增强 (预期提升: +0.05-0.10)

将检索到的法条作为上下文：

```python
# 使用 RAG 系统
# 先构建法条向量库，然后检索增强
```

---

## 关于 RAG 的建议

### 当前模型可以用于 RAG 吗？

**可以！** 有两种使用方式：

#### 方式 1: 检索增强分类（推荐）

```
用户问题 → 向量检索相关法条 → 拼接问题+法条 → BERT 分类 → 答案
```

优点：
- 利用现有模型
- 检索提供额外上下文
- 适合选择题

#### 方式 2: 生成式 RAG

```
用户问题 → 向量检索 → LLM 生成答案
```

需要更大的模型（如 ChatGLM-6B），但你的 8GB 显存可能不够。

### RAG 系统使用步骤

1. **构建向量库**（如果还没有）:
   ```bash
   cd week2_baseline
   python 2_retrieval/sfks_chroma.py
   ```

2. **运行 RAG 评估**:
   ```bash
   cd week3_rag
   python 2_rag_pipeline/sfks_rag_system.py
   ```

3. **查看对比结果**:
   - 无 RAG vs 有 RAG
   - 检查 RAG 是否真的提升了效果

---

## 模型效果的合理预期

对于司法考试这种专业领域：

| F1 范围 | 评价 | 说明 |
|---------|------|------|
| 0.25 以下 | 差 | 不如随机猜测 |
| 0.25-0.35 | 一般 | 当前水平 ? |
| 0.35-0.45 | 良好 | 可用于辅助 |
| 0.45-0.55 | 优秀 | 需要更大模型 |
| 0.55+ | 专家级 | 需要专业法律模型 |

**注意**：司法考试通过率本身就很低（约 10-15%），AI 模型达到 35%+ 已经相当不错！

---

## 推荐下一步

1. **立即可做**：运行 RAG 系统测试效果
   ```bash
   cd week3_rag
   python 2_rag_pipeline/sfks_rag_system.py
   ```

2. **需要时间**：使用全量数据重新训练
   ```bash
   cd week2_baseline
   python 3_model_training/sfks_bert_v4_improved.py
   ```

3. **需要资源**：尝试更大的模型（ChatGLM-6B 需要更多显存）

---

## 文件索引

| 文件 | 说明 |
|------|------|
| `week2_baseline/3_model_training/sfks_bert_v4_improved.py` | 最佳训练脚本 |
| `week3_rag/2_rag_pipeline/sfks_rag_system.py` | RAG 问答系统 |
| `week2_baseline/2_retrieval/sfks_chroma.py` | 向量库构建 |
| `week2_baseline/models/sfks_bert_lora_v4_improved/` | 训练好的模型 |

## 相关文档

- [优化指南](OPTIMIZATION_GUIDE.md) - 优化方法和策略
- [第三周报告](../week3_rag/reports/WEEK3_REPORT.md) - RAG系统评估

