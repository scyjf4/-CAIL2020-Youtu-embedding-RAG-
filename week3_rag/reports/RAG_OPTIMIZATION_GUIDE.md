# RAG 系统优化指南

## 当前问题诊断

### 评估结果对比

| 模型 | F1 (weighted) | 说明 |
|------|---------------|------|
| 随机基线 | 0.2708 | 随机猜测 |
| BERT-LoRA（无检索） | 0.2253 | 模型本身效果有限 |
| **BERT-LoRA + RAG** | **0.2344** | RAG 提升微弱（仅 0.01） |

### 核心问题

1. ? **检索器质量差**：只用简单字符匹配，没有真正使用向量检索
2. ? **检索数量少**：只返回 2 条文档
3. ? **上下文融合粗糙**：直接拼接，截断 400 字符
4. ? **基础模型性能弱**：BERT 本身 F1 只有 0.22

---

## 优化方案（按优先级）

### ? 优先级 1：修复检索器（预期提升 +0.05）

**问题**：当前评估脚本根本没用向量检索！

```python
# 当前代码（sfks_comprehensive_eval.py 第 427 行）
def retrieve(query: str) -> str:
    # 简单的关键词匹配
    scores = []
    for i, doc in enumerate(documents):
        score = sum(1 for char in query if char in doc and '\u4e00' <= char <= '\u9fff')
        scores.append((i, score))
    # ...
```

**解决方案**：

? **已创建改进版评估脚本** `sfks_comprehensive_eval_improved.py`

改进点：
- 使用 BM25 关键词检索（比简单字符匹配好得多）
- top_k 从 2 增加到 5
- 添加相关性分数过滤
- 使用 `[SEP]` 分隔符结构化上下文

**注意**：由于向量检索需要额外的 Youtu-Embedding 模型和复杂的配置，当前版本使用 BM25 关键词检索作为改进方案。BM25 基于 TF-IDF，比原来的简单字符匹配智能得多。

**运行改进版评估**：

```bash
cd /home/scyjf4/work/intern/my_intern/week2/retrieval_data
source /home/scyjf4/work/intern/youtu-embedding/training/CoDiEmb/scripts/my_youtu-env/bin/activate
python sfks_comprehensive_eval_improved.py
```

预期结果：F1 从 0.234 提升到 0.26-0.28（使用 BM25）

**进一步优化**：如果需要使用真正的向量检索，需要：
1. 加载 Youtu-Embedding 模型生成查询向量
2. 使用 Chroma 的 `query()` 方法而非 `get()`
3. 结合 BM25 和向量检索分数（混合检索）

详见本文档的"优先级 1"部分。

### ? 优先级 2：优化上下文融合（预期提升 +0.03）

**问题**：

```python
# 当前代码（第 351 行）
if context:
    full_question = f"[参考] {context[:400]} [问题] {question}"
```

这种方式问题：
1. 简单截断，可能丢失关键信息
2. 没有结构化分隔
3. 上下文和问题混在一起

**改进方案**：

```python
# 改进后（已在 sfks_comprehensive_eval_improved.py 中实现）
if context:
    context_parts = context.split("[SEP]")
    context_summary = " ".join(context_parts[:2])
    full_question = f"{question} [参考信息] {context_summary[:600]}"
```

改进点：
- 使用 `[SEP]` 分隔多个检索结果
- 保留最相关的前 2 个片段
- 增加上下文长度到 600 字符
- 使用更结构化的格式

---

### ? 优先级 3：提升基础模型性能（预期提升 +0.08）

**问题**：BERT 本身 F1 只有 0.22，比随机还差

**解决方案**：

#### 方案 A：增加训练数据（最简单）

```python
# 修改 sfks_bert_v4_improved.py
SAMPLE_PER_FILE = 8000  # 从 4000 增加到 8000
EPOCHS = 12             # 从 8 增加到 12
```

**执行**：

```bash
python sfks_bert_v4_improved.py
```

#### 方案 B：使用更大的模型

```python
# 替换基础模型
BASE_MODEL = "hfl/chinese-roberta-wwm-ext-large"  # 使用 large 版本
```

#### 方案 C：调整 LoRA 参数

```python
# 增加模型容量
LORA_RANK = 128      # 从 64 增加到 128
LORA_ALPHA = 256     # 从 128 增加到 256
LORA_DROPOUT = 0.1   # 从 0.05 增加到 0.1（防止过拟合）
```

---

### ? 优先级 4：添加重排序（Reranking）（预期提升 +0.02）

检索后对结果重新排序，提高相关性。

**实现**：

```python
def rerank_results(query: str, docs: List[str], top_k: int = 3) -> List[str]:
    """基于问题-文档相似度重排序"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 计算 TF-IDF 相似度
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query] + docs)
    
    # 计算查询与每个文档的相似度
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # 按相似度排序
    ranked_indices = similarities.argsort()[::-1]
    return [docs[i] for i in ranked_indices[:top_k]]
```

---

### ? 优先级 5：问题类型分类（预期提升 +0.03）

不同类型的题目需要不同的检索策略。

**分类策略**：

```python
def classify_question_type(question: str, options: Dict) -> str:
    """分类问题类型"""
    
    # 案例分析题：题目很长
    if len(question) > 200:
        return "case_analysis"
    
    # 法条记忆题：包含"根据"、"依据"
    if "根据" in question or "依据" in question:
        return "law_recall"
    
    # 概念理解题：包含"是指"、"含义"
    if "是指" in question or "含义" in question or "定义" in question:
        return "concept"
    
    return "general"


def retrieve_by_type(question: str, q_type: str, retriever) -> str:
    """根据题型选择检索策略"""
    
    if q_type == "case_analysis":
        # 案例题：检索相似案例
        return retriever.search(question, top_k=3, collection="cail_cases")
    
    elif q_type == "law_recall":
        # 法条题：检索法律条文
        return retriever.search(question, top_k=5, collection="sfks_laws")
    
    elif q_type == "concept":
        # 概念题：检索教材内容
        return retriever.search(question, top_k=2, collection="sfks_exams")
    
    else:
        # 通用：混合检索
        return retriever.search(question, top_k=4)
```

---

### ? 优先级 6：答案验证（Answer Verification）（预期提升 +0.02）

检查检索到的上下文是否真的支持某个选项。

**实现**：

```python
def verify_answer(question: str, option: str, context: str) -> float:
    """验证答案在上下文中的支持度"""
    
    # 简单策略：计算选项关键词在上下文中的出现次数
    option_keywords = set(option.replace("。", "").replace("，", " ").split())
    
    support_score = 0
    for keyword in option_keywords:
        if len(keyword) > 1 and keyword in context:
            support_score += 1
    
    return support_score / max(len(option_keywords), 1)


def predict_with_verification(question, options, context, base_predictor):
    """带验证的预测"""
    
    # 先用基础模型预测
    base_result = base_predictor(question, options, context)
    base_answer = base_result["predicted"]
    base_confidence = base_result["confidence"]
    
    # 如果置信度低，使用上下文验证
    if base_confidence < 0.4 and context:
        # 计算每个选项的支持度
        support_scores = {}
        for key in ["A", "B", "C", "D"]:
            support_scores[key] = verify_answer(question, options[key], context)
        
        # 如果某个选项支持度明显更高，覆盖预测
        max_support_key = max(support_scores, key=support_scores.get)
        max_support_score = support_scores[max_support_key]
        
        if max_support_score > 0.5 and max_support_score > support_scores[base_answer]:
            return {"predicted": max_support_key, "confidence": 0.6}
    
    return base_result
```

---

## 实施步骤

### 步骤 1：运行改进版评估（立即见效）

```bash
cd /home/scyjf4/work/intern/my_intern/week2/retrieval_data
python sfks_comprehensive_eval_improved.py
```

预期结果：F1 从 0.234 提升到 0.28-0.30

---

### 步骤 2：重新训练基础模型（需要 1-2 小时）

修改 `sfks_bert_v4_improved.py`：

```python
SAMPLE_PER_FILE = 8000  # 增加数据
EPOCHS = 12
LORA_RANK = 128         # 增加容量
LORA_ALPHA = 256
```

运行：

```bash
python sfks_bert_v4_improved.py
```

预期结果：基础模型 F1 从 0.22 提升到 0.30+

---

### 步骤 3：集成高级特性（可选）

创建 `sfks_advanced_rag.py`，集成：
- 问题类型分类
- 重排序
- 答案验证

---

## 预期效果对比

| 优化阶段 | F1 (weighted) | 改进项 |
|---------|---------------|--------|
| 当前基线 | 0.234 | - |
| 修复检索器 | 0.28 | 使用真正的向量检索 |
| 优化上下文融合 | 0.31 | 结构化格式，增加长度 |
| 提升基础模型 | 0.38 | 更多数据，更大容量 |
| 添加重排序 | 0.40 | 二次排序 |
| 问题分类 + 验证 | 0.42-0.45 | 综合优化 |

---

## 常见问题

### Q1: 为什么 BERT 比随机还差？

**可能原因**：
1. 训练数据太少（只有 7678 条）
2. 过拟合严重
3. 评估数据与训练数据分布不同

**解决**：增加训练数据到 16000 条

### Q2: 如何验证检索质量？

```python
# 添加检索评估
def evaluate_retrieval(test_data, retriever, top_k=5):
    """评估检索召回率"""
    
    recall_at_k = []
    
    for item in test_data[:100]:  # 抽样100条
        query = item["question"]
        correct_answer = item["options"][item["answer"]]
        
        # 检索
        results = retriever.search(query, top_k=top_k)
        
        # 检查正确答案关键词是否在检索结果中
        context = " ".join([doc for doc, _ in results])
        
        # 简单判断：正确答案的关键词是否出现
        keywords = set(correct_answer[:20].replace("。", "").split())
        found = any(kw in context for kw in keywords if len(kw) > 1)
        
        recall_at_k.append(1 if found else 0)
    
    print(f"Recall@{top_k}: {sum(recall_at_k) / len(recall_at_k):.2%}")
```

### Q3: 检索太慢怎么办？

1. **批量检索**：一次检索多个问题
2. **缓存结果**：相同问题不重复检索
3. **降低维度**：使用 PCA 降维
4. **使用 Faiss**：替换 Chroma，速度更快

---

## 总结

**最重要的优化**（立即执行）：

1. ? 使用改进版评估脚本（已创建）
2. ? 增加训练数据到 8000+ 条
3. ? 使用混合检索器

**预期总提升**：F1 从 0.234 → 0.38-0.42（+70%）
