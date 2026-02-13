# CAIL2020 司法考试问答系统 - 优化方案

## ? 概述

本项目实现了三种优化方案来提升司法考试问答模型的 F1 分数：

1. **领域自适应预训练 (DAP)** - 用法律文本继续预训练 BERT
2. **混合检索 (向量 + BM25)** - 结合语义和关键词检索
3. **综合优化的 LoRA 微调** - 改进的训练策略

## ? 快速开始

### 环境准备

```bash
cd /home/scyjf4/work/intern/my_intern/week2_baseline
```

### 方案1: 领域自适应预训练 (推荐先执行)

```bash
# 用司法考试文本对 BERT 进行 MLM 预训练
python 3_model_training/sfks_dap_pretrain.py
```

**原理**: 让 BERT 在法律领域文本上继续预训练，学习法律术语和表达方式。

**输出**: `models/sfks_bert_dap/` - 领域适应的 BERT 模型

### 方案2: 混合检索测试

```bash
# 测试混合检索效果
cd ../week3_rag
python 1_hybrid_retrieval/sfks_hybrid_retriever.py
```

**原理**: 结合向量检索（语义相似）和 BM25（关键词匹配），提升召回率。

**参数**:
- `alpha=0.6` - 60% 向量权重，40% BM25 权重

### 方案3: 综合优化训练

```bash
cd ../week2_baseline
# 使用所有优化策略进行训练
python 3_model_training/sfks_bert_v3.py

# 评估模型
python 4_evaluation/sfks_bert_evaluate_v3.py
```

**优化点**:
- 只使用单选题（过滤多选题）
- 优先使用 DAP 模型作为基础
- 增强的问题表示（融入选项上下文）
- 更大的 LoRA 秩 (32)
- 扩展目标模块 (query, key, value, dense)
- 早停机制
- 更多训练轮数

## ? 文件说明

| 文件 | 功能 |
|------|------|
| `3_model_training/sfks_dap_pretrain.py` | 领域自适应预训练 |
| `3_model_training/sfks_bert_v3.py` | 综合优化版训练 |
| `3_model_training/sfks_bert_v2.py` | 基础优化版训练 |
| `4_evaluation/sfks_bert_evaluate_v3.py` | 综合版评估 |
| `4_evaluation/sfks_bert_evaluate_v2.py` | 基础版评估 |

## ? 预期效果

| 版本 | 预期 Accuracy | 预期 F1 |
|------|--------------|---------|
| v1 (原始) | ~0.17 | ~0.17 |
| v2 (单选题) | ~0.35-0.45 | ~0.35-0.45 |
| v3 (综合优化) | ~0.45-0.55 | ~0.45-0.55 |
| v3 + DAP | ~0.50-0.60 | ~0.50-0.60 |

## ? 超参数调优建议

### LoRA 参数
```python
LORA_RANK = 32      # 可尝试 16, 32, 64
LORA_ALPHA = 64     # 通常为 RANK 的 2 倍
LORA_DROPOUT = 0.1  # 可尝试 0.05, 0.1, 0.15
```

### 训练参数
```python
learning_rate = 2e-5        # 可尝试 1e-5, 2e-5, 5e-5
batch_size = 16             # 显存允许可增加
num_train_epochs = 20       # 配合早停使用
warmup_ratio = 0.1
```

## ? 进阶优化

1. **数据增强**
   - 同义词替换
   - 回译增强
   - 选项打乱

2. **集成学习**
   - 训练多个模型
   - 投票融合

3. **对比学习**
   - 正负样本对比
   - 增强选项区分能力

## ? 推荐执行顺序

```bash
# 1. 先做领域预训练（可选但推荐）
python 3_model_training/sfks_dap_pretrain.py

# 2. 综合优化训练
python 3_model_training/sfks_bert_v3.py

# 3. 评估
python 4_evaluation/sfks_bert_evaluate_v3.py
```

## 相关文档

- [性能提升指南](PERFORMANCE_GUIDE.md) - 更多性能优化建议
- [第三周报告](../week3_rag/reports/WEEK3_REPORT.md) - RAG系统评估结果

