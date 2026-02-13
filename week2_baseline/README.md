# 第二周：基线模型构建与微调

## 任务目标

1. ? 使用Youtu-Embedding构建文本检索模块
2. ? 在CAIL2020子集上微调轻量级问答模型（BERT+LoRA）
3. ? 评估检索模块和问答模型的基线性能
4. ? 准备中期汇报材料

## 目录结构

```
week2_baseline/
├── 1_data_preprocessing/      # 数据预处理
├── 2_retrieval/              # 检索模块
├── 3_model_training/         # 模型训练
├── 4_evaluation/             # 模型评估
├── models/                   # 训练好的模型
├── vector_stores/            # 向量数据库
└── reports/                  # 报告和文档
```

## 快速开始

### 1. 安装依赖

```bash
cd week2_baseline
pip install -r requirements.txt
```

### 2. 数据质量检查

```bash
python 1_data_preprocessing/check_data_quality.py
```

### 3. 构建向量检索库

```bash
# 构建Chroma向量数据库
python 2_retrieval/sfks_chroma.py

# 测试检索
python 2_retrieval/sfks_query.py
```

### 4. 训练BERT+LoRA模型

```bash
# V3版本（推荐）
python 3_model_training/sfks_bert_v3.py

# V4改进版（全数据集）
python 3_model_training/sfks_bert_v4_improved.py
```

### 5. 评估模型性能

```bash
# 基础评估
python 4_evaluation/sfks_bert_evaluate.py

# V3评估
python 4_evaluation/sfks_bert_evaluate_v3.py
```

## 关键交付物

### 1. 代码仓库 ?

- **数据预处理**: `1_data_preprocessing/`
  - `check_data_quality.py` - 数据质量检查
  - `diagnose_data.py` - 数据诊断

- **检索模块**: `2_retrieval/`
  - `sfks_chroma.py` - 构建Chroma向量库
  - `sfks_query.py` - 检索查询接口
  - `retrieve_for_sfks.py` - 批量检索

- **模型训练**: `3_model_training/`
  - `sfks_bert.py` - BERT基线
  - `sfks_bert_v2.py` - BERT v2（优化版）
  - `sfks_bert_v3.py` - BERT v3（LoRA微调）
  - `sfks_bert_v4_improved.py` - BERT v4（最佳版本）

- **模型评估**: `4_evaluation/`
  - `sfks_bert_inference.py` - 推理引擎
  - `sfks_bert_evaluate.py` - 评估脚本

### 2. 基线模型性能报告 

**文件**: `reports/baseline_performance.md`

#### 检索模块性能

| 指标 | 值 | 说明 |
|------|-----|------|
| Recall@1 | - | Top-1召回率 |
| Recall@5 | - | Top-5召回率 |
| Recall@10 | - | Top-10召回率 |

#### 问答模型性能

| 模型版本 | 准确率 | F1-Weighted | F1-Macro | 训练样本 |
|---------|--------|-------------|----------|---------|
| V1 基线 | - | 0.178 | - | 2000 |
| V2 优化 | - | - | - | - |
| **V4 最佳** | **0.331** | **0.331** | **0.331** | **7678** |

#### LoRA配置

```python
LoraConfig(
    r=64,                  # LoRA秩
    lora_alpha=128,        # 缩放系数
    lora_dropout=0.05,     # Dropout
    target_modules=[       # 目标模块
        "query", "key", "value", "dense"
    ],
)
```

**训练参数**:
- 数据集: CAIL2020-SFKS（司法考试）
- 样本数: 7,678条
- Epoch: 8
- Batch Size: 8
- Learning Rate: 2e-4
- 优化器: AdamW

### 3. 中期汇报PPT素材 

**文件**: `reports/LORA_PPT_MATERIAL.md`

包含内容:
- LoRA原理介绍
- 传统微调 vs LoRA对比
- 项目中的应用场景
- 关键参数配置说明
- 训练效果展示
- 性能对比分析

## 技术栈

- **预训练模型**: BERT (`bert-base-chinese`)
- **微调方法**: LoRA (Low-Rank Adaptation)
- **向量库**: Chroma
- **嵌入模型**: Youtu-Embedding
- **框架**: Transformers, PEFT

## 训练好的模型

所有训练好的模型保存在 `models/` 目录:

- `sfks_bert_lora_output/` - V1基线模型
- `sfks_bert_lora_output_v3/` - V3模型
- `sfks_bert_lora_v4_improved/` - **V4最佳模型（推荐）**
- `sfks_bert_lora_v5_no_dap/` - V5无DAP版本

## 性能优化

详见文档:
- `reports/README_LORA.md` - LoRA使用说明
- `reports/LORA_QUICK_REFERENCE.md` - LoRA快速参考
- `../docs/OPTIMIZATION_GUIDE.md` - 优化指南

## 常见问题


### Q1: 向量库构建失败？
参考: `../docs/troubleshooting/SFKS_VECTORIZATION_FIX.md`

### Q2: F1分数如何提升？
参考: `../docs/troubleshooting/IMPROVE_F1.md`

## 下一步

完成第二周任务后，进入第三周:
- 实现混合检索（向量+BM25）
- 集成LLM构建RAG系统
- 端到端评估

详见: `../week3_rag/README.md`
