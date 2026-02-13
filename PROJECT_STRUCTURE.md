# CAIL2020 智能司法问答系统 - 项目结构说明

## 项目概述

基于CAIL2020数据集和Youtu-Embedding的智能司法问答系统，包含检索增强生成(RAG)和模型微调。

## 目录结构

```
my_intern/
├── README.md                          # 项目总体说明
├── PROJECT_STRUCTURE.md              # 本文件：项目结构说明
├── requirements.txt                   # 项目依赖
│
├── data/                             # 原始数据
│   ├── sfks/                         # 司法考试数据集
│   │   ├── 0_train.json              # 单选题训练数据
│   │   └── 1_train.json              # 多选题训练数据
│   └── ydlj/                         # 阅读理解数据集
│       └── ...                       # 阅读理解相关数据
│
├── week2_baseline/                   # 第二周：基线模型构建与微调
│   ├── README.md                     # 第二周任务说明
│   ├── requirements.txt              # 第二周依赖
│   │
│   ├── 1_data_preprocessing/         # 数据预处理
│   │   ├── check_data_quality.py    # 数据质量检查
│   │   └── diagnose_data.py         # 数据诊断
│   │
│   ├── 2_retrieval/                  # 检索模块
│   │   ├── sfks_chroma.py           # 向量数据库构建
│   │   ├── sfks_query.py            # 检索查询
│   │   └── retrieve_for_sfks.py     # 批量检索
│   │
│   ├── 3_model_training/             # 模型训练
│   │   ├── sfks_bert.py             # BERT基线训练
│   │   ├── sfks_bert_v2.py          # BERT v2（改进版）
│   │   ├── sfks_bert_v3.py          # BERT v3（LoRA微调）
│   │   └── sfks_train.py            # 通用训练脚本
│   │
│   ├── 4_evaluation/                 # 模型评估
│   │   ├── sfks_bert_inference.py   # 推理脚本
│   │   ├── sfks_bert_evaluate.py    # 评估脚本v1
│   │   ├── sfks_bert_evaluate_v2.py # 评估脚本v2
│   │   └── sfks_bert_evaluate_v3.py # 评估脚本v3
│   │
│   ├── models/                       # 训练好的模型
│   │   ├── sfks_bert_lora_output/   # BERT+LoRA模型
│   │   └── sfks_bert_lora_v4_improved/ # V4改进版
│   │
│   ├── vector_stores/                # 向量数据库
│   │   ├── chroma_sfks_db/          # Chroma向量库
│   │   └── cail_train_embeddings.npy # 预计算向量
│   │
│   └── reports/                      # 报告和文档
│       ├── baseline_performance.md   # 基线性能报告
│       ├── LORA_PPT_MATERIAL.md     # LoRA PPT素材
│       └── evaluation_results.json   # 评估结果
│
├── week3_rag/                        # 第三周：混合检索与RAG系统
│   ├── README.md                     # 第三周任务说明
│   ├── requirements.txt              # 第三周依赖
│   │
│   ├── 1_hybrid_retrieval/           # 混合检索
│   │   └── sfks_hybrid_retriever.py # 向量+BM25混合检索
│   │
│   ├── 2_rag_pipeline/               # RAG流水线
│   │   ├── sfks_rag_pipeline.py     # RAG主流程
│   │   ├── sfks_rag_system.py       # RAG系统封装
│   │   └── sfks_qwen_rag.py         # Qwen+RAG集成
│   │
│   ├── 3_llm_integration/            # LLM集成
│   │   └── sfks_chatglm_lora.py     # ChatGLM+LoRA
│   │
│   ├── 4_evaluation/                 # 系统评估
│   │   ├── sfks_comprehensive_eval.py # 综合评估
│   │   └── sfks_comprehensive_eval_improved.py # 改进评估
│   │
│   └── reports/                      # 报告和文档
│       ├── WEEK3_REPORT.md          # 第三周报告
│       ├── RAG_OPTIMIZATION_GUIDE.md # RAG优化指南
│       └── rag_pipeline_evaluation.json # 评估结果
│
├── docs/                             # 文档中心
│   ├── OPTIMIZATION_GUIDE.md        # 优化指南
│   └── PERFORMANCE_GUIDE.md         # 性能指南
│
└── scripts/                          # 辅助脚本
    ├── fix_all_dependencies.py       # 依赖修复
    └── fix_all_dependencies.sh       # 依赖修复（Shell版）
```

## 第二周交付物

### 1. 代码仓库
- **数据预处理**: `week2_baseline/1_data_preprocessing/`
- **检索模块**: `week2_baseline/2_retrieval/`
- **模型微调**: `week2_baseline/3_model_training/`
- **评估代码**: `week2_baseline/4_evaluation/`

### 2. 基线模型性能报告
- 文件位置: `week2_baseline/reports/baseline_performance.md`
- 包含内容:
  - 检索模块 Recall@K 指标
  - BERT+LoRA 微调后的 F1 分数
  - 对比不同版本的性能

### 3. 中期汇报PPT素材
- 文件位置: `week2_baseline/reports/LORA_PPT_MATERIAL.md`
- 包含内容:
  - LoRA 原理介绍
  - 项目中的应用
  - 训练效果展示

## 第三周交付物

### 1. 混合检索模块代码
- 文件位置: `week3_rag/1_hybrid_retrieval/sfks_hybrid_retriever.py`
- 功能: 向量检索 + BM25 关键词检索

### 2. 端到端RAG问答系统原型
- 文件位置: `week3_rag/2_rag_pipeline/`
- 功能: 完整的检索增强生成流水线

### 3. 系统评测报告
- 文件位置: `week3_rag/reports/WEEK3_REPORT.md`
- 包含内容:
  - 单一检索 vs 混合检索对比
  - RAG系统 vs 基线模型对比
  - 性能分析和优化建议

## 核心技术栈

- **预训练模型**: BERT, ChatGLM-6B, Qwen
- **向量库**: Chroma, FAISS
- **检索**: Youtu-Embedding, BM25
- **微调**: LoRA (Low-Rank Adaptation)
- **框架**: Transformers, PEFT, LangChain

## 快速开始

### 第二周任务
```bash
cd week2_baseline
pip install -r requirements.txt

# 1. 数据预处理
python 1_data_preprocessing/check_data_quality.py

# 2. 构建向量库
python 2_retrieval/sfks_chroma.py

# 3. 训练模型
python 3_model_training/sfks_bert_v3.py

# 4. 评估模型
python 4_evaluation/sfks_bert_evaluate.py
```

### 第三周任务
```bash
cd week3_rag
pip install -r requirements.txt

# 1. 测试混合检索
python 1_hybrid_retrieval/sfks_hybrid_retriever.py

# 2. 运行RAG系统
python 2_rag_pipeline/sfks_rag_pipeline.py

# 3. 综合评估
python 4_evaluation/sfks_comprehensive_eval_improved.py
```

## 文档资源

- [优化指南](docs/OPTIMIZATION_GUIDE.md) - 模型和系统优化方法
- [性能指南](docs/PERFORMANCE_GUIDE.md) - 性能提升策略
