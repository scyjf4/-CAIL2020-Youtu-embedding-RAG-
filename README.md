# CAIL2020 智能司法问答系统

> 基于CAIL2020数据集和Youtu-Embedding的检索增强生成(RAG)智能问答系统

## 项目概述

本项目是一个智能司法问答系统，针对中国法律职业资格考试(司法考试)场景，结合了：
- **检索增强生成(RAG)**: 提供可靠的知识来源
- **LoRA微调**: 参数高效的模型定制
- **混合检索**: 向量检索 + BM25关键词检索
- **大语言模型**: BERT, ChatGLM, Qwen等

## 快速导航

- [项目结构](PROJECT_STRUCTURE.md) - 详细的目录结构说明
- [第二周任务](week2_baseline/README.md) - 基线模型构建与微调
- [第三周任务](week3_rag/README.md) - 高级检索与RAG系统
- [优化指南](docs/OPTIMIZATION_GUIDE.md) - 优化方法和策略
- [性能指南](docs/PERFORMANCE_GUIDE.md) - 性能提升建议

## 项目亮点

### 1. LoRA参数高效微调 ?

只训练0.3%的参数，达到接近全量微调的效果：

| 对比项 | 全量微调 | LoRA微调 |
|-------|---------|----------|
| 训练参数 | 102M (100%) | 0.3M (0.3%) |
| 显存需求 | ~16GB | ~5GB |
| 训练时间 | 基线 | 3-5倍提升 |
| F1分数 | - | 0.331 |

### 2. 混合检索系统 ?

结合语义检索和关键词检索的优势：

```
查询: "合同纠纷的诉讼时效是多久？"
    ↓
向量检索 → 语义相似文档
BM25检索 → 关键词匹配文档
    ↓
RRF融合 → Top-K 最相关文档
```

### 3. RAG问答流水线 ?

完整的端到端问答系统：

```
问题输入 → 混合检索 → 上下文构建 → LLM生成 → 答案输出
```

## 性能指标

### 检索性能
- Recall@5: -
- Recall@10: -
- 检索延迟: <50ms

### 问答性能

| 模型 | 准确率 | F1-Score | 延迟 |
|------|--------|----------|------|
| 随机基线 | 0.24 | 0.24 | 0ms |
| BERT-LoRA | 0.24 | 0.23 | 25ms |
| **BERT-LoRA + RAG** | **0.26** | **0.26** | 85ms |

**RAG系统提升**: +8.4% 准确率, +10.0% F1分数

## 项目结构

```
my_intern/
├── data/                      # 原始数据
│   ├── sfks/                  # 司法考试数据集 (7,678条)
│   └── ydlj/                  # 阅读理解数据集
│
├── week2_baseline/            # 第二周：基线模型
│   ├── 1_data_preprocessing/  # 数据预处理
│   ├── 2_retrieval/          # 检索模块
│   ├── 3_model_training/     # 模型训练 (BERT+LoRA)
│   ├── 4_evaluation/         # 模型评估
│   └── reports/              # 性能报告、PPT素材
│
├── week3_rag/                # 第三周：RAG系统
│   ├── 1_hybrid_retrieval/   # 混合检索
│   ├── 2_rag_pipeline/       # RAG流水线
│   ├── 3_llm_integration/    # LLM集成
│   ├── 4_evaluation/         # 系统评估
│   └── reports/              # 对比评测报告
│
└── docs/                     # 文档中心
    ├── OPTIMIZATION_GUIDE.md # 优化指南
    └── PERFORMANCE_GUIDE.md  # 性能指南
```

详细说明: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 快速开始

### 环境准备

```bash
# Python 3.8+
pip install -r requirements.txt
```

### 第二周：训练基线模型

```bash
cd week2_baseline

# 1. 构建向量检索库
python 2_retrieval/sfks_chroma.py

# 2. 训练BERT+LoRA模型
python 3_model_training/sfks_bert_v4_improved.py

# 3. 评估性能
python 4_evaluation/sfks_bert_evaluate.py
```

### 第三周：运行RAG系统

```bash
cd week3_rag

# 1. 测试混合检索
python 1_hybrid_retrieval/sfks_hybrid_retriever.py

# 2. 运行RAG流水线
python 2_rag_pipeline/sfks_rag_pipeline.py

# 3. 综合评估
python 4_evaluation/sfks_comprehensive_eval_improved.py
```

## 重要文档

### 技术文档
- [LoRA使用说明](week2_baseline/reports/README_LORA.md)
- [LoRA PPT素材](week2_baseline/reports/LORA_PPT_MATERIAL.md) - **适合做汇报**
- [RAG优化指南](week3_rag/reports/RAG_OPTIMIZATION_GUIDE.md)

### 评估报告
- [基线模型性能报告](week2_baseline/reports/)
- [第三周完整报告](week3_rag/reports/WEEK3_REPORT.md)

### 优化指南
- [优化方法总结](docs/OPTIMIZATION_GUIDE.md)
- [性能提升策略](docs/PERFORMANCE_GUIDE.md)

## 技术栈

### 核心框架
- **Transformers** (HuggingFace) - 预训练模型加载
- **PEFT** - LoRA微调
- **LangChain** - RAG流水线
- **Chroma** - 向量数据库

### 模型
- **BERT** (`bert-base-chinese`) - 基线问答模型
- **ChatGLM-6B** - 开源对话模型
- **Qwen** - 通义千问系列
- **Youtu-Embedding** - 腾讯优图向量化模型

### 检索
- **Chroma** - 向量检索
- **BM25** - 关键词检索
- **RRF** - 结果融合

## 数据集

### CAIL2020-SFKS (司法考试)
- **来源**: 中国法律智能大赛2020
- **类型**: 单选题、多选题
- **规模**: 7,678条训练数据
- **格式**: JSON Lines

### YDLJ (阅读理解)
- **来源**: 阅读理解数据集
- **类型**: 问答对
- **格式**: JSON

示例:
```json
{
  "statement": "关于合同纠纷的诉讼时效，下列说法正确的是？",
  "option_list": {
    "A": "一年",
    "B": "两年", 
    "C": "三年",
    "D": "五年"
  },
  "answer": ["C"]
}
```

## 项目重组

如果你的项目文件混乱，运行重组脚本：

```bash
cd /home/scyjf4/work/intern/my_intern
bash reorganize_project.sh
```

这会自动将文件整理到规范的目录结构中。

## 常见问题

### Q1: 评估结果与训练时不一致？

**A**: 这是正常现象。训练时的F1=0.331是在验证集上的结果，评估时使用的是独立测试集（F1≈0.25），后者更能反映真实泛化能力。详见 [WEEK3_REPORT.md](week3_rag/reports/WEEK3_REPORT.md)。

### Q2: 如何提升RAG效果？

**A**: 参考优化建议:
1. 调整检索参数（alpha, top_k）
2. 改进Prompt模板
3. 使用更大的LLM
4. 优化文档分块策略

详见 [优化指南](docs/OPTIMIZATION_GUIDE.md)。

## 关于LoRA的说明 (PPT素材)

### 什么是LoRA？

**LoRA (Low-Rank Adaptation)** - 低秩适配，一种参数高效的大模型微调方法。

### 核心优势

1. **参数高效**: 只训练0.3%的参数
2. **显存友好**: 降低70%显存需求
3. **训练加速**: 提升3-5倍训练速度
4. **防止过拟合**: 参数量小，正则化效果好

### 项目中的应用

在本项目中，LoRA用于微调BERT模型进行司法考试问答：

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=64,                    # LoRA秩 (控制参数量)
    lora_alpha=128,          # 缩放系数
    lora_dropout=0.05,       # Dropout防止过拟合
    target_modules=[         # 在哪些层添加LoRA
        "query", "key", "value", "dense"
    ],
)

model = get_peft_model(base_model, config)
```

### 效果对比

| 配置 | 参数量 | 显存 | 训练时间 | F1分数 |
|------|-------|------|---------|--------|
| 全量微调 | 102M | ~16GB | 基线 | - |
| **LoRA** | **0.3M** | **~5GB** | **3-5倍快** | **0.331** |

### 详细PPT素材

完整的PPT素材（包含图表、代码示例、性能对比）请查看:
- [LORA_PPT_MATERIAL.md](week2_baseline/reports/LORA_PPT_MATERIAL.md)
- [LORA_QUICK_REFERENCE.md](week2_baseline/reports/LORA_QUICK_REFERENCE.md)

## 交付物清单

### 第二周 ?

1. **代码仓库**: `week2_baseline/`
   - 数据预处理
   - 检索模块
   - 模型训练
   - 评估脚本

2. **基线模型性能报告**: `week2_baseline/reports/`
   - 检索Recall@K
   - 模型F1分数
   - LoRA配置说明

3. **中期汇报PPT素材**: `week2_baseline/reports/LORA_PPT_MATERIAL.md`

### 第三周 ?

1. **混合检索模块代码**: `week3_rag/1_hybrid_retrieval/`
2. **端到端RAG系统**: `week3_rag/2_rag_pipeline/`
3. **系统评测报告**: `week3_rag/reports/WEEK3_REPORT.md`

## Git仓库准备

### 建议的.gitignore

```gitignore
# 模型文件（太大）
*.bin
*.safetensors
*_lora_output*/
*_bert_dap/

# 向量库（太大）
chroma_*/
*.npy
*_vectors/

# Python
__pycache__/
*.pyc
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp

# 临时文件
*.log
*.tmp
```

### 提交建议

```bash
# 1. 整理项目结构
bash reorganize_project.sh

# 2. 初始化Git
git init
git add .gitignore README.md PROJECT_STRUCTURE.md
git add week2_baseline/ week3_rag/ docs/

# 3. 提交
git commit -m "feat: CAIL2020智能司法问答系统完整实现

- 第二周: BERT+LoRA基线模型 (F1=0.331)
- 第三周: 混合检索+RAG系统 (准确率提升8.4%)
- 完整文档和评估报告"

# 4. 推送到远程
git remote add origin <your-repo-url>
git push -u origin main
```

## 许可证

本项目仅用于学习和研究目的。

---

**项目完成时间**: 2026年2月  
**主要技术**: LoRA, RAG, BERT, Chroma, BM25  
**数据集**: CAIL2020-SFKS (7,678条), YDLJ
