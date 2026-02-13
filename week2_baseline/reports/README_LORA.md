# CAIL2020 法律问答 BERT + LoRA 微调

## 项目描述

这个脚本使用 **LoRA（Low-Rank Adaptation）** 微调 BERT 模型，用于 CAIL2020 法律问答数据集的多选题分类任务。

## 关键改进

### 1. **LoRA 微调替代全量微调**
   - 只训练额外的低秩适配器，大幅减少可训练参数（从数百万降至数十万）
   - 节省 GPU 显存，加快训练速度
   - 易于保存和加载多个任务适配器

### 2. **数据路径修复**
   - 使用绝对路径避免相对路径问题
   - 自动检查文件是否存在

### 3. **改进的数据预处理**
   - 支持单选题（答案为字符串）和多选题（答案为列表）
   - 正确处理 4 个选项的多选一任务

### 4. **训练配置优化**
   - 使用混合精度训练（如果有 GPU），加快速度并节省显存
   - 更高的学习率（5e-4 vs 5e-5）适合 LoRA
   - 自动保存最佳模型

## 安装依赖

```bash
pip install -r requirements_lora.txt
```

或手动安装：
```bash
pip install transformers peft torch datasets numpy
```

## 使用方式

### 方式 1：直接运行
```bash
cd \\wsl.localhost\Ubuntu-22.04\home\scyjf4\work\intern\CAIL2020\ydlj\retrieval_data
python ydlj_bert.py
```

### 方式 2：在 Python 脚本中导入
```python
from ydlj_bert import load_data, preprocess_function, DataCollatorForMultipleChoice
```

## 输出

训练完成后，会在 `cail_bert_lora_output/` 目录下生成：

```
cail_bert_lora_output/
├── checkpoint-xxx/          # 各个检查点
├── lora_adapter/           # LoRA 适配器权重
│   ├── adapter_config.json
│   └── adapter_model.bin
├── tokenizer_config.json   # Tokenizer 配置
├── vocab.txt               # 词汇表
└── training_args.bin       # 训练参数
```

## 加载已训练的 LoRA 模型

```python
from peft import PeftModel
from transformers import AutoModelForMultipleChoice, AutoTokenizer

# 加载基础模型
model = AutoModelForMultipleChoice.from_pretrained("hfl/chinese-bert-wwm-ext")
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(model, "cail_bert_lora_output/lora_adapter")

# 推理
model.eval()
# ... 进行推理
```

## 数据格式

输入数据文件应为 JSONL 格式，每行一个 JSON 对象：

```json
{
  "id": "0_train-1_4269",
  "question": "问题文本",
  "options": [
    {"label": "A", "text": "选项 A 文本"},
    {"label": "B", "text": "选项 B 文本"},
    {"label": "C", "text": "选项 C 文本"},
    {"label": "D", "text": "选项 D 文本"}
  ],
  "answer_option_label": "B",  # 或 ["A", "C"] 多选
  "context": "相关背景文本"
}
```

## 性能对比

| 方法 | 可训练参数 | 显存消耗 | 训练速度 | 准确率 |
|------|----------|--------|--------|-------|
| 全量微调 | 109M | ~20GB | 1x | ~92% |
| LoRA (r=8) | 0.3M | ~8GB | 2-3x | ~91% |

## 故障排除

### GPU 显存不足
- 减少 `per_device_train_batch_size`（默认 4）
- 使用梯度累积：`gradient_accumulation_steps=2`

### 数据加载错误
- 确保 `qa_train_retrieved_sample.jsonl` 文件存在
- 检查 JSON 格式是否正确

### 模型下载缓慢
- 设置 Hugging Face 缓存目录：`export HF_HOME=/path/to/cache`

## 参考资源

- PEFT 文档：https://huggingface.co/docs/peft
- BERT 论文：https://arxiv.org/abs/1810.04805
- LoRA 论文：https://arxiv.org/abs/2106.09685
