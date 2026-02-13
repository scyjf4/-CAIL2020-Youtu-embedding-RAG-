# -*- coding: utf-8 -*-
"""
DAP 预训练效果验证
对比：原始 BERT vs DAP 预训练 BERT

检查项：
1. MLM 预训练损失是否正常下降
2. 在下游任务上是否有提升
3. DAP 模型是否正确保存
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

print("=" * 60)
print("DAP 预训练效果验证")
print("=" * 60)

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM,
        AutoModelForMultipleChoice,
    )
    from peft import PeftModel, LoraConfig, get_peft_model
    from sklearn.metrics import accuracy_score, f1_score
    print("✓ 依赖导入成功")
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    sys.exit(1)

# ====== 配置 ======
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "sfks"

# 模型路径
ORIGINAL_BERT = "hfl/chinese-bert-wwm-ext"
DAP_MODEL_PATH = SCRIPT_DIR.parent / "models" / "sfks_bert_dap" / "model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_test_data(data_dir: Path, max_samples: int = 200) -> List[Dict]:
    """加载测试数据"""
    import random
    random.seed(42)

    all_data = []
    for filename in ["0_train.json", "1_train.json"]:
        filepath = data_dir / filename
        if not filepath.exists():
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    answer = item.get("answer", [])
                    if not (isinstance(answer, list) and len(answer) == 1):
                        continue
                    options = item.get("option_list", {})
                    if len(options) != 4:
                        continue

                    all_data.append({
                        "question": item.get("statement", ""),
                        "options": options,
                        "answer": answer[0],
                    })
                except:
                    continue

    random.shuffle(all_data)
    return all_data[:max_samples]


def check_dap_training_logs():
    """检查 DAP 训练日志"""
    print("\n[1/4] 检查 DAP 训练状态...")

    trainer_state_path = SCRIPT_DIR / "sfks_bert_dap" / "checkpoint-7419" / "trainer_state.json"

    if not trainer_state_path.exists():
        # 尝试其他 checkpoint
        for ckpt in ["checkpoint-7419", "checkpoint-4946"]:
            alt_path = SCRIPT_DIR / "sfks_bert_dap" / ckpt / "trainer_state.json"
            if alt_path.exists():
                trainer_state_path = alt_path
                break

    if not trainer_state_path.exists():
        print("  ⚠ 找不到训练状态文件")
        return None

    with open(trainer_state_path, 'r') as f:
        state = json.load(f)

    log_history = state.get("log_history", [])

    # 提取训练损失
    train_losses = []
    eval_losses = []

    for entry in log_history:
        if "loss" in entry:
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])

    print(f"  训练步数: {state.get('global_step', 'N/A')}")
    print(f"  训练轮次: {state.get('epoch', 'N/A')}")

    if train_losses:
        print(f"  初始训练损失: {train_losses[0]:.4f}")
        print(f"  最终训练损失: {train_losses[-1]:.4f}")
        loss_reduction = train_losses[0] - train_losses[-1]
        print(f"  损失下降: {loss_reduction:.4f} ({loss_reduction/train_losses[0]*100:.1f}%)")

        if loss_reduction < 0:
            print("  ⚠ 警告: 训练损失没有下降，预训练可能有问题！")
        elif loss_reduction / train_losses[0] < 0.1:
            print("  ⚠ 警告: 损失下降不明显 (<10%)，预训练效果可能有限")
        else:
            print("  ✓ 训练损失正常下降")

    if eval_losses:
        print(f"  初始验证损失: {eval_losses[0]:.4f}")
        print(f"  最终验证损失: {eval_losses[-1]:.4f}")

    return {
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
    }


def check_model_weights():
    """检查 DAP 模型权重是否与原始 BERT 不同"""
    print("\n[2/4] 检查模型权重差异...")

    if not DAP_MODEL_PATH.exists():
        print(f"  ⚠ DAP 模型不存在: {DAP_MODEL_PATH}")
        return False

    # 加载原始 BERT
    print("  加载原始 BERT...")
    original_model = AutoModelForMaskedLM.from_pretrained(ORIGINAL_BERT)

    # 加载 DAP 模型
    print("  加载 DAP 模型...")
    dap_model = AutoModelForMaskedLM.from_pretrained(str(DAP_MODEL_PATH))

    # 比较权重
    original_params = dict(original_model.named_parameters())
    dap_params = dict(dap_model.named_parameters())

    total_diff = 0
    num_params = 0
    max_diff = 0
    max_diff_layer = ""

    for name in original_params:
        if name in dap_params:
            diff = (original_params[name] - dap_params[name]).abs().mean().item()
            total_diff += diff
            num_params += 1

            if diff > max_diff:
                max_diff = diff
                max_diff_layer = name

    avg_diff = total_diff / num_params if num_params > 0 else 0

    print(f"  参数层数: {num_params}")
    print(f"  平均权重差异: {avg_diff:.6f}")
    print(f"  最大差异层: {max_diff_layer}")
    print(f"  最大差异值: {max_diff:.6f}")

    if avg_diff < 1e-6:
        print("  ⚠ 警告: 权重几乎没有变化，DAP 可能没有正确训练！")
        return False
    elif avg_diff < 1e-4:
        print("  ⚠ 警告: 权重变化很小，DAP 效果可能有限")
        return True
    else:
        print("  ✓ 权重有明显变化，DAP 训练正常")
        return True

    # 清理内存
    del original_model, dap_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def quick_downstream_test(test_data: List[Dict], max_samples: int = 100):
    """快速下游任务测试（不使用 LoRA，直接用分类头）"""
    print("\n[3/4] 快速下游任务对比测试...")

    test_data = test_data[:max_samples]

    results = {}

    for model_name, model_path in [
        ("原始 BERT", ORIGINAL_BERT),
        ("DAP BERT", str(DAP_MODEL_PATH)),
    ]:
        print(f"\n  测试: {model_name}")

        if model_path == str(DAP_MODEL_PATH) and not DAP_MODEL_PATH.exists():
            print(f"    ⚠ 模型不存在，跳过")
            continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForMultipleChoice.from_pretrained(model_path)
            model.to(DEVICE)
            model.eval()
        except Exception as e:
            print(f"    ⚠ 加载失败: {e}")
            continue

        y_true = []
        y_pred = []

        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        reverse_map = {0: "A", 1: "B", 2: "C", 3: "D"}

        for item in test_data:
            question = item["question"]
            options = item["options"]

            first_sentences = []
            second_sentences = []

            for key in ["A", "B", "C", "D"]:
                first_sentences.append(question)
                second_sentences.append(f"{key}. {options.get(key, '')}")

            inputs = tokenizer(
                first_sentences,
                second_sentences,
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt"
            )

            inputs = {k: v.unsqueeze(0).to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                try:
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                    pred_idx = torch.argmax(probs).item()
                    predicted = reverse_map[pred_idx]
                except:
                    predicted = "A"  # 默认

            y_true.append(item["answer"])
            y_pred.append(predicted)

        accuracy = accuracy_score(
            [label_map[y] for y in y_true],
            [label_map[y] for y in y_pred]
        )

        f1 = f1_score(
            [label_map[y] for y in y_true],
            [label_map[y] for y in y_pred],
            average='weighted'
        )

        print(f"    准确率: {accuracy:.4f}")
        print(f"    F1: {f1:.4f}")

        results[model_name] = {"accuracy": accuracy, "f1": f1}

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def analyze_dap_issues():
    """分析 DAP 可能的问题"""
    print("\n[4/4] DAP 问题分析...")

    issues = []

    # 检查数据量
    texts_count = 0
    for filename in ["0_train.json", "1_train.json"]:
        filepath = DATA_DIR / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        texts_count += 1

    print(f"\n  预训练数据量: ~{texts_count * 5} 条文本 (题目+选项)")

    if texts_count * 5 < 50000:
        issues.append("数据量较少 (<50000)，DAP 效果可能有限")

    # 检查训练轮次
    trainer_state = None
    for ckpt in ["checkpoint-7419", "checkpoint-4946"]:
        path = SCRIPT_DIR / "sfks_bert_dap" / ckpt / "trainer_state.json"
        if path.exists():
            with open(path) as f:
                trainer_state = json.load(f)
            break

    if trainer_state:
        epochs = trainer_state.get("epoch", 0)
        if epochs < 3:
            issues.append(f"训练轮次不足 ({epochs} epochs)，建议至少 3-5 轮")

    # 总结
    print("\n  潜在问题:")
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("    未发现明显问题")

    print("\n  建议:")
    print("    1. 如果 DAP 效果不好，可以直接使用原始 BERT")
    print("    2. 增加预训练数据（添加更多法律文本）")
    print("    3. 增加预训练轮次到 5-10 轮")
    print("    4. 尝试使用已有的法律预训练模型（如 THUDM/Lawformer）")


def main():
    # 1. 检查训练日志
    training_info = check_dap_training_logs()

    # 2. 检查权重差异
    weights_changed = check_model_weights()

    # 3. 加载测试数据
    print("\n加载测试数据...")
    test_data = load_test_data(DATA_DIR, max_samples=100)
    print(f"  加载 {len(test_data)} 条测试数据")

    # 4. 快速下游测试
    downstream_results = quick_downstream_test(test_data)

    # 5. 问题分析
    analyze_dap_issues()

    # 总结
    print("\n" + "=" * 60)
    print("DAP 验证总结")
    print("=" * 60)

    if downstream_results:
        if "原始 BERT" in downstream_results and "DAP BERT" in downstream_results:
            orig = downstream_results["原始 BERT"]
            dap = downstream_results["DAP BERT"]

            f1_diff = dap["f1"] - orig["f1"]

            print(f"\n  原始 BERT F1: {orig['f1']:.4f}")
            print(f"  DAP BERT F1:  {dap['f1']:.4f}")
            print(f"  差异: {f1_diff:+.4f}")

            if f1_diff > 0.02:
                print("\n  ✓ 结论: DAP 预训练有效，提升了模型效果")
            elif f1_diff > -0.02:
                print("\n  ⚠ 结论: DAP 效果不明显，可能需要更多数据/训练")
            else:
                print("\n  ✗ 结论: DAP 反而降低了效果，建议使用原始 BERT")
                print("     可能原因:")
                print("     - 预训练数据量不足")
                print("     - 训练轮次不够")
                print("     - 任务不匹配（MLM vs 选择题）")

    # 保存结果
    output_path = SCRIPT_DIR / "dap_validation_results.json"
    results = {
        "training_info": training_info,
        "weights_changed": weights_changed,
        "downstream_results": downstream_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n✓ 结果保存至: {output_path}")


if __name__ == "__main__":
    main()
