#!/usr/bin/env python3
"""
数据质量验证脚本
检查参考资料与问题的相关度
"""

import json
from pathlib import Path
from collections import Counter

def simple_relevance_score(text1, text2, min_length=3):
    """
    简单的相关性评分（基于关键词匹配）
    """
    # 分词（简化方式：分割成长度 >= 3 的词）
    words1 = set([w for w in text1.split() if len(w) >= 3])
    words2 = set([w for w in text2.split() if len(w) >= 3])

    if not words1 or not words2:
        return 0.0

    # 计算 Jaccard 相似度
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0

def main():
    data_file = Path(__file__).parent / "qa_train_retrieved_sample.jsonl"

    print("=" * 80)
    print("数据质量验证报告")
    print("=" * 80)

    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                data.append(json.loads(line))

    print(f"\n总数据量：{len(data)} 条\n")

    # 分析前 20 条
    print("=" * 80)
    print("前 20 条数据的相关度分析")
    print("=" * 80)

    relevance_scores = []
    low_relevance_indices = []

    for i in range(min(20, len(data))):
        item = data[i]
        question = item.get('question', '')
        context = item.get('context', '')
        answer_label = item.get('answer_option_label', '')

        # 计算相关度
        score = simple_relevance_score(question, context)
        relevance_scores.append(score)

        if score < 0.2:
            low_relevance_indices.append(i)

        print(f"\n样本 {i + 1}:")
        print(f"  问题: {question[:60]}...")
        print(f"  答案: {answer_label}")
        print(f"  相关度: {score:.3f} {'?? 太低!' if score < 0.2 else '?'}")
        print(f"  Context 长度: {len(context)} 字")

        if score < 0.2:
            print(f"  Context 开头: {context[:80]}...")

    # 统计
    print("\n" + "=" * 80)
    print("统计汇总")
    print("=" * 80)

    print(f"\n前 20 条数据的相关度统计：")
    print(f"  平均相关度: {sum(relevance_scores) / len(relevance_scores):.3f}")
    print(f"  最高相关度: {max(relevance_scores):.3f}")
    print(f"  最低相关度: {min(relevance_scores):.3f}")
    print(f"  相关度 < 0.2 的样本: {len(low_relevance_indices)} 条 ({len(low_relevance_indices)/20*100:.1f}%)")

    # 全量统计
    print(f"\n全量数据相关度统计（所有 {len(data)} 条）：")
    all_scores = []
    low_count = 0

    for item in data:
        question = item.get('question', '')
        context = item.get('context', '')
        score = simple_relevance_score(question, context)
        all_scores.append(score)
        if score < 0.2:
            low_count += 1

    print(f"  平均相关度: {sum(all_scores) / len(all_scores):.3f}")
    print(f"  相关度 < 0.2 的比例: {low_count / len(data) * 100:.1f}%")
    print(f"  相关度分布:")

    # 分桶统计
    buckets = {
        "极低 (0.0-0.1)": sum(1 for s in all_scores if s < 0.1),
        "很低 (0.1-0.2)": sum(1 for s in all_scores if 0.1 <= s < 0.2),
        "低 (0.2-0.3)": sum(1 for s in all_scores if 0.2 <= s < 0.3),
        "中 (0.3-0.4)": sum(1 for s in all_scores if 0.3 <= s < 0.4),
        "较高 (0.4+)": sum(1 for s in all_scores if s >= 0.4),
    }

    for bucket, count in buckets.items():
        percentage = count / len(data) * 100
        bar = "█" * int(percentage / 2)
        print(f"    {bucket:15s}: {count:4d} ({percentage:5.1f}%) {bar}")

    # 诊断
    print("\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)

    avg_score = sum(all_scores) / len(all_scores)

    if avg_score < 0.15:
        print("\n? 数据质量极差！")
        print("  - 参考资料与问题的相关度极低（平均仅 {:.3f}）".format(avg_score))
        print("  - 超过 {}% 的数据完全无关".format(low_count / len(data) * 100))
        print("  - 这解释了为什么 F1 分数只有 35%（接近随机猜测）")
        print("\n  建议：")
        print("  1. 检查数据生成过程（retrieve_for_sfks.py）")
        print("  2. 重新运行检索以获取相关的参考资料")
        print("  3. 或使用高质量的法律教材而不是判决书")

    elif avg_score < 0.3:
        print("\n??  数据质量较差")
        print("  - 参考资料与问题的相关度较低（平均 {:.3f}）".format(avg_score))
        print("  - 约 {}% 的数据相关度不足".format(low_count / len(data) * 100))
        print("\n  建议：")
        print("  1. 尝试数据清理（过滤相关度极低的样本）")
        print("  2. 重新训练并观察 F1 分数是否改善")
        print("  3. 如果还是不够好，考虑重新生成数据")

    else:
        print("\n? 数据质量还可以")
        print("  - 参考资料与问题的平均相关度：{:.3f}".format(avg_score))
        print("  - 低质量样本比例：{:.1f}%".format(low_count / len(data) * 100))
        print("\n  建议：")
        print("  1. 继续调整模型和训练参数")
        print("  2. 或者尝试数据清理过滤低相关度样本")

    print("\n" + "=" * 80)

    # 保存详细报告
    report = {
        "total_samples": len(data),
        "avg_relevance": sum(all_scores) / len(all_scores),
        "low_relevance_count": low_count,
        "low_relevance_percentage": low_count / len(data) * 100,
        "distribution": buckets,
    }

    with open(Path(__file__).parent / "data_quality_report.json", 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n? 详细报告已保存至 data_quality_report.json")

if __name__ == "__main__":
    main()
