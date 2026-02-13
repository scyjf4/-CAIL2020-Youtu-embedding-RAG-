# -*- coding: utf-8 -*-
"""构建司法考试 SFKS 选择题到 QA 子集的数据脚本。

读取 data/sfks/0_train.json 和 1_train.json，将原始选择题数据
转换为统一的 QA 格式，输出到 data/sfks/qa_train.jsonl。

保留字段：
- id: 统一为 "source-原始id" 格式
- question: 来自原始 statement（多选题可加“可多选”提示）
- context: 题干 + 各个选项，拼成一段文本
- options: 结构化选项列表 [{"label": "A", "text": "..."}, ...]
- answer: 正确选项的文本（单选为字符串，多选为字符串列表）
- answer_option_label: 正确选项的标签（单选为字符串，多选为标签列表）
- subject: 原始 subject
- type: 原始 type
- source: "0_train" 或 "1_train"

使用方式：
    python build_sfks_qa_dataset.py

运行后将在 data/sfks/ 下生成 qa_train.jsonl 文件。
"""

import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple, Union

# 相对路径以当前脚本所在目录为基准
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据目录在项目根目录的 data/sfks/ 下
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "sfks")

INPUT_FILES = [
    ("0_train.json", "0_train"),
    ("1_train.json", "1_train"),
]

OUTPUT_FILE = os.path.join(DATA_DIR, "qa_train.jsonl")

OPTION_LABELS = "ABCDEFGH"


def parse_option_list(option_list: Any) -> List[Dict[str, str]]:
    """将原始 option_list 解析为统一结构: [{"label": "A", "text": "..."}, ...].

    兼容多种格式:
    - list[str]: ["A. xxx", "B. yyy", ...] 或 ["xxx", "yyy", ...]
    - dict[str, str]: {"A": "xxx", "B": "yyy", ...}
    """
    options: List[Dict[str, str]] = []

    if isinstance(option_list, dict):
        # dict 形式: key 是 label, value 是文本
        for key in sorted(option_list.keys()):
            label = str(key).strip().upper()
            if not label:
                continue
            text = str(option_list[key]).strip()
            if not text:
                continue
            options.append({"label": label, "text": text})
        return options

    if isinstance(option_list, (list, tuple)):
        for idx, raw in enumerate(option_list):
            if raw is None:
                continue
            s = str(raw).strip()
            if not s:
                continue

            # 尝试匹配 "A. xxx" / "A、xxx" / "A) xxx" / "A）xxx" 等形式
            m = re.match(r"^([A-Za-z])\s*[\.、\)）]\s*(.*)$", s)
            if m:
                label = m.group(1).upper()
                text = m.group(2).strip()
            else:
                # 退化: 使用索引映射到 A/B/C/D...
                label = OPTION_LABELS[idx] if idx < len(OPTION_LABELS) else str(idx)
                text = s

            if not text:
                continue
            options.append({"label": label, "text": text})

        return options

    # 未知类型，返回空
    return []


def normalize_answer_labels(answer: Any) -> List[str]:
    """将原始 answer 标准化为标签列表, 如 ["A"] 或 ["A", "C"]."""
    if answer is None:
        return []

    # 数组形式: ["A", "C"]
    if isinstance(answer, (list, tuple)):
        labels: List[str] = []
        for a in answer:
            if a is None:
                continue
            s = str(a).strip().upper()
            if not s:
                continue
            # 可能是 "AC" 这种连写, 展开
            for ch in s:
                if ch in OPTION_LABELS and ch not in labels:
                    labels.append(ch)
        return labels

    # 其他类型统一转字符串
    s = str(answer).strip().upper()
    if not s:
        return []

    # 单个字母: "A"
    if len(s) == 1 and s in OPTION_LABELS:
        return [s]

    # 像 "AC" 这种
    labels: List[str] = []
    for ch in s:
        if ch in OPTION_LABELS and ch not in labels:
            labels.append(ch)
    return labels


def build_context(statement: str, options: List[Dict[str, str]]) -> str:
    """构造 context 文本: 题干 + 每个选项一行."""
    lines = [statement.strip()] if statement else []
    for opt in options:
        label = opt.get("label", "")
        text = opt.get("text", "")
        if not text:
            continue
        if label:
            lines.append(f"{label}. {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def iter_examples() -> Iterable[Dict[str, Any]]:
    """遍历 0_train.json / 1_train.json, 逐条生成 QA 样本."""
    for filename, source in INPUT_FILES:
        input_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(input_path):
            print(f"[WARN] 输入文件不存在: {input_path}")
            continue

        print(f"[INFO] 处理文件: {input_path} (source={source})")

        # 文件可能是按行一个 json, 也可能是一个 json 数组
        with open(input_path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == "[":
                # json 数组
                data = json.load(f)
                records = data
            else:
                # json lines
                records = [json.loads(line) for line in f if line.strip()]

        for raw in records:
            raw_id = raw.get("id")
            statement = str(raw.get("statement", "")).strip()
            option_list = raw.get("option_list")
            raw_answer = raw.get("answer")
            subject = raw.get("subject")
            q_type = raw.get("type")

            if raw_id is None or not statement or option_list is None or raw_answer is None:
                # 关键字段缺失, 跳过
                continue

            # 统一 id
            ex_id = f"{source}-{raw_id}"

            # 解析选项
            options = parse_option_list(option_list)
            if len(options) < 2:
                continue

            # 解析答案标签
            answer_labels = normalize_answer_labels(raw_answer)
            if not answer_labels:
                continue

            # 构建 label -> text 映射
            label_to_text = {opt["label"]: opt["text"] for opt in options if opt.get("label")}

            # 过滤掉在 options 中找不到的 label
            valid_labels = [lab for lab in answer_labels if lab in label_to_text]
            if not valid_labels:
                continue

            # 构造 answer 文本
            if len(valid_labels) == 1:
                answer_text: Union[str, List[str]] = label_to_text[valid_labels[0]]
                answer_option_label: Union[str, List[str]] = valid_labels[0]
            else:
                # 多选: 文本列表, 保持 label 顺序
                answer_text = [label_to_text[lab] for lab in valid_labels]
                answer_option_label = valid_labels

            # question: 直接用 statement, 多选题可加提示
            question = statement
            if len(valid_labels) > 1:
                # 多选题提示, 避免重复添加
                if "可多选" not in question:
                    question = question.rstrip("。?？") + "（可多选）"

            # context: 题干 + 选项
            context = build_context(statement, options)

            example: Dict[str, Any] = {
                "id": ex_id,
                "question": question,
                "context": context,
                "options": options,
                "answer": answer_text,
                "answer_option_label": answer_option_label,
                "subject": subject,
                "type": q_type,
                "source": source,
            }

            yield example


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    num_examples = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for ex in iter_examples():
            out_f.write(json.dumps(ex, ensure_ascii=False))
            out_f.write("\n")
            num_examples += 1

    print(f"[INFO] 写入样本数: {num_examples}")
    print(f"[INFO] 输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
