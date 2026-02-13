# -*- coding: utf-8 -*-
import json
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
PLOT_DIR = BASE_DIR / "analysis" / "plots"


def analyze_sfks():
    """司法考试：0_train.json / 1_train.json (JSON Lines).

    Fields per line (from spec):
      - answer: 正确答案
      - id: 题目ID
      - option_list: 题目每个选项的描述 (list of strings)
      - statement: 题干
    """
    stats = []
    for fname in ["0_train.json", "1_train.json"]:
        path = BASE_DIR / "sfks" / fname
        if not path.exists():
            continue
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                items.append(obj)
        for item in items:
            # 根据官方说明，题干字段是 statement
            q = item.get("statement", "") or ""
            # 选项列表字段是 option_list
            options = item.get("option_list", []) or []
            stats.append({
                "file": fname,
                "task": "sfks",
                "q_len": len(str(q)),
                "avg_opt_len": sum(len(str(o)) for o in options) / max(len(options), 1),
            })
    return pd.DataFrame(stats)


def analyze_sfzy():
    """司法摘要：train.json (JSON Lines).

    Each line is a JSON object with fields:
      - id: 样本ID
      - summary: 摘要内容 (string)
      - text: 原文句子列表，每个元素包含 sentence / label
    我们统计：摘要长度 summary_len，以及原文总长度 doc_len (所有 sentence 拼接后长度)。
    """
    path = BASE_DIR / "sfzy" / "train.json"
    if not path.exists():
        return pd.DataFrame()

    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(obj)

    stats = []
    for item in items:
        summary = item.get("summary", "") or ""
        # text 是句子列表，每个元素有 sentence / label
        text_list = item.get("text", []) or []
        sentences = []
        for sent_obj in text_list:
            s = None
            if isinstance(sent_obj, dict):
                s = sent_obj.get("sentence")
            elif isinstance(sent_obj, str):
                s = sent_obj
            if s:
                sentences.append(str(s))
        full_doc = "".join(sentences)
        stats.append({
            "task": "sfzy",
            "doc_len": len(full_doc),
            "summ_len": len(str(summary)),
        })
    return pd.DataFrame(stats)


def analyze_ydlj():
    """阅读理解：task1 train.json (JSON list or JSON Lines).

    Fields per sample (from spec):
      - _id: 案例唯一标识
      - context: 案例内容 (HotpotQA 风格，单个篇章：标题 + 句子列表)
      - question: 问题
      - answer: 回答
      - supporting_facts: 证据句 (不在这里用)

    我们统计：question 长度 q_len，以及 context 展开后的总长度 context_len。
    """
    path = BASE_DIR / "ydlj" / "train.json"
    if not path.exists():
        return pd.DataFrame()

    # 支持 JSON 数组或 JSON Lines 两种格式
    items = []
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()
        if text:
            try:
                loaded = json.loads(text)
                if isinstance(loaded, list):
                    items = loaded
                else:
                    items = [loaded]
            except json.JSONDecodeError:
                # JSON Lines
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    stats = []
    for item in items:
        q = item.get("question", "") or ""
        context = item.get("context")
        context_text = ""
        # context 与 HotpotQA 一致：形如 [ [title, [sent1, sent2, ...]] ]
        if isinstance(context, list) and context:
            # 只取第一个篇章
            first_par = context[0]
            # first_par 可能是 [title, [sent1, sent2, ...]] 或 dict
            if isinstance(first_par, list) and len(first_par) >= 2:
                title = first_par[0]
                sents = first_par[1]
                parts = []
                if isinstance(title, str):
                    parts.append(title)
                if isinstance(sents, list):
                    for s in sents:
                        if isinstance(s, str):
                            parts.append(s)
                context_text = "".join(parts)
            elif isinstance(first_par, dict):
                # 兜底：如果是 dict，就把所有值拼起来
                parts = []
                for v in first_par.values():
                    if isinstance(v, str):
                        parts.append(v)
                    elif isinstance(v, list):
                        for s in v:
                            if isinstance(s, str):
                                parts.append(s)
                context_text = "".join(parts)
        elif isinstance(context, str):
            context_text = context

        stats.append({
            "task": "ydlj",
            "q_len": len(str(q)),
            "context_len": len(context_text),
        })
    return pd.DataFrame(stats)


def analyze_lbwj():
    """论辩挖掘：SMP-CAIL2020-text-train.csv + SMP-CAIL2020-train2.csv.

    - SMP-CAIL2020-text-train.csv: 全部句子，包含 columns:
        sentence_id, text_id, position, sentence
    - SMP-CAIL2020-train2.csv: 论点对，包含 columns:
        id, text_id, sc, A, B, C, D, E, answer

    这里做两类统计：
      1) 文本级：SMP-CAIL2020-text-train.csv 中 sentence 的长度分布；
      2) 论点对级：sc 及候选辩方论点 A-E 的长度；answer 作为标签分布。
    """
    # 1) 文本级句子长度
    text_path = BASE_DIR / "lbwj" / "SMP-CAIL2020-text-train.csv"
    text_df = None
    if text_path.exists():
        text_df = pd.read_csv(text_path, encoding="utf-8", on_bad_lines="skip")
        if "sentence" in text_df.columns:
            text_df["sentence_len"] = text_df["sentence"].astype(str).str.len()
        else:
            text_df["sentence_len"] = 0

    # 2) 论点对级
    pair_path = BASE_DIR / "lbwj" / "SMP-CAIL2020-train2.csv"
    if not pair_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(pair_path, encoding="utf-8", on_bad_lines="skip")

    # sc 是诉方论点；A-E 是候选辩方论点
    df["sc_len"] = df.get("sc", "").astype(str).str.len() if "sc" in df.columns else 0
    for opt in ["A", "B", "C", "D", "E"]:
        if opt in df.columns:
            df[f"{opt}_len"] = df[opt].astype(str).str.len()
        else:
            df[f"{opt}_len"] = 0

    # answer 为正确辩方论点对应的选项（A-E）
    if "answer" in df.columns:
        df["label"] = df["answer"].astype(str)
    else:
        df["label"] = "unknown"

    # 返回论点对级的简单统计视图（用于 plot）
    pair_stats = df[["sc_len", "A_len", "B_len", "C_len", "D_len", "E_len", "label"]].copy()
    pair_stats["task"] = "lbwj"

    # 如果需要也可以把 text_df 返回，但当前主脚本只用 pair_stats
    return pair_stats


def plot_histogram(data, column, title, filename, bins=50):
    if data.empty or column not in data.columns:
        return
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    series = data[column].dropna()
    if series.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel("Length (characters)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150)
    plt.close()


def plot_bar(categories, counts, title, filename):
    if not categories or not counts:
        return
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(categories, counts)
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150)
    plt.close()


if __name__ == "__main__":
    sfks_df = analyze_sfks()
    sfzy_df = analyze_sfzy()
    ydlj_df = analyze_ydlj()
    lbwj_df = analyze_lbwj()

    # Print summary in English
    print("Judicial Exam (sfks) samples:", len(sfks_df), "avg statement length:", sfks_df["q_len"].mean() if not sfks_df.empty else 0)
    print("Judicial Summarization (sfzy) samples:", len(sfzy_df), "avg document length:", sfzy_df["doc_len"].mean() if not sfzy_df.empty else 0)
    print("Reading Comprehension (ydlj) samples:", len(ydlj_df), "avg context length:", ydlj_df["context_len"].mean() if not ydlj_df.empty else 0)
    print("Argument Mining (lbwj) samples:", len(lbwj_df), "avg prosecution argument length:", lbwj_df["sc_len"].mean() if not lbwj_df.empty else 0)

    out_dir = BASE_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not sfks_df.empty:
        sfks_df.to_csv(out_dir / "sfks_stats.csv", index=False)
    if not sfzy_df.empty:
        sfzy_df.to_csv(out_dir / "sfzy_stats.csv", index=False)
    if not ydlj_df.empty:
        ydlj_df.to_csv(out_dir / "ydlj_stats.csv", index=False)
    if not lbwj_df.empty:
        lbwj_df.to_csv(out_dir / "lbwj_stats.csv", index=False)

    # Generate plots with English titles
    plot_histogram(sfks_df, "q_len", "Judicial Exam: statement length distribution", "sfks_q_len_hist.png")
    plot_histogram(sfzy_df, "doc_len", "Judicial Summarization: document length distribution", "sfzy_doc_len_hist.png")
    plot_histogram(ydlj_df, "context_len", "Reading Comprehension: context length distribution", "ydlj_context_len_hist.png")
    plot_histogram(lbwj_df, "sc_len", "Argument Mining: prosecution argument length distribution", "lbwj_sc_len_hist.png")

    if not lbwj_df.empty:
        label_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        labels_mapped = lbwj_df["label"].map(label_map).fillna(lbwj_df["label"])
        label_counts = labels_mapped.value_counts()
        plot_bar(
            label_counts.index.tolist(),
            label_counts.values.tolist(),
            "Argument Mining: correct candidate option distribution",
            "lbwj_label_distribution.png"
        )