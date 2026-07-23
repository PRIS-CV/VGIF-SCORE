"""
================================================================================
 VGIF 项目 — QA 正确率随 prompt 中相对位置变化的可视化分析
================================================================================
 分析维度: prompt 中答案对应概念首次出现的位置 / prompt总词数 = 相对位置
 假设: 越靠后的概念，视频生成越容易出错 → QA准确率随相对位置增大而下降
================================================================================
"""
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ══════════════════════════════════════════════════════════════════════════════
# 0. 全局配置
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "scripts" / "output_figs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 干净数据目录 (由 scripts/organize_clean_data.py 生成)
CLEAN_QA_DIR = BASE_DIR / "data" / "clean_qa"

DOMAIN_ABBR = {
    "Commercial & Product Showcase": "Product",
    "Creative & Surreal Expression": "Creative",
    "Dynamics & Physical Interaction": "Dynamics",
    "Emotion & Atmosphere Expression": "Emotion",
    "Narrative & Cinematic Storytelling": "Narrative",
    "Performance & Sports Embodied Motion": "Performance",
    "Performance, Sports & Embodied Motion": "Performance",
    "Spatial Composition & Scene Orchestration": "Spatial",
    "Travel & Nature Living World": "Nature",
    "Travel, Nature & Living World": "Nature",
}

MODEL_COLORS = {
    "Kling-V3": "#E64B35", "Seedance-2.0": "#4DBBD5", "Wan-2.7": "#00A087",
    "ViduQ3-Turbo": "#DC0000", "PixVerse-V6": "#8491B4", "LTX-2.0": "#3C5488",
    "Wan2.2-A14B": "#7E6148", "HyVideo-1.5": "#F39B7F", "LongCat-Video": "#91D1C2",
    "Mochi-1": "#B09C85", "CogVideoX-1.5": "#FF7F00", "MAGI-1": "#6A3D9A",
    "URSA": "#B15928", "InfinityStar": "#33A02C",
}
MODEL_MARKERS = {
    "Kling-V3": "o", "Seedance-2.0": "s", "Wan-2.7": "D", "ViduQ3-Turbo": "^",
    "PixVerse-V6": "v", "LTX-2.0": "<", "Wan2.2-A14B": ">", "HyVideo-1.5": "p",
    "LongCat-Video": "h", "Mochi-1": "H", "CogVideoX-1.5": "*", "MAGI-1": "X",
    "URSA": "P", "InfinityStar": "d",
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. 核心算法: 从问题文本定位 prompt 中的对应位置
# ══════════════════════════════════════════════════════════════════════════════

# 常见问句前缀/后缀，用于清洗问题文本以提取关键短语
_Q_PREFIXES = [
    'is there a ', 'is there an ', 'is the ', 'is a ', 'does the ', 'does a ',
    'are the ', 'are there ', 'do the ', 'was the ', 'were the ',
    'can the ', 'will the ', 'should the ', 'has the ', 'have the ',
]
_Q_SUFFIXES = [
    ' shown in the scene', ' visible in the scene', ' present in the scene',
    ' displayed', ' visible', ' shown', ' present in the video',
    ' depicted', ' observed', ' in the scene', ' in the video',
    ' appearing in the scene', ' appear in the scene', ' seen',
    ' clearly visible', ' clearly shown', ' clearly depicted',
]

def find_relative_position(question: str, prompt_lower: str, n_words: int) -> float:
    """
    从问题文本中提取关键短语，在prompt中定位，计算相对位置。

    算法:
      1. 去掉问句的固定前缀 (Is there a..., Does the... 等)
      2. 去掉问句的固定后缀 (...shown in the scene? 等)
      3. 在 prompt 中查找清洗后的文本
      4. 若找到，返回 词位置 / 总词数
      5. 若未找到，回退到最长公共子串匹配
      6. 仍失败返回 NaN

    返回: 0~1 之间的相对位置，或 NaN
    """
    q_lower = question.lower().rstrip('?')
    # 去前缀
    for p in _Q_PREFIXES:
        if q_lower.startswith(p):
            q_lower = q_lower[len(p):]
            break
    # 去后缀
    for s in _Q_SUFFIXES:
        if q_lower.endswith(s):
            q_lower = q_lower[:-len(s)]
            break

    # 精确匹配
    idx = prompt_lower.find(q_lower)
    if idx >= 0:
        word_pos = len(prompt_lower[:idx].split())
        return word_pos / max(n_words, 1)

    # 回退: 最长公共子串
    q_words = q_lower.split()
    best_pos = None
    best_len = 0
    for i in range(len(q_words)):
        for j in range(i + 2, len(q_words) + 1):
            phrase = ' '.join(q_words[i:j])
            pos = prompt_lower.find(phrase)
            if pos >= 0 and len(phrase) > best_len:
                best_len = len(phrase)
                best_pos = len(prompt_lower[:pos].split())

    if best_pos is not None:
        return best_pos / max(n_words, 1)
    return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 2. 数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_position_data():
    """
    从 data/clean_qa/ 目录加载所有模型的QA数据，计算每题的相对位置。

    data/clean_qa/ 中的JSON已经过整理（去重、去旧），直接读取即可。
    """
    if not CLEAN_QA_DIR.exists():
        raise FileNotFoundError(
            f"干净数据目录不存在: {CLEAN_QA_DIR}\n"
            f"请先运行 scripts/organize_clean_data.py 生成干净数据"
        )

    rows = []
    for model_dir in sorted(CLEAN_QA_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for json_path in model_dir.glob("*.json"):
            try:
                d = json.load(open(json_path, "r", encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not d.get("success") or "results" not in d:
                continue
            rows.extend(_process_video(d, model_name))
        print(f"  {model_name:18s}: {len(rows)} question records")

    df = pd.DataFrame(rows)
    n_before = len(df)
    df = df.dropna(subset=["rel_position"]).copy()
    print(f"\n  总题目: {n_before:,} → 成功定位: {len(df):,} ({len(df)/n_before*100:.1f}%)")
    return df


def _process_video(data, model_name):
    """处理单个视频JSON，返回逐题的记录列表。"""
    prompt = data.get("prompt", "")
    if not prompt:
        return []
    prompt_lower = prompt.lower()
    n_words = len(prompt.split())

    domain_full = data.get("matched_macro_domain", "Unknown")
    domain_en = domain_full.split("(")[0].strip() if "(" in domain_full else domain_full
    domain_short = DOMAIN_ABBR.get(domain_en, domain_en)

    video_name = re.sub(
        r'_(?:gemini.*?_|g\d+.*?_|latestcfg_)?qa_eval_dependency_rounds\.json$', '',
        Path(data.get("output_file", data.get("video_file", ""))).name
    )

    recs = []
    for r in data.get("results", []):
        rel_pos = find_relative_position(r["question"], prompt_lower, n_words)
        recs.append({
            "model": model_name,
            "domain": domain_short,
            "video": video_name,
            "rel_position": rel_pos,
            "correct": r.get("correct", False),
            "question_type": r.get("type", "other"),
        })
    return recs


# ══════════════════════════════════════════════════════════════════════════════
# 3. 数据聚合
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_position_bins(df, n_bins=10):
    """按相对位置分箱，聚合准确率。"""
    bins = np.linspace(0, 1, n_bins + 1)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(n_bins)]
    df = df.copy()
    df["bin"] = pd.cut(df["rel_position"], bins=bins, labels=labels, include_lowest=True)

    agg = df.groupby(["model", "bin"]).agg(
        correct=("correct", "sum"),
        total=("correct", "count"),
    ).reset_index()
    agg["accuracy"] = agg["correct"] / agg["total"] * 100
    agg["bin_mid"] = agg["bin"].apply(
        lambda x: (float(x.split("-")[0]) + float(x.split("-")[1])) / 2
    )
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# 4. 绘图
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig_pos1_main(df, agg):
    """主图: QA正确率 vs 相对位置 (所有模型折线图)"""
    model_order = df.groupby("model")["correct"].mean().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(16, 9))
    for model in model_order:
        sub = agg[agg["model"] == model].sort_values("bin_mid")
        if sub.empty:
            continue
        color = MODEL_COLORS.get(model, "#888888")
        marker = MODEL_MARKERS.get(model, "o")
        ax.plot(sub["bin_mid"], sub["accuracy"], marker=marker, color=color,
                linewidth=1.5, markersize=6, label=model, zorder=3,
                markeredgecolor="white", markeredgewidth=0.3)

    ax.set_xlabel("Relative Position in Prompt", fontsize=12)
    ax.set_ylabel("QA Accuracy (%)", fontsize=12)
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.set_xlim(-0.02, 1.02)
    ax.legend(ncol=2, frameon=True, edgecolor="#cccccc", loc="upper right",
              fontsize=7.5, title="Model", title_fontsize=8)
    ax.set_title("QA Accuracy by Relative Position in Prompt", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_pos1_accuracy_vs_position.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig_pos1_accuracy_vs_position.png", format="png")
    plt.close(fig)
    print("  [OK] 图pos1: 主图 — QA正确率 vs 相对位置")


def plot_fig_pos2_heatmap(df, agg):
    """热力图: 模型 × 相对位置 (无n标注)"""
    pivot = agg.pivot_table(index="model", columns="bin", values="accuracy")
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    cmap = LinearSegmentedColormap.from_list("blue_white_yellow",
        ["#A8D8EA", "#F8F8F8", "#FFE066"])
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="#333333", fontweight="medium")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("QA Accuracy (%)", fontsize=10)
    ax.set_title("QA Accuracy Heatmap: Model × Relative Position", fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_pos2_heatmap.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig_pos2_heatmap.png", format="png")
    plt.close(fig)
    print("  [OK] 图pos2: 热力图 — 模型×相对位置")


def plot_fig_pos3_early_vs_late(df):
    """前1/3 vs 后1/3 分组柱状图"""
    df = df.copy()
    df["group"] = df["rel_position"].apply(
        lambda x: "Early (0.00-0.33)" if x <= 0.33 else ("Late (0.67-1.00)" if x >= 0.67 else "Middle (0.34-0.66)")
    )
    agg = df.groupby(["model", "group"]).agg(
        correct=("correct", "sum"), total=("correct", "count"),
    ).reset_index()
    agg["accuracy"] = agg["correct"] / agg["total"] * 100

    model_order = df.groupby("model")["correct"].mean().sort_values(ascending=False).index.tolist()
    groups = ["Early (0.00-0.33)", "Middle (0.34-0.66)", "Late (0.67-1.00)"]
    colors = ["#4393C3", "#F4A582", "#D6604D"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(model_order))
    width = 0.25
    for i, (grp, c) in enumerate(zip(groups, colors)):
        sub = agg[agg["group"] == grp].set_index("model").reindex(model_order)
        ax.bar(x + i * width, sub["accuracy"].values, width, color=c,
               edgecolor="white", linewidth=0.3, label=grp)

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("QA Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, edgecolor="#cccccc", fontsize=9, loc="upper right")
    ax.set_title("Early vs Late Position Accuracy by Model", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_pos3_early_vs_late.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig_pos3_early_vs_late.png", format="png")
    plt.close(fig)
    print("  [OK] 图pos3: 前vs后 — 分组柱状图")


def plot_fig_pos4_overall_drop(df, agg):
    """整体下降曲线 + 样本分布"""
    overall = df.groupby(pd.cut(df["rel_position"], np.linspace(0, 1, 11), include_lowest=True)).agg(
        correct=("correct", "sum"), total=("correct", "count"),
    ).reset_index()
    overall.columns = ["bin", "correct", "total"]
    overall["accuracy"] = overall["correct"] / overall["total"] * 100
    overall["bin_mid"] = overall["bin"].apply(
        lambda x: (float(str(x).split(",")[0].strip("([")) + float(str(x).split(",")[1].strip("])"))) / 2
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.plot(overall["bin_mid"], overall["accuracy"], "o-", color="#2166AC",
             linewidth=2.5, markersize=9, markerfacecolor="white",
             markeredgecolor="#2166AC", markeredgewidth=1.5)
    for _, r in overall.iterrows():
        ax1.annotate(f"{r['accuracy']:.1f}%", (r["bin_mid"], r["accuracy"]),
                     textcoords="offset points", xytext=(0, 14), ha="center",
                     fontsize=8.5, fontweight="bold", color="#333333")
    ax1.set_xlabel("Relative Position in Prompt", fontsize=11)
    ax1.set_ylabel("QA Accuracy (%)", fontsize=11)
    ax1.set_ylim(-5, 110)
    ax1.set_title("Overall Accuracy by Position", fontweight="bold", fontsize=11)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.5)

    bin_counts = df.groupby(pd.cut(df["rel_position"], np.linspace(0, 1, 11), include_lowest=True)).size().reset_index(name="count")
    bin_counts.columns = ["bin", "count"]
    bin_counts["bin_str"] = bin_counts["bin"].astype(str).apply(
        lambda x: f"{float(x.split(',')[0].strip('([')):.1f}-{float(x.split(',')[1].strip('])')):.1f}"
    )
    ax2.bar(range(len(bin_counts)), bin_counts["count"], color="#92C5DE", edgecolor="#4393C3", linewidth=0.5)
    ax2.set_xticks(range(len(bin_counts)))
    ax2.set_xticklabels(bin_counts["bin_str"], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Number of Questions", fontsize=11)
    ax2.set_title("Question Count by Position", fontweight="bold", fontsize=11)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.suptitle("Overall Position-Accuracy Analysis (All Models Pooled)", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_pos4_overall_position.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig_pos4_overall_position.png", format="png")
    plt.close(fig)
    print("  [OK] 图pos4: 整体位置趋势 + 分布")


# ── 4.5 图pos5: 按六种题型分类的位置趋势 ──
def plot_fig_pos5_by_question_type(df):
    """
    六种维度(主体/位置/特征/动作/状态/因果)的QA正确率随相对位置变化。

    题型说明:
      - Entity   (主体): 实体存在性，如 "Is there a dog in the scene?"
      - Location (位置): 空间位置关系，如 "Is the cat on the table?"
      - Attribute(特征): 属性描述，如 "Is the car red?"
      - Action   (动作): 动作识别，如 "Does the person wave?"
      - State    (状态): 状态判断，如 "Is the glass full?"
      - Causal   (因果): 因果关系，如 "Did X cause Y?"
    """
    type_map = {
        "entity": "Entity", "location": "Location", "attribute": "Attribute",
        "action": "Action", "state": "State", "causal": "Causal",
    }
    df = df.copy()
    df["qtype"] = df["question_type"].map(type_map).fillna("Other")

    # 按题型 × 位置分箱聚合
    bins = np.linspace(0, 1, 11)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]
    df["bin"] = pd.cut(df["rel_position"], bins=bins, labels=labels, include_lowest=True)

    agg = df.groupby(["qtype", "bin"]).agg(
        correct=("correct", "sum"), total=("correct", "count"),
    ).reset_index()
    agg["accuracy"] = agg["correct"] / agg["total"] * 100
    agg["bin_mid"] = agg["bin"].apply(
        lambda x: (float(x.split("-")[0]) + float(x.split("-")[1])) / 2
    )

    qtype_order = ["Entity", "Attribute", "Action", "State", "Causal", "Location"]
    qtype_colors = {
        "Entity": "#8DD3C7", "Attribute": "#FFFFB3", "Action": "#BEBADA",
        "State": "#FB8072", "Causal": "#80B1D3", "Location": "#FDB462",
    }

    fig, ax = plt.subplots(figsize=(7.5, 9))
    for qt in qtype_order:
        sub = agg[agg["qtype"] == qt].sort_values("bin_mid")
        if sub.empty:
            continue
        c = qtype_colors.get(qt, "#888888")
        ax.plot(sub["bin_mid"], sub["accuracy"], "o-", color=c,
                linewidth=2, markersize=6, label=qt,
                markerfacecolor="white", markeredgecolor=c, markeredgewidth=1.5)

    ax.set_xlabel("Relative Position in Prompt", fontsize=12)
    ax.set_ylabel("QA Accuracy (%)", fontsize=12)
    ax.set_ylim(-5, 110)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=True, edgecolor="#cccccc", fontsize=9,
              loc="upper right", title="Question Type", title_fontsize=9)
    ax.set_title("Accuracy by Question Type and Relative Position", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_pos5_by_question_type.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig_pos5_by_question_type.png", format="png")
    plt.close(fig)
    print("  [OK] 图pos5: 按六种题型 — 位置趋势")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("VGIF — QA正确率 vs Prompt相对位置 可视化分析")
    print("=" * 60)
    print()

    print("加载数据并计算相对位置...")
    df = load_position_data()

    # 修正Wan2.2 depth=9噪声 (仅影响该模型在深度图的显示，不影响位置图)
    mask = (df["model"] == "Wan2.2-A14B")
    # 位置图中不需要此修正，跳过

    print(f"\n相对位置分布:")
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        sub = df[(df["rel_position"] >= lo) & (df["rel_position"] < hi)]
        acc = sub["correct"].mean() * 100
        print(f"  {lo:.1f}-{hi:.1f}: {acc:5.1f}% (n={len(sub):,})")

    agg = aggregate_position_bins(df, n_bins=10)
    print(f"\n生成图表...")

    plot_fig_pos1_main(df, agg)
    plot_fig_pos2_heatmap(df, agg)
    plot_fig_pos3_early_vs_late(df)
    plot_fig_pos4_overall_drop(df, agg)
    plot_fig_pos5_by_question_type(df)

    print(f"\n所有图表已保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
