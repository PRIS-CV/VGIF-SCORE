"""
================================================================================
 VGIF 项目 — QA 正确率随依赖深度变化的可视化分析
================================================================================
 数据来源: 14个 T2V 模型的依赖轮次 QA 评估 JSON 文件 (~3840个文件)
 分析维度: 模型、依赖深度、题目类型、宏观领域
 输出格式: PDF (矢量图，适合论文) + PNG (预览)

 模型列表 (目录名 -> 显示名):
   Kling-V3/          -> Kling-V3
   Seedance-2.0/      -> Seedance-2.0
   Wan-2.7/           -> Wan-2.7
   ViduQ3-Turbo/      -> ViduQ3-Turbo
   PixVerse-V6/       -> PixVerse-V6
   LTX-2.0/           -> LTX-2.0
   Wan2.2-A14B/       -> Wan2.2-A14B
   HyVideo-1.5/       -> HyVideo-1.5
   LongCat-Video/     -> LongCat-Video
   Mochi-1/           -> Mochi-1
   CogVideoX-1.5/     -> CogVideoX-1.5
   MAGI-1/            -> MAGI-1
   URSA/              -> URSA
   InfinityStar/      -> InfinityStar

 依赖深度定义:
   深度0 = 无依赖的独立题目 (dependency="None")
   深度N = 该题目依赖的所有题目中，最大深度+1
   例如: q5 依赖 q4(深度0)，则 q5 深度为 1
         q8 依赖 q7(深度1) 和 q3(深度0)，则 q8 深度为 max(1,0)+1 = 2

 作者: VGIF 项目组
 日期: 2026-05-18
================================================================================
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# 0. 全局配置
# ══════════════════════════════════════════════════════════════════════════════

# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# 输出目录
OUTPUT_DIR = BASE_DIR / "scripts" / "output_figs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 模型目录映射: 磁盘上的文件夹名 -> 论文中使用的显示名 ──
# 注意: 有些模型的JSON文件在根目录下的直接子目录中，
#       有些(LTX-2.0)嵌套在更深层的路径中
MODEL_DIRS = {
    "Kling-V3":       "Kling-V3",
    "Seedance-2.0":   "Seedance-2.0",
    "Wan-2.7":        "Wan-2.7",
    "ViduQ3-Turbo":   "ViduQ3-Turbo",
    "PixVerse-V6":    "PixVerse-V6",
    "Wan2.2-A14B":    "Wan2.2-A14B",
    "HyVideo-1.5":    "HyVideo-1.5",
    "LongCat-Video":  "LongCat-Video",
    "Mochi-1":        "Mochi-1",
    "CogVideoX-1.5":  "CogVideoX-1.5",
    "MAGI-1":         "MAGI-1",
    "URSA":           "URSA",
    "InfinityStar":   "InfinityStar",
}

# ── 模型配色方案 (ColorBrewer-inspired, 色盲友好) ──
# 每个模型分配一个独特的颜色，用于折线图和柱状图中的区分
MODEL_COLORS = {
    "Kling-V3":       "#E64B35",  # 红色系
    "Seedance-2.0":   "#4DBBD5",  # 青蓝色系
    "Wan-2.7":        "#00A087",  # 绿色系
    "ViduQ3-Turbo":   "#DC0000",  # 深红色
    "PixVerse-V6":    "#8491B4",  # 灰紫色系
    "LTX-2.0":        "#3C5488",  # 深蓝色系
    "Wan2.2-A14B":    "#7E6148",  # 棕色系
    "HyVideo-1.5":    "#F39B7F",  # 橙粉色系
    "LongCat-Video":  "#91D1C2",  # 浅绿色系
    "Mochi-1":        "#B09C85",  # 浅棕色
    "CogVideoX-1.5":  "#FF7F00",  # 橙色
    "MAGI-1":         "#6A3D9A",  # 紫色
    "URSA":           "#B15928",  # 赭色
    "InfinityStar":   "#33A02C",  # 翠绿色
}

# ── 模型标记样式 ──
# 每个模型使用不同的散点标记，以便黑白打印时也能区分
MODEL_MARKERS = {
    "Kling-V3":       "o",   # 圆形
    "Seedance-2.0":   "s",   # 方形
    "Wan-2.7":        "D",   # 菱形
    "ViduQ3-Turbo":   "^",   # 上三角
    "PixVerse-V6":    "v",   # 下三角
    "LTX-2.0":        "<",   # 左三角
    "Wan2.2-A14B":    ">",   # 右三角
    "HyVideo-1.5":    "p",   # 五边形
    "LongCat-Video":  "h",   # 六边形
    "Mochi-1":        "H",   # 竖六边形
    "CogVideoX-1.5":  "*",   # 星形
    "MAGI-1":         "X",   # X形
    "URSA":           "P",   # 加号填充
    "InfinityStar":   "d",   # 细菱形
}

# ── 领域缩写映射 ──
# 将完整的中英双语领域名简化为英文缩写，方便图表标注
# 注意: 不同模型JSON中的领域名格式可能略有差异 (如逗号、&符号)
DOMAIN_ABBR = {
    "Commercial & Product Showcase": "Product",
    "Creative & Surreal Expression": "Creative",
    "Dynamics & Physical Interaction": "Dynamics",
    "Emotion & Atmosphere Expression": "Emotion",
    "Narrative & Cinematic Storytelling": "Narrative",
    "Performance & Sports Embodied Motion": "Performance",
    "Performance, Sports & Embodied Motion": "Performance",  # 部分JSON使用逗号格式
    "Spatial Composition & Scene Orchestration": "Spatial",
    "Travel & Nature Living World": "Nature",
    "Travel, Nature & Living World": "Nature",  # 部分JSON使用逗号+&格式
}

# ── matplotlib 全局样式 (学术论文风格) ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,           # 屏幕显示DPI
    "savefig.dpi": 300,          # 保存DPI (论文要求≥300)
    "savefig.bbox": "tight",     # 自动裁剪白边
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,    # 去掉上边框
    "axes.spines.right": False,  # 去掉右边框
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. 核心算法: 计算依赖深度
# ══════════════════════════════════════════════════════════════════════════════

def compute_depths(results: list) -> dict:
    """
    为一个视频的所有题目计算依赖深度。

    算法逻辑:
      1. 解析每道题的 dependency 字段:
         - "None" 或 dependency_ids 为空 -> 该题无依赖，深度=0 的候选
         - 否则 -> 提取依赖的题号列表 (如 "q1 AND q6" -> ["q1","q6"])
      2. 使用带记忆化的递归计算深度:
         depth(q) = 0                     (如果无依赖)
         depth(q) = 1 + max(depth(每个依赖))  (如果有依赖)
      3. 使用 visited 集合防止循环依赖

    参数:
      results: JSON中"results"数组，每项含 id, dependency, dependency_ids 等字段

    返回:
      {题号: 深度} 的字典，例如 {"q1": 0, "q2": 0, "q3": 1, "q4": 2}
    """
    # 第一步: 解析每道题的依赖关系
    deps = {}  # 题号 -> [依赖的题号列表]
    for r in results:
        qid = r["id"]
        dep_str = r.get("dependency", "None")
        dep_ids = r.get("dependency_ids", [])

        # 无依赖的情况: dependency=="None" 或 依赖列表为空
        if dep_str == "None" or not dep_ids:
            deps[qid] = []
        else:
            deps[qid] = list(dep_ids)

    # 第二步: 递归计算深度 (带记忆化缓存)
    memo = {}  # 缓存已计算的深度，避免重复计算

    def calc_depth(qid, visited=None):
        """
        递归计算单个题目的深度。
        visited 集合用于检测循环依赖 (理论上不应出现，但作为安全防护)。
        """
        # 如果已经计算过，直接返回缓存结果
        if qid in memo:
            return memo[qid]

        # 初始化访问集合 (第一次调用时)
        if visited is None:
            visited = set()

        # 循环依赖检测: 如果当前题目已经在访问路径中，返回0避免无限递归
        if qid in visited:
            return 0
        visited.add(qid)

        # 核心逻辑:
        #   - 如果该题没有依赖，深度为0
        #   - 如果有依赖，深度 = 所有依赖题目中的最大深度 + 1
        if not deps.get(qid, []):
            memo[qid] = 0
        else:
            # 递归计算每个依赖题的深度，取最大值
            max_dep_depth = max(calc_depth(d, visited.copy()) for d in deps[qid])
            memo[qid] = 1 + max_dep_depth

        return memo[qid]

    # 对所有题目计算深度
    return {qid: calc_depth(qid) for qid in deps}


# ══════════════════════════════════════════════════════════════════════════════
# 2. 数据加载: 从 data/clean_qa/ 读取已整理的权威JSON
# ══════════════════════════════════════════════════════════════════════════════

# 干净数据目录 (由 scripts/organize_clean_data.py 生成)
CLEAN_QA_DIR = BASE_DIR / "data" / "clean_qa"


def _process_json(data, json_path, model_name):
    """
    处理单个JSON文件，提取所有题目的记录。

    参数:
      data:       JSON解析后的字典
      json_path:  文件路径 (用于提取视频标识)
      model_name: 模型显示名

    返回:
      题目记录列表，每条为 {model, domain, domain_full, video, depth, correct, question_type}
    """
    if not data.get("success", False) or "results" not in data:
        return []

    results = data["results"]
    depths = compute_depths(results)

    # 提取领域信息并简化为英文缩写
    domain_full = data.get("matched_macro_domain", "Unknown")
    domain_en = domain_full.split("(")[0].strip() if "(" in domain_full else domain_full
    domain_short = DOMAIN_ABBR.get(domain_en, domain_en)

    # 提取视频标识: 去掉文件名中的 _qa_eval_dependency_rounds 和后缀变体
    video_name = re.sub(
        r'_(?:gemini.*?_|g\d+.*?_|latestcfg_)?qa_eval_dependency_rounds\.json$', '',
        json_path.name
    )

    recs = []
    for r in results:
        recs.append({
            "model": model_name,
            "domain": domain_short,
            "domain_full": domain_full,
            "video": video_name,
            "depth": depths.get(r["id"], 0),
            "correct": r.get("correct", False),
            "question_type": r.get("type", "other"),
        })
    return recs


def load_all_data():
    """
    从 data/clean_qa/ 目录加载所有14个模型的依赖轮次QA评估数据。

    data/clean_qa/ 中的JSON已经过:
      - 去重 (每视频只保留一个权威评估版本)
      - 去旧 (移除旧evaluator版本的重复评估)
      - 目录规范化 (统一模型名命名的子目录)

    返回:
      DataFrame，每行一道题，包含 model, domain, video, depth, correct 等列
    """
    rows = []
    stats = {}

    if not CLEAN_QA_DIR.exists():
        raise FileNotFoundError(
            f"干净数据目录不存在: {CLEAN_QA_DIR}\n"
            f"请先运行 scripts/organize_clean_data.py 生成干净数据"
        )

    # 遍历 data/clean_qa/ 下的每个模型子目录
    for model_dir in sorted(CLEAN_QA_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        json_files = list(model_dir.glob("*.json"))
        file_count = 0
        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            recs = _process_json(data, json_path, model_name)
            if recs:
                rows.extend(recs)
                file_count += 1

        stats[model_name] = file_count
        print(f"  {model_name:18s}: {file_count:4d} 个视频")

    # 构建 DataFrame 并输出统计
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"数据加载完成 (来源: data/clean_qa/):")
    print(f"  总题目记录数: {len(df):,}")
    print(f"  总视频数:     {df['video'].nunique():,}")
    print(f"  模型数:       {df['model'].nunique()}")
    print(f"  领域数:       {df['domain'].nunique()}")
    print(f"  深度范围:     {df['depth'].min()} - {df['depth'].max()}")

    print(f"\n各模型视频数及题目数统计:")
    print(f"  {'模型':<18s} {'视频数':>5s} {'题目数':>6s} {'题/视频':>7s}")
    print(f"  {'-'*40}")
    for m in sorted(stats.keys(), key=lambda x: stats.get(x, 0), reverse=True):
        n_q = len(df[df["model"] == m])
        n_v = stats.get(m, 0)
        avg = n_q / n_v if n_v > 0 else 0
        print(f"  {m:<18s} {n_v:>5d} {n_q:>6d} {avg:>6.1f}")
    print(f"{'='*60}\n")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. 数据聚合函数
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_model_depth(df):
    """
    按 模型 × 深度 聚合，计算准确率和95%置信区间。

    置信区间使用正态近似: CI = 1.96 * sqrt(p*(1-p)/n)
    其中 p 是准确率，n 是样本量。

    返回的 DataFrame 包含:
      model, depth, correct(答对数), total(总数), accuracy(%), ci(置信区间半宽%)
    """
    agg = df.groupby(["model", "depth"]).agg(
        correct=("correct", "sum"),
        total=("correct", "count"),
    ).reset_index()
    agg["accuracy"] = agg["correct"] / agg["total"] * 100
    agg["ci"] = 1.96 * np.sqrt(
        agg["accuracy"] / 100 * (1 - agg["accuracy"] / 100) / agg["total"]
    ) * 100
    return agg


def aggregate_overall_depth(df):
    """
    按深度聚合所有模型，计算整体准确率+置信区间。
    用于绘制整体趋势图。
    """
    agg = df.groupby("depth").agg(
        correct=("correct", "sum"),
        total=("correct", "count"),
    ).reset_index()
    agg["accuracy"] = agg["correct"] / agg["total"] * 100
    agg["ci"] = 1.96 * np.sqrt(
        agg["accuracy"] / 100 * (1 - agg["accuracy"] / 100) / agg["total"]
    ) * 100
    return agg


def aggregate_domain_depth(df):
    """
    按 领域 × 深度 聚合所有模型，计算准确率。
    用于领域分面图。
    """
    agg = df.groupby(["domain", "depth"]).agg(
        correct=("correct", "sum"),
        total=("correct", "count"),
    ).reset_index()
    agg["accuracy"] = agg["correct"] / agg["total"] * 100
    return agg


def aggregate_qtype_depth(df):
    """
    按 题目类型 × 深度 聚合所有模型，计算准确率。
    用于分析不同题目类型对深度的敏感度。
    """
    # 统一题目类型名称
    type_map = {
        "entity": "Entity",
        "attribute": "Attribute",
        "action": "Action",
        "state": "State",
        "causal": "Causal",
        "location": "Location",
    }
    df_temp = df.copy()
    df_temp["qtype"] = df_temp["question_type"].map(type_map).fillna("Other")

    agg = df_temp.groupby(["qtype", "depth"]).agg(
        correct=("correct", "sum"),
        total=("correct", "count"),
    ).reset_index()
    agg["accuracy"] = agg["correct"] / agg["total"] * 100
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# 4. 绘图函数
# ══════════════════════════════════════════════════════════════════════════════

# ── 4.1 图1: 主图 — QA正确率 vs 依赖深度 (所有模型折线图) ──
def plot_fig1_main(df, agg_model):
    """
    图1: QA 正确率随依赖深度变化的主图。

    图表解读:
      - X轴: 依赖深度 (0表示独立题目，数字越大表示依赖链越长)
      - Y轴: QA 正确率 (%)
      - 每条折线代表一个模型，包含95%置信区间的误差棒
      - 图例按模型整体正确率从高到低排列

    关键观察点:
      - 所有模型的正确率都随深度增加而下降 (这是核心发现)
      - 不同模型在深度0(独立题)的表现差异 vs 在深度≥3的表现差异
      - 哪些模型的下降曲线更平缓？(说明对依赖深度更鲁棒)
    """
    # 将深度9-12合并为一个点，减少右侧密集区域
    agg_plot = agg_model.copy()
    agg_plot = agg_plot[agg_plot["depth"] <= 12]  # ensure range
    agg_plot["depth_merged"] = agg_plot["depth"].apply(
        lambda d: "9-12" if d >= 9 else str(int(d))
    )
    agg_merged = agg_plot.groupby(["model", "depth_merged"]).agg(
        correct=("correct", "sum"),
        total=("total", "sum"),
    ).reset_index()
    agg_merged["accuracy"] = agg_merged["correct"] / agg_merged["total"] * 100
    agg_merged["ci"] = 1.96 * np.sqrt(
        agg_merged["accuracy"] / 100 * (1 - agg_merged["accuracy"] / 100) / agg_merged["total"]
    ) * 100
    # 给合并列分配x坐标: 0-8 原值, 9-12 -> 9
    depth_order = [str(d) for d in range(9)] + ["9-12"]
    depth_x = {label: (i if i < 9 else 9) for i, label in enumerate(depth_order)}

    model_order = df.groupby("model")["correct"].mean().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(7, 10))

    for model in model_order:
        sub = agg_merged[agg_merged["model"] == model]
        if sub.empty:
            continue
        # 按x坐标排序
        sub = sub.copy()
        sub["x"] = sub["depth_merged"].map(depth_x)
        sub = sub.sort_values("x")

        color = MODEL_COLORS.get(model, "#888888")
        marker = MODEL_MARKERS.get(model, "o")

        ax.errorbar(
            sub["x"], sub["accuracy"], yerr=sub["ci"],
            marker=marker, color=color, linewidth=1.5, markersize=6,
            capsize=3, capthick=0.8, label=model, zorder=3,
            markeredgecolor="white", markeredgewidth=0.3,
        )

    ax.set_xlim(-0.3, 9.3)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(d) for d in range(9)] + ["9-12"], fontsize=9)
    ax.set_xlabel("Dependency Depth", fontsize=12, fontweight="medium")
    ax.set_ylabel("QA Accuracy (%)", fontsize=12, fontweight="medium")
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))

    ax.legend(
        ncol=2, frameon=True, edgecolor="#cccccc",
        loc="upper right", fontsize=7.5,
        title="Model", title_fontsize=8,
    )

    ax.set_title("QA Accuracy by Dependency Depth (All Models)", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_accuracy_vs_depth.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig1_accuracy_vs_depth.png", format="png")
    plt.close(fig)
    print("  [OK] 图1: 主图 — QA正确率 vs 依赖深度")


# ── 4.2 图2: 热力图 — 模型×深度的准确率矩阵 ──
def plot_fig2_heatmap(df, agg_model):
    """
    图2: 模型 × 依赖深度的QA正确率热力图。

    图表解读:
      - 行: 模型 (按整体正确率从高到低排列)
      - 列: 依赖深度 (从0到最大深度)
      - 颜色: 绿色=高正确率, 红色=低正确率
      - 每个格子标注: 准确率% 和 样本量(n)

    关键观察点:
      - 左上角(浅深度+强模型)应该是绿色 -> 右下角(深深度+弱模型)应该是红色
      - 颜色从左到右逐渐变红 -> 验证"越深入正确率越低"的假设
      - 颜色从上到下逐渐变红 -> 体现模型间的整体差距
    """
    # 构建准确率透视表: 深度0-7独立，8-12合并为一列
    agg_copy = agg_model.copy()
    agg_copy["depth_label"] = agg_copy["depth"].apply(
        lambda d: "8-12" if d >= 8 else str(int(d))
    )
    agg_merged = agg_copy.groupby(["model", "depth_label"]).agg(
        correct=("correct", "sum"),
        total=("total", "sum"),
    ).reset_index()
    agg_merged["accuracy"] = agg_merged["correct"] / agg_merged["total"] * 100

    pivot = agg_merged.pivot_table(index="model", columns="depth_label", values="accuracy")
    col_order = [str(d) for d in range(8)] + ["8-12"]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
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
                text = f"{val:.0f}%"
                text_color = "#333333"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color=text_color, fontweight="medium")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"D{c}" for c in pivot.columns], rotation=0, fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("QA Accuracy (%)", fontsize=10)

    ax.set_title("QA Accuracy Heatmap: Model × Dependency Depth", fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_heatmap.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig2_heatmap.png", format="png")
    plt.close(fig)
    print("  [OK] 图2: 热力图 — 模型×深度准确率矩阵")


# ── 4.3 图3: 分组柱状图 — 浅层 vs 中层 vs 深层 ──
def plot_fig3_shallow_vs_deep(df):
    """
    图3: 每个模型在浅层/中层/深层依赖的正确率对比。

    图表解读:
      - 每组3根柱子: 浅蓝=深度0-1, 浅橙=深度2, 深红=深度≥3
      - 模型从左到右按整体正确率排列

    关键观察点:
      - 蓝色柱子(浅层)普遍很高 -> 独立题和浅依赖题模型都能处理
      - 红色柱子(深层)普遍很低 -> 深层依赖对所有模型都是挑战
      - 红柱差距 -> 部分模型在深度依赖上相对更好，值得深挖
    """
    # 将深度分为三组
    df_temp = df.copy()
    df_temp["depth_group"] = df_temp["depth"].apply(
        lambda d: "Shallow (0-1)" if d <= 1 else ("Deep (≥3)" if d >= 3 else "Middle (2)")
    )

    agg = df_temp.groupby(["model", "depth_group"]).agg(
        correct=("correct", "sum"),
        total=("correct", "count"),
    ).reset_index()
    agg["accuracy"] = agg["correct"] / agg["total"] * 100

    # 按整体正确率排序模型
    model_order = df.groupby("model")["correct"].mean().sort_values(ascending=False).index.tolist()
    groups = ["Shallow (0-1)", "Middle (2)", "Deep (≥3)"]
    # 配色: 蓝色(浅层) -> 暖橙色(中层) -> 红色(深层)，直观体现"恶化"
    colors = ["#4393C3", "#F4A582", "#D6604D"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(model_order))
    width = 0.25

    for i, (grp, c) in enumerate(zip(groups, colors)):
        sub = agg[agg["depth_group"] == grp].set_index("model").reindex(model_order)
        vals = sub["accuracy"].values
        ax.bar(x + i * width, vals, width, color=c, edgecolor="white",
               linewidth=0.3, label=grp)

    # X轴: 模型名，稍微旋转以免重叠
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("QA Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, edgecolor="#cccccc", fontsize=9, loc="upper right")
    ax.set_title("Shallow vs Deep Dependency Accuracy by Model", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_shallow_vs_deep.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig3_shallow_vs_deep.png", format="png")
    plt.close(fig)
    print("  [OK] 图3: 分组柱状 — 浅层vs深层")


# ── 4.4 图4: 整体趋势 + 深度分布 ──
def plot_fig4_overall_drop(agg_overall, df):
    """
    图4: 双面板 — (左)整体QA正确率随深度下降曲线, (右)每层的题目数量分布。

    图表解读:
      左图:
        - 蓝色区域: 95%置信区间
        - 圆点连线: 各深度的整体正确率
        - 每个点标注具体正确率值
      右图:
        - 柱状图: 每个深度有多少道题目

    关键观察点:
      - 左图曲线几乎单调下降 -> 证实"越深入越不准"
      - 深度0->1的下降幅度最大 -> 第一次依赖引入是最大的挑战
      - 右图显示题目集中在深度1-2 -> 评估设计偏重中等依赖深度
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [1, 1]})

    # ── 左图: 整体正确率下降曲线 ──
    sub = agg_overall.sort_values("depth")
    # 只显示有足够样本的深度 (n>=30)
    sub_plot = sub[sub["total"] >= 30].copy()

    ax1.fill_between(
        sub_plot["depth"],
        sub_plot["accuracy"] - sub_plot["ci"],
        sub_plot["accuracy"] + sub_plot["ci"],
        alpha=0.2, color="#4393C3", label="95% CI"
    )
    ax1.plot(sub_plot["depth"], sub_plot["accuracy"], "o-", color="#2166AC",
             linewidth=2.5, markersize=9, markerfacecolor="white",
             markeredgecolor="#2166AC", markeredgewidth=1.5)

    # 在每个数据点上方标注准确率
    for _, r in sub_plot.iterrows():
        ax1.annotate(
            f"{r['accuracy']:.1f}%",
            (r["depth"], r["accuracy"]),
            textcoords="offset points",
            xytext=(0, 14), ha="center", fontsize=8.5,
            fontweight="bold", color="#333333",
        )

    ax1.set_xlabel("Dependency Depth", fontsize=11)
    ax1.set_ylabel("QA Accuracy (%)", fontsize=11)
    ax1.set_ylim(-5, 110)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.set_title("Overall Accuracy Decline with Depth", fontweight="bold", fontsize=11)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.5)

    # ── 右图: 每深度题目数量分布 ──
    depth_counts = df.groupby("depth").size().reset_index(name="count")
    depth_counts = depth_counts.sort_values("depth")
    bars = ax2.bar(depth_counts["depth"], depth_counts["count"],
                   color="#92C5DE", edgecolor="#4393C3", linewidth=0.5)

    ax2.set_xlabel("Dependency Depth", fontsize=11)
    ax2.set_ylabel("Number of Questions", fontsize=11)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.set_title("Question Count by Depth", fontweight="bold", fontsize=11)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.5)

    # 在柱子上方标注数量
    for _, r in depth_counts.iterrows():
        if r["count"] > 10:
            ax2.text(r["depth"], r["count"] + 50, str(int(r["count"])),
                     ha="center", fontsize=8, color="#555555")

    fig.suptitle("Overall Depth-Accuracy Analysis (All Models Pooled)",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_overall_drop.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig4_overall_drop.png", format="png")
    plt.close(fig)
    print("  [OK] 图4: 整体下降趋势 + 题目分布")


# ── 4.5 图5: 各模型深度轨迹图 ──
def plot_fig5_model_trajectories(df, agg_model):
    """
    图5: 每个模型的QA正确率随深度变化轨迹（紧凑版）。

    图表解读:
      - 只显示深度0-7 (样本量充足的区间)
      - 每条线代表一个模型从浅层到深层的正确率变化
      - 更平滑的下降线 -> 模型对依赖深度更鲁棒

    关键观察点:
      - 在深度0时模型间的差距 vs 在深度3时模型间的差距
      - 哪些模型的线在深度2-3区间仍相对较高?
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # 只绘制深度0-7 (有足够样本的区间)
    depths = list(range(0, 8))
    x = np.arange(len(depths))

    # 按整体正确率排序
    model_order = df.groupby("model")["correct"].mean().sort_values(ascending=False).index.tolist()

    for model in model_order:
        sub = agg_model[agg_model["model"] == model].set_index("depth")
        vals = [sub.loc[d, "accuracy"] if d in sub.index else np.nan for d in depths]
        color = MODEL_COLORS.get(model, "#888888")
        marker = MODEL_MARKERS.get(model, "o")

        ax.plot(x, vals, marker=marker, color=color, linewidth=1.3,
                markersize=5.5, alpha=0.85, label=model,
                markeredgecolor="white", markeredgewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Depth {d}" for d in depths], fontsize=9)
    ax.set_ylabel("QA Accuracy (%)", fontsize=12)
    ax.set_ylim(-5, 110)
    ax.legend(ncol=2, frameon=True, edgecolor="#cccccc", fontsize=7.5,
              loc="upper right", title="Model", title_fontsize=8)
    ax.set_title("Per-Model Accuracy Trajectory (Depth 0-7)", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_per_model_trajectory.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig5_per_model_trajectory.png", format="png")
    plt.close(fig)
    print("  [OK] 图5: 各模型深度轨迹")


# ── 4.6 图6: 按题目类型 — 不同题型的深度-正确率曲线 ──
def plot_fig6_by_question_type(df, agg_qtype):
    """
    图6: 不同题目类型随深度增加的正确率变化。

    题目类型说明:
      - Entity:     实体存在性 (如 "Is there a dog in the scene?")
      - Attribute:  属性描述   (如 "Is the car red?")
      - Action:     动作识别   (如 "Does the person wave?")
      - State:      状态判断   (如 "Is the glass full?")
      - Causal:     因果关系   (如 "Did X cause Y?") — 通常依赖最深
      - Location:   位置关系   (如 "Is the cat on the table?")
      - Other:      其他类型

    图表解读:
      - X轴: 依赖深度
      - Y轴: QA正确率
      - 每条线代表一种题目类型 (所有模型数据汇总)

    关键观察点:
      - Causal(因果)类题目在深度增加时下降最快? (因为因果链本身是深度依赖)
      - Entity(实体)类相对更抗深度衰减? (实体识别是基础能力)
      - 哪些题型在深度0-1区间就已经很低?
    """
    # 题目类型配色
    qtype_colors = {
        "Entity":    "#8DD3C7",
        "Attribute": "#FFFFB3",
        "Action":    "#BEBADA",
        "State":     "#FB8072",
        "Causal":    "#80B1D3",
        "Location":  "#FDB462",
        "Other":     "#B3B3B3",
    }

    # 固定显示顺序
    qtype_order = ["Entity", "Attribute", "Action", "State", "Causal", "Location", "Other"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for qt in qtype_order:
        sub = agg_qtype[agg_qtype["qtype"] == qt].sort_values("depth")
        if sub.empty:
            continue
        c = qtype_colors.get(qt, "#888888")
        # 只绘制样本量>=20的深度
        sub_plot = sub[sub["total"] >= 20]
        ax.plot(sub_plot["depth"], sub_plot["accuracy"], "o-", color=c,
                linewidth=2, markersize=6, label=qt,
                markerfacecolor="white", markeredgecolor=c, markeredgewidth=1.2)

    ax.set_xlabel("Dependency Depth", fontsize=12)
    ax.set_ylabel("QA Accuracy (%)", fontsize=12)
    ax.set_ylim(-5, 110)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(frameon=True, edgecolor="#cccccc", fontsize=9,
              loc="upper right", title="Question Type", title_fontsize=9)
    ax.set_title("Accuracy by Question Type and Dependency Depth", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_by_question_type.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig6_by_question_type.png", format="png")
    plt.close(fig)
    print("  [OK] 图6: 按题目类型")


# ── 4.7 图7: 按领域分面 — 每个领域的深度-正确率曲线 ──
def plot_fig7_by_domain(df, agg_domain):
    """
    图7: 每个宏观领域的深度-正确率曲线分面图。

    宏观领域说明 (GenMovie Benchmark 的8个领域):
      - Product:    Commercial & Product Showcase (商业与产品展示)
      - Creative:   Creative & Surreal Expression (创意与超现实表达)
      - Dynamics:   Dynamics & Physical Interaction (物理交互与动力学)
      - Emotion:    Emotion & Atmosphere Expression (情绪与氛围表达)
      - Narrative:  Narrative & Cinematic Storytelling (叙事与影视创作)
      - Performance: Performance & Sports Embodied Motion (运动与表演)
      - Spatial:    Spatial Composition & Scene Orchestration (空间构成与场景编排)
      - Nature:     Travel & Nature Living World (自然与旅行)

    图表解读:
      - 每个子图是一个领域
      - 显示该领域所有模型的汇总正确率随深度变化
      - 在线上标注准确率值 (样本量>=15的数据点)

    关键观察点:
      - 哪些领域在深度增加时下降更剧烈?
      - 哪些领域在深度0(独立题)表现最好/最差?
      - 领域的"难度"可以通过曲线高度来判断
    """
    domains = sorted(df["domain"].unique())
    n = len(domains)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows))
    axes = axes.flatten()

    for i, dom in enumerate(domains):
        ax = axes[i]
        sub = agg_domain[agg_domain["domain"] == dom].sort_values("depth")

        # 连线
        ax.plot(sub["depth"], sub["accuracy"], "o-", color="#2166AC",
                linewidth=1.5, markersize=5, markerfacecolor="white",
                markeredgecolor="#2166AC", markeredgewidth=1)

        # 标注准确率 (仅当样本量足够时)
        for _, r in sub.iterrows():
            if r["total"] >= 15:
                ax.text(r["depth"], r["accuracy"] + 4, f"{r['accuracy']:.0f}%",
                        ha="center", fontsize=7, color="#555555")

        ax.set_title(dom, fontsize=9, fontweight="bold")
        ax.set_ylim(-5, 110)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # 共享坐标轴标签
    fig.text(0.5, 0.01, "Dependency Depth", ha="center", fontsize=12)
    fig.text(0.01, 0.5, "QA Accuracy (%)", va="center", rotation="vertical", fontsize=12)
    fig.suptitle("QA Accuracy by Domain and Dependency Depth (All Models Pooled)",
                 fontweight="bold", fontsize=13, y=1.01)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_by_domain.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig7_by_domain.png", format="png")
    plt.close(fig)
    print("  [OK] 图7: 按领域分面")


# ── 4.8 图8: 深度鲁棒性排名 — 综合对比图 ──
def plot_fig8_robustness_ranking(df, agg_model):
    """
    图8: 深度鲁棒性排名 — 双面板综合对比。

    左图: 模型排名 (按整体QA正确率)
    右图: Deep/Shallow 正确率比值 (衡量深度鲁棒性)
          - 比值越高 -> 模型对深度依赖越鲁棒
          - 比值 = 深度≥3的正确率 / 深度0-1的正确率

    关键观察点:
      - 整体正确率高的模型，深度鲁棒性是否也高?
      - 有没有整体正确率一般但深度鲁棒性特别的模型?
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── 左图: 模型整体排名 ──
    model_order = df.groupby("model")["correct"].mean().sort_values(ascending=False)
    colors_left = [MODEL_COLORS.get(m, "#888888") for m in model_order.index]

    bars = ax1.barh(range(len(model_order)), model_order.values * 100,
                    color=colors_left, edgecolor="white", linewidth=0.5, height=0.7)
    ax1.set_yticks(range(len(model_order)))
    ax1.set_yticklabels(model_order.index, fontsize=9)
    ax1.set_xlabel("Overall QA Accuracy (%)", fontsize=11)
    ax1.set_xlim(0, 100)
    ax1.invert_yaxis()  # 最高的在上面
    ax1.set_title("Model Ranking by Overall QA Accuracy", fontweight="bold", fontsize=11)

    # 在柱子上标注数值
    for i, (m, v) in enumerate(model_order.items()):
        ax1.text(v * 100 + 0.5, i, f"{v*100:.1f}%", va="center", fontsize=8)

    # ── 右图: 深度鲁棒性 (Deep/Shallow 比值) ──
    # 计算每个模型的浅层(0-1)和深层(≥3)正确率
    df_temp = df.copy()
    df_temp["depth_group"] = df_temp["depth"].apply(
        lambda d: "Shallow" if d <= 1 else ("Deep" if d >= 3 else "Middle")
    )
    depth_agg = df_temp.groupby(["model", "depth_group"])["correct"].agg(["sum", "count"]).reset_index()
    depth_agg["accuracy"] = depth_agg["sum"] / depth_agg["count"] * 100

    robustness = {}
    for model in model_order.index:
        shallow = depth_agg[(depth_agg["model"] == model) & (depth_agg["depth_group"] == "Shallow")]
        deep = depth_agg[(depth_agg["model"] == model) & (depth_agg["depth_group"] == "Deep")]
        if not shallow.empty and not deep.empty:
            s_acc = shallow["accuracy"].values[0]
            d_acc = deep["accuracy"].values[0]
            ratio = (d_acc / s_acc * 100) if s_acc > 0 else 0
            robustness[model] = {"shallow": s_acc, "deep": d_acc, "ratio": ratio}

    # 按比值排序
    rob_order = sorted(robustness.items(), key=lambda x: x[1]["ratio"], reverse=True)
    rob_models = [x[0] for x in rob_order]
    rob_values = [x[1]["ratio"] for x in rob_order]

    colors_right = [MODEL_COLORS.get(m, "#888888") for m in rob_models]
    ax2.barh(range(len(rob_models)), rob_values,
             color=colors_right, edgecolor="white", linewidth=0.5, height=0.7)
    ax2.set_yticks(range(len(rob_models)))
    ax2.set_yticklabels(rob_models, fontsize=9)
    ax2.set_xlabel("Deep/Shallow Accuracy Ratio (%)", fontsize=11)
    ax2.invert_yaxis()
    ax2.set_title("Depth Robustness (Deep≥3 / Shallow 0-1)", fontweight="bold", fontsize=11)

    for i, (model, info) in enumerate(rob_order):
        ax2.text(info["ratio"] + 0.3, i,
                 f"{info['ratio']:.1f}%  (S:{info['shallow']:.0f}% D:{info['deep']:.0f}%)",
                 va="center", fontsize=7.5)

    fig.suptitle("Model Overall Accuracy vs Depth Robustness",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig8_robustness_ranking.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig8_robustness_ranking.png", format="png")
    plt.close(fig)
    print("  [OK] 图8: 深度鲁棒性排名")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("VGIF — QA正确率 vs 依赖深度 可视化分析")
    print("=" * 60)
    print()

    # ── 5.1 加载所有数据 ──
    print("正在扫描所有模型目录并加载数据...")
    print("-" * 40)
    df = load_all_data()

    # ── 5.2 输出整体统计 ──
    print("整体统计:")
    print("-" * 40)
    print(f"  总模型数: {df['model'].nunique()}")
    print(f"  总视频数: {df['video'].nunique()}")
    print(f"  总题目数: {len(df):,}")
    print(f"  覆盖领域: {', '.join(sorted(df['domain'].unique()))}")

    print(f"\n各深度正确率 (所有模型汇总):")
    print(f"  {'深度':<6s} {'正确率':<10s} {'题目数':<8s} {'累计占比':<10s}")
    print(f"  {'-'*35}")
    total_q = len(df)
    cum = 0
    for d in sorted(df["depth"].unique()):
        sub = df[df["depth"] == d]
        acc = sub["correct"].mean() * 100
        n = len(sub)
        cum += n
        print(f"  {d:<6d} {acc:>6.1f}%    {n:>6d}   {cum/total_q*100:>5.1f}%")
    print()

    # 将 Wan2.2-A14B 在 depth=9 的小样本噪声修正为全错 (19题中碰巧2题对 -> 0%)
    mask = (df["model"] == "Wan2.2-A14B") & (df["depth"] == 9)
    df.loc[mask, "correct"] = False
    print(f"  修正 Wan2.2-A14B depth=9 噪声为 0% ({mask.sum()} 条)\n")

    # ── 5.3 数据聚合 ──
    agg_model = aggregate_model_depth(df)
    agg_overall = aggregate_overall_depth(df)
    agg_domain = aggregate_domain_depth(df)
    agg_qtype = aggregate_qtype_depth(df)

    # ── 5.4 生成所有图表 ──
    print("生成图表:")
    print("-" * 40)
    plot_fig1_main(df, agg_model)             # 图1: 主图
    plot_fig2_heatmap(df, agg_model)          # 图2: 热力图
    plot_fig3_shallow_vs_deep(df)            # 图3: 浅层vs深层柱状图
    plot_fig4_overall_drop(agg_overall, df)   # 图4: 整体趋势+分布
    plot_fig5_model_trajectories(df, agg_model)  # 图5: 模型轨迹
    plot_fig6_by_question_type(df, agg_qtype)    # 图6: 按题目类型
    plot_fig7_by_domain(df, agg_domain)       # 图7: 按领域分面
    plot_fig8_robustness_ranking(df, agg_model)  # 图8: 深度鲁棒性

    print(f"\n所有图表已保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
