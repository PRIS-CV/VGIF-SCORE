"""
Generate paper-ready analysis figures for the GenMovie QA experiment.

The script reads the benchmark metadata CSV and the referenced
*_qa_eval_dependency_rounds.json files, then writes PNG/PDF figures plus a few
CSV tables for inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_QA_BASE = REPO_DIR / "data" / "genmovie_benchmark_v1"
DEFAULT_METADATA_CSV = DEFAULT_QA_BASE / "metadata" / "qa_eval_results.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_paper_figs"

MODEL_DISPLAY = {
    "cogvideox": "CogVideoX",
    "infinitystar": "InfinityStar",
    "kling": "Kling",
    "ltx2": "LTX-2",
    "mochi_1": "Mochi-1",
    "pixverse_v6": "PixVerse V6",
    "seedance2_0": "Seedance 2.0",
    "viduq3_turbo": "Vidu Q3 Turbo",
    "wan2_2": "Wan2.2",
    "wan2_7": "Wan2.7",
}

MODEL_COLORS = {
    "CogVideoX": "#D55E00",
    "InfinityStar": "#0072B2",
    "Kling": "#009E73",
    "LTX-2": "#CC79A7",
    "Mochi-1": "#E69F00",
    "PixVerse V6": "#56B4E9",
    "Seedance 2.0": "#6A3D9A",
    "Vidu Q3 Turbo": "#B15928",
    "Wan2.2": "#1B9E77",
    "Wan2.7": "#7570B3",
}

QUESTION_DISPLAY = {
    "location": "Location",
    "entity": "Entity",
    "attribute": "Attribute",
    "action": "Action",
    "state": "State",
    "causal": "Causal",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paper-ready figures from dependency-round QA results."
    )
    parser.add_argument("--qa-base", type=Path, default=DEFAULT_QA_BASE)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--depth-cap",
        type=int,
        default=8,
        help="Depths above this value are merged into a final N+ bucket.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def strip_parenthetical(value: Any) -> str:
    text = str(value or "Unknown").strip()
    return re.sub(r"\s*\(.*?\)\s*$", "", text).strip() or "Unknown"


def question_label(value: Any) -> str:
    key = str(value or "other").strip().lower()
    return QUESTION_DISPLAY.get(key, key.replace("_", " ").title())


def compute_depths(results: list[dict[str, Any]]) -> dict[str, int]:
    deps: dict[str, list[str]] = {}
    for row in results:
        qid = row.get("id")
        if not isinstance(qid, str):
            continue
        dependency_ids = row.get("dependency_ids")
        if isinstance(dependency_ids, list):
            deps[qid] = [str(item) for item in dependency_ids if isinstance(item, str)]
        else:
            deps[qid] = re.findall(r"q\d+", str(row.get("dependency") or ""))

    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(qid: str) -> int:
        if qid in memo:
            return memo[qid]
        if qid in visiting:
            return 0
        visiting.add(qid)
        parents = [parent for parent in deps.get(qid, []) if parent in deps]
        memo[qid] = 0 if not parents else 1 + max(depth(parent) for parent in parents)
        visiting.remove(qid)
        return memo[qid]

    return {qid: depth(qid) for qid in deps}


def depth_bin(depth: int, cap: int) -> int:
    return min(int(depth), cap)


def depth_label(depth: int, cap: int) -> str:
    return f"{cap}+" if depth >= cap else str(depth)


def normal_ci(correct: float, total: float) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / total) * 100.0


def load_question_rows(metadata_csv: Path, qa_base: Path, depth_cap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as file_obj:
        for meta in csv.DictReader(file_obj):
            if meta.get("question_mode") != "dependency-rounds":
                continue

            json_path = qa_base / str(meta.get("qa_json_path") or "")
            if not json_path.is_file():
                continue

            payload = load_json(json_path)
            if payload.get("success") is not True or not isinstance(payload.get("results"), list):
                continue

            model = MODEL_DISPLAY.get(meta.get("model_id"), str(meta.get("model_id")))
            depths = compute_depths(payload["results"])
            macro_domain = strip_parenthetical(meta.get("macro_domain") or payload.get("matched_macro_domain"))
            micro_domain = strip_parenthetical(meta.get("micro_domain") or payload.get("matched_micro_domain"))

            for result in payload["results"]:
                qid = result.get("id")
                if not isinstance(qid, str):
                    continue
                raw_correct = bool(result.get("answer_match"))
                final_correct = bool(result.get("correct"))
                dependency_passed = bool(result.get("dependency_passed"))
                is_blocked = raw_correct and not dependency_passed
                depth_value = depths.get(qid, 0)
                rows.append(
                    {
                        "clip_id": meta.get("clip_id"),
                        "model_id": meta.get("model_id"),
                        "model": model,
                        "prompt_id": meta.get("prompt_id"),
                        "macro_domain": macro_domain,
                        "micro_domain": micro_domain,
                        "question_id": qid,
                        "question_type": question_label(result.get("type")),
                        "depth": depth_value,
                        "depth_bin": depth_bin(depth_value, depth_cap),
                        "depth_label": depth_label(depth_bin(depth_value, depth_cap), depth_cap),
                        "raw_correct": raw_correct,
                        "final_correct": final_correct,
                        "dependency_blocked": is_blocked,
                        "error_category": (
                            "Correct"
                            if final_correct
                            else "Dependency blocked"
                            if is_blocked
                            else "Answer mismatch"
                        ),
                    }
                )

    if not rows:
        raise SystemExit(f"No dependency-round QA rows found from {metadata_csv}")
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png")
    plt.close(fig)


def save_tables(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "question_level_records.csv", index=False, encoding="utf-8-sig")

    depth = (
        df.groupby(["depth_bin", "depth_label"])
        .agg(
            total=("final_correct", "size"),
            raw_correct=("raw_correct", "sum"),
            final_correct=("final_correct", "sum"),
            dependency_blocked=("dependency_blocked", "sum"),
        )
        .reset_index()
        .sort_values("depth_bin")
    )
    depth["raw_accuracy_percent"] = depth["raw_correct"] / depth["total"] * 100
    depth["final_accuracy_percent"] = depth["final_correct"] / depth["total"] * 100
    depth["dependency_blocked_percent"] = depth["dependency_blocked"] / depth["total"] * 100
    depth.to_csv(output_dir / "table_depth_accuracy.csv", index=False, encoding="utf-8-sig")

    model = (
        df.groupby("model")
        .agg(
            total=("final_correct", "size"),
            raw_correct=("raw_correct", "sum"),
            final_correct=("final_correct", "sum"),
            dependency_blocked=("dependency_blocked", "sum"),
        )
        .reset_index()
    )
    model["raw_accuracy_percent"] = model["raw_correct"] / model["total"] * 100
    model["final_accuracy_percent"] = model["final_correct"] / model["total"] * 100
    model = model.sort_values("final_accuracy_percent", ascending=False)
    model.to_csv(output_dir / "table_model_overall.csv", index=False, encoding="utf-8-sig")


def model_order(df: pd.DataFrame) -> list[str]:
    order = (
        df.groupby("model")["final_correct"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return list(order)


def plot_depth_accuracy(df: pd.DataFrame, output_dir: Path, depth_cap: int) -> None:
    grouped = (
        df.groupby(["depth_bin", "depth_label"])
        .agg(
            total=("final_correct", "size"),
            raw_correct=("raw_correct", "sum"),
            final_correct=("final_correct", "sum"),
            dependency_blocked=("dependency_blocked", "sum"),
        )
        .reset_index()
        .sort_values("depth_bin")
    )
    grouped["raw_accuracy"] = grouped["raw_correct"] / grouped["total"] * 100
    grouped["final_accuracy"] = grouped["final_correct"] / grouped["total"] * 100
    grouped["raw_ci"] = grouped.apply(lambda r: normal_ci(r["raw_correct"], r["total"]), axis=1)
    grouped["final_ci"] = grouped.apply(lambda r: normal_ci(r["final_correct"], r["total"]), axis=1)
    grouped["blocked_rate"] = grouped["dependency_blocked"] / grouped["total"] * 100

    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    ax.fill_between(
        x,
        grouped["final_accuracy"] - grouped["final_ci"],
        grouped["final_accuracy"] + grouped["final_ci"],
        color="#0072B2",
        alpha=0.16,
        linewidth=0,
    )
    ax.errorbar(
        x,
        grouped["final_accuracy"],
        yerr=grouped["final_ci"],
        marker="o",
        color="#0072B2",
        linewidth=2.0,
        capsize=3,
        label="Dependency-aware accuracy",
    )
    ax.errorbar(
        x,
        grouped["raw_accuracy"],
        yerr=grouped["raw_ci"],
        marker="s",
        color="#D55E00",
        linewidth=1.7,
        capsize=3,
        label="Raw answer accuracy",
    )
    ax.plot(
        x,
        grouped["blocked_rate"],
        marker="^",
        color="#6A3D9A",
        linewidth=1.5,
        label="Dependency-blocked rate",
    )

    for idx, row in grouped.iterrows():
        ax.text(
            x[idx],
            max(row["raw_accuracy"], row["final_accuracy"]) + 5,
            f"n={int(row['total'])}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(grouped["depth_label"])
    ax.set_xlabel("Dependency depth")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.legend(loc="upper right", frameon=True, edgecolor="#CCCCCC")
    ax.set_title("QA Accuracy Declines with Dependency Depth", fontweight="bold", pad=8)
    save_figure(fig, output_dir, "fig1_depth_accuracy_raw_final")


def plot_model_depth_heatmap(df: pd.DataFrame, output_dir: Path, depth_cap: int) -> None:
    agg = (
        df.groupby(["model", "depth_bin"])
        .agg(total=("final_correct", "size"), correct=("final_correct", "sum"))
        .reset_index()
    )
    agg["accuracy"] = agg["correct"] / agg["total"] * 100
    order = model_order(df)
    depth_bins = sorted(df["depth_bin"].unique())
    pivot = agg.pivot(index="model", columns="depth_bin", values="accuracy").reindex(order)
    counts = agg.pivot(index="model", columns="depth_bin", values="total").reindex(order)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(pivot[depth_bins].values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)

    for row_index, model in enumerate(pivot.index):
        for col_index, depth_value in enumerate(depth_bins):
            value = pivot.loc[model, depth_value]
            count = counts.loc[model, depth_value]
            if pd.isna(value):
                continue
            color = "white" if value >= 60 else "#1A1A1A"
            ax.text(
                col_index,
                row_index,
                f"{value:.0f}\n({int(count)})",
                ha="center",
                va="center",
                fontsize=6.8,
                color=color,
            )

    ax.set_xticks(np.arange(len(depth_bins)))
    ax.set_xticklabels([depth_label(int(item), depth_cap) for item in depth_bins])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Dependency depth")
    ax.set_title("Model Accuracy by Dependency Depth", fontweight="bold", pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Accuracy (%)")
    save_figure(fig, output_dir, "fig2_model_depth_heatmap")


def plot_error_propagation(df: pd.DataFrame, output_dir: Path) -> None:
    categories = ["Correct", "Answer mismatch", "Dependency blocked"]
    colors = {
        "Correct": "#009E73",
        "Answer mismatch": "#D55E00",
        "Dependency blocked": "#6A3D9A",
    }
    counts = (
        df.groupby(["depth_bin", "depth_label", "error_category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("depth_bin")
    )
    for category in categories:
        if category not in counts:
            counts[category] = 0
    totals = counts[categories].sum(axis=1)
    percents = counts[categories].div(totals, axis=0) * 100

    x = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    bottom = np.zeros(len(counts))
    for category in categories:
        ax.bar(
            x,
            percents[category],
            bottom=bottom,
            color=colors[category],
            edgecolor="white",
            linewidth=0.4,
            label=category,
        )
        bottom += percents[category].to_numpy()

    for idx, total in enumerate(totals):
        ax.text(idx, 102, f"n={int(total)}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(counts["depth_label"])
    ax.set_xlabel("Dependency depth")
    ax.set_ylabel("Question share (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    ax.set_title("Where Errors Come From as Dependency Depth Increases", fontweight="bold", pad=8)
    save_figure(fig, output_dir, "fig3_error_propagation_by_depth")


def plot_shallow_deep_drop(df: pd.DataFrame, output_dir: Path) -> None:
    work = df[df["depth_bin"] >= 0].copy()
    work["depth_group"] = np.where(
        work["depth"] <= 1,
        "Shallow",
        np.where(work["depth"] >= 3, "Deep", "Middle"),
    )
    grouped = (
        work[work["depth_group"].isin(["Shallow", "Deep"])]
        .groupby(["model", "depth_group"])
        .agg(total=("final_correct", "size"), correct=("final_correct", "sum"))
        .reset_index()
    )
    grouped["accuracy"] = grouped["correct"] / grouped["total"] * 100
    pivot = grouped.pivot(index="model", columns="depth_group", values="accuracy")
    pivot = pivot.dropna(subset=["Shallow", "Deep"])
    pivot["drop"] = pivot["Shallow"] - pivot["Deep"]
    pivot = pivot.sort_values("drop", ascending=False)

    y = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.hlines(y, pivot["Deep"], pivot["Shallow"], color="#BDBDBD", linewidth=2.0, zorder=1)
    ax.scatter(pivot["Shallow"], y, color="#0072B2", s=44, label="Depth 0-1", zorder=3)
    ax.scatter(pivot["Deep"], y, color="#D55E00", s=44, label="Depth >=3", zorder=3)

    for idx, (_, row) in enumerate(pivot.iterrows()):
        ax.text(
            max(row["Shallow"], row["Deep"]) + 1.5,
            idx,
            f"{row['drop']:.1f} pt",
            va="center",
            fontsize=7.5,
            color="#4D4D4D",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("QA accuracy (%)")
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    ax.legend(loc="lower right", frameon=True, edgecolor="#CCCCCC")
    ax.set_title("Accuracy Drop from Shallow to Deep Dependencies", fontweight="bold", pad=8)
    save_figure(fig, output_dir, "fig4_model_shallow_deep_drop")


def plot_question_type_accuracy(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        df.groupby("question_type")
        .agg(
            total=("final_correct", "size"),
            raw_correct=("raw_correct", "sum"),
            final_correct=("final_correct", "sum"),
            dependency_blocked=("dependency_blocked", "sum"),
        )
        .reset_index()
    )
    grouped["raw_accuracy"] = grouped["raw_correct"] / grouped["total"] * 100
    grouped["final_accuracy"] = grouped["final_correct"] / grouped["total"] * 100
    grouped = grouped.sort_values("final_accuracy", ascending=True)

    y = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.barh(
        y - 0.18,
        grouped["raw_accuracy"],
        height=0.34,
        color="#E69F00",
        label="Raw answer accuracy",
    )
    ax.barh(
        y + 0.18,
        grouped["final_accuracy"],
        height=0.34,
        color="#0072B2",
        label="Dependency-aware accuracy",
    )
    for idx, row in grouped.iterrows():
        y_index = grouped.index.get_loc(idx)
        ax.text(
            max(row["raw_accuracy"], row["final_accuracy"]) + 1.2,
            y_index,
            f"n={int(row['total'])}",
            va="center",
            fontsize=7.5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(grouped["question_type"])
    ax.set_xlabel("QA accuracy (%)")
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    ax.legend(loc="lower right", frameon=True, edgecolor="#CCCCCC")
    ax.set_title("Accuracy by Question Type", fontweight="bold", pad=8)
    save_figure(fig, output_dir, "fig5_question_type_accuracy")


def plot_video_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    per_video = (
        df.groupby(["model", "clip_id"])
        .agg(total=("final_correct", "size"), correct=("final_correct", "sum"))
        .reset_index()
    )
    per_video["accuracy"] = per_video["correct"] / per_video["total"] * 100
    order = model_order(df)
    values = [per_video.loc[per_video["model"] == model, "accuracy"].to_numpy() for model in order]

    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    box = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#1A1A1A", "linewidth": 1.4},
        whiskerprops={"color": "#666666"},
        capprops={"color": "#666666"},
    )
    for patch, model in zip(box["boxes"], order):
        patch.set_facecolor(MODEL_COLORS.get(model, "#999999"))
        patch.set_alpha(0.62)
        patch.set_edgecolor("#555555")

    rng = np.random.default_rng(4)
    for idx, (model, vals) in enumerate(zip(order, values), start=1):
        jitter = rng.normal(loc=idx, scale=0.045, size=len(vals))
        ax.scatter(jitter, vals, s=10, color="#333333", alpha=0.35, linewidths=0)

    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=28, ha="right")
    ax.set_ylabel("Per-video QA accuracy (%)")
    ax.set_ylim(-2, 102)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.set_title("Per-Video Accuracy Distribution by Model", fontweight="bold", pad=8)
    save_figure(fig, output_dir, "fig6_per_video_accuracy_distribution")


def plot_micro_domain_accuracy(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        df.groupby("micro_domain")
        .agg(
            total=("final_correct", "size"),
            correct=("final_correct", "sum"),
            blocked=("dependency_blocked", "sum"),
        )
        .reset_index()
    )
    grouped["accuracy"] = grouped["correct"] / grouped["total"] * 100
    grouped["blocked_rate"] = grouped["blocked"] / grouped["total"] * 100
    grouped = grouped.sort_values("accuracy", ascending=True)

    y = np.arange(len(grouped))
    labels = ["\n".join(textwrap.wrap(item, width=30)) for item in grouped["micro_domain"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.barh(y, grouped["accuracy"], color="#009E73", alpha=0.82, label="Accuracy")
    ax.scatter(grouped["blocked_rate"], y, color="#6A3D9A", s=45, label="Dependency-blocked rate", zorder=3)

    for idx, row in grouped.iterrows():
        y_index = grouped.index.get_loc(idx)
        ax.text(row["accuracy"] + 1.2, y_index, f"{row['accuracy']:.1f}%", va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rate (%)")
    max_rate = max(float(grouped["accuracy"].max()), float(grouped["blocked_rate"].max()))
    x_max = min(105, max(50, math.ceil((max_rate + 8) / 10) * 10))
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10 if x_max <= 60 else 20))
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    ax.legend(loc="lower right", frameon=True, edgecolor="#CCCCCC")
    ax.set_title("Accuracy Across Narrative Subdomains", fontweight="bold", pad=8)
    save_figure(fig, output_dir, "fig7_micro_domain_accuracy")


def main() -> int:
    args = parse_args()
    qa_base = args.qa_base.resolve()
    metadata_csv = args.metadata_csv.resolve()
    output_dir = args.output_dir.resolve()

    df = load_question_rows(metadata_csv, qa_base, args.depth_cap)
    save_tables(df, output_dir)

    plot_depth_accuracy(df, output_dir, args.depth_cap)
    plot_model_depth_heatmap(df, output_dir, args.depth_cap)
    plot_error_propagation(df, output_dir)
    plot_shallow_deep_drop(df, output_dir)
    plot_question_type_accuracy(df, output_dir)
    plot_video_distribution(df, output_dir)
    plot_micro_domain_accuracy(df, output_dir)

    overall_raw = df["raw_correct"].mean() * 100
    overall_final = df["final_correct"].mean() * 100
    blocked = df["dependency_blocked"].mean() * 100
    print(f"Rows: {len(df)}")
    print(f"Videos: {df['clip_id'].nunique()}, models: {df['model'].nunique()}")
    print(f"Overall raw accuracy: {overall_raw:.2f}%")
    print(f"Overall dependency-aware accuracy: {overall_final:.2f}%")
    print(f"Dependency-blocked rate: {blocked:.2f}%")
    print(f"Figures written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
