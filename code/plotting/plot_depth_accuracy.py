"""
Publication figures for QA accuracy analysis across all model folders.

This version does not use the 299-video benchmark metadata subset. It scans the
14 canonical model folders in the repository and uses every
*_qa_eval_dependency_rounds.json file found under those folders.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_all_model_figs"
QA_SUFFIX = "_qa_eval_dependency_rounds.json"

MODEL_FOLDERS = [
    "Kling-V3",
    "Seedance-2.0",
    "Wan-2.7",
    "ViduQ3-Turbo",
    "PixVerse-V6",
    "LTX-2.0",
    "Wan2.2-A14B",
    "HyVideo-1.5",
    "LongCat-Video",
    "Mochi-1",
    "CogVideoX-1.5",
    "MAGI-1",
    "URSA",
    "InfinityStar",
]

MODEL_COLORS = {
    "Kling-V3": "#0072B2",
    "Seedance-2.0": "#D55E00",
    "Wan-2.7": "#009E73",
    "ViduQ3-Turbo": "#CC79A7",
    "PixVerse-V6": "#E69F00",
    "LTX-2.0": "#56B4E9",
    "Wan2.2-A14B": "#6A3D9A",
    "HyVideo-1.5": "#8C564B",
    "LongCat-Video": "#17BECF",
    "Mochi-1": "#BCBD22",
    "CogVideoX-1.5": "#1F77B4",
    "MAGI-1": "#FF7F0E",
    "URSA": "#2CA02C",
    "InfinityStar": "#7F7F7F",
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
        "legend.fontsize": 7.5,
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
        description="Plot QA accuracy figures from all canonical model folders."
    )
    parser.add_argument("--repo-dir", type=Path, default=REPO_DIR)
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


def normalize_question_type(value: Any) -> str:
    key = str(value or "other").strip().lower()
    return QUESTION_DISPLAY.get(key, key.replace("_", " ").title())


def extract_dependency_ids(row: dict[str, Any]) -> list[str]:
    dependency_ids = row.get("dependency_ids")
    if isinstance(dependency_ids, list):
        return [item for item in dependency_ids if isinstance(item, str)]
    return re.findall(r"q\d+", str(row.get("dependency") or ""))


def compute_depths(results: list[dict[str, Any]]) -> dict[str, int]:
    deps: dict[str, list[str]] = {}
    for row in results:
        qid = row.get("id")
        if isinstance(qid, str):
            deps[qid] = extract_dependency_ids(row)

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


def depth_label(depth_value: int, cap: int, max_depth: int) -> str:
    if int(depth_value) >= cap:
        return f"{cap}-{max_depth}" if max_depth > cap else str(cap)
    return str(int(depth_value))


def normal_ci(correct: float, total: float) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / total) * 100.0


def find_eval_files(repo_dir: Path) -> list[tuple[str, Path]]:
    eval_files: list[tuple[str, Path]] = []
    for model in MODEL_FOLDERS:
        model_dir = repo_dir / model
        if not model_dir.is_dir():
            continue
        for path in sorted(model_dir.rglob(f"*{QA_SUFFIX}")):
            if path.is_file():
                eval_files.append((model, path))
    return eval_files


def load_question_rows(repo_dir: Path, depth_cap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []

    for model in MODEL_FOLDERS:
        model_files = [path for found_model, path in find_eval_files(repo_dir) if found_model == model]
        loaded_files = 0
        loaded_questions = 0
        skipped_files = 0

        for path in model_files:
            try:
                payload = load_json(path)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                skipped_files += 1
                continue

            results = payload.get("results")
            if not isinstance(results, list) or not results:
                skipped_files += 1
                continue

            loaded_files += 1
            depths = compute_depths(results)
            clip_name = Path(str(payload.get("video_file") or path.name)).stem

            for result in results:
                qid = result.get("id")
                if not isinstance(qid, str):
                    continue
                raw_correct = bool(result.get("answer_match"))
                final_correct = bool(result.get("correct"))
                dependency_passed = bool(result.get("dependency_passed"))
                blocked = raw_correct and not dependency_passed
                depth_value = depths.get(qid, 0)
                loaded_questions += 1
                rows.append(
                    {
                        "model": model,
                        "source_file": str(path),
                        "clip_name": clip_name,
                        "question_id": qid,
                        "question_type": normalize_question_type(result.get("type")),
                        "depth": depth_value,
                        "depth_bin": depth_bin(depth_value, depth_cap),
                        "depth_label": str(depth_bin(depth_value, depth_cap)),
                        "raw_correct": raw_correct,
                        "final_correct": final_correct,
                        "dependency_blocked": blocked,
                        "error_category": (
                            "Correct"
                            if final_correct
                            else "Dependency blocked"
                            if blocked
                            else "Answer mismatch"
                        ),
                    }
                )

        count_rows.append(
            {
                "model": model,
                "eval_file_count": len(model_files),
                "loaded_eval_file_count": loaded_files,
                "skipped_eval_file_count": skipped_files,
                "question_count": loaded_questions,
            }
        )

    if not rows:
        raise SystemExit("No dependency-round QA records were loaded.")

    df = pd.DataFrame(rows)
    max_depth = int(df["depth"].max())
    df["depth_label"] = df["depth_bin"].apply(
        lambda value: depth_label(int(value), depth_cap, max_depth)
    )
    return df, pd.DataFrame(count_rows)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png")
    plt.close(fig)


def save_tables(df: pd.DataFrame, counts: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output_dir / "table_model_file_counts.csv", index=False, encoding="utf-8-sig")
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
    model["dependency_blocked_percent"] = model["dependency_blocked"] / model["total"] * 100
    model = model.set_index("model").reindex(MODEL_FOLDERS).reset_index()
    model.to_csv(output_dir / "table_model_accuracy.csv", index=False, encoding="utf-8-sig")


def plot_overall_depth(df: pd.DataFrame, output_dir: Path, depth_cap: int) -> None:
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
    grouped["blocked_rate"] = grouped["dependency_blocked"] / grouped["total"] * 100
    grouped["raw_ci"] = grouped.apply(lambda row: normal_ci(row["raw_correct"], row["total"]), axis=1)
    grouped["final_ci"] = grouped.apply(lambda row: normal_ci(row["final_correct"], row["total"]), axis=1)

    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
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
        linewidth=1.8,
        capsize=3,
        label="Raw answer accuracy",
    )
    ax.plot(
        x,
        grouped["blocked_rate"],
        marker="^",
        color="#6A3D9A",
        linewidth=1.6,
        label="Dependency-blocked rate",
    )
    for idx, row in grouped.iterrows():
        ax.text(
            x[idx],
            max(row["raw_accuracy"], row["final_accuracy"]) + 4,
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
    save_figure(fig, output_dir, "fig1_overall_depth_accuracy")


def plot_model_depth_lines(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        df.groupby(["model", "depth_bin"])
        .agg(total=("final_correct", "size"), correct=("final_correct", "sum"))
        .reset_index()
    )
    grouped["accuracy"] = grouped["correct"] / grouped["total"] * 100
    depth_bins = sorted(df["depth_bin"].unique())

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for model in MODEL_FOLDERS:
        sub = grouped[grouped["model"] == model].set_index("depth_bin").reindex(depth_bins)
        if sub["accuracy"].isna().all():
            continue
        ax.plot(
            depth_bins,
            sub["accuracy"],
            marker="o",
            markersize=3.8,
            linewidth=1.25,
            color=MODEL_COLORS.get(model, "#777777"),
            label=model,
            alpha=0.9,
        )
    ax.set_xlabel("Dependency depth")
    ax.set_ylabel("Dependency-aware QA accuracy (%)")
    ax.set_ylim(-3, 103)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.legend(ncol=2, loc="upper right", frameon=True, edgecolor="#CCCCCC")
    ax.set_title("Per-Model Accuracy Trajectory by Dependency Depth", fontweight="bold", pad=8)
    save_figure(fig, output_dir, "fig2_model_depth_trajectories")


def plot_model_depth_heatmap(df: pd.DataFrame, output_dir: Path, depth_cap: int) -> None:
    grouped = (
        df.groupby(["model", "depth_bin"])
        .agg(total=("final_correct", "size"), correct=("final_correct", "sum"))
        .reset_index()
    )
    grouped["accuracy"] = grouped["correct"] / grouped["total"] * 100
    depth_bins = sorted(df["depth_bin"].unique())
    pivot = grouped.pivot(index="model", columns="depth_bin", values="accuracy").reindex(MODEL_FOLDERS)
    counts = grouped.pivot(index="model", columns="depth_bin", values="total").reindex(MODEL_FOLDERS)

    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    im = ax.imshow(pivot[depth_bins].values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    for row_idx, model in enumerate(pivot.index):
        for col_idx, depth_value in enumerate(depth_bins):
            value = pivot.loc[model, depth_value]
            total = counts.loc[model, depth_value]
            if pd.isna(value):
                continue
            color = "white" if value >= 60 else "#1A1A1A"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.0f}\n({int(total)})",
                ha="center",
                va="center",
                fontsize=6.1,
                color=color,
            )
    ax.set_xticks(np.arange(len(depth_bins)))
    max_depth = int(df["depth"].max())
    ax.set_xticklabels([depth_label(item, depth_cap, max_depth) for item in depth_bins])
    ax.set_yticks(np.arange(len(MODEL_FOLDERS)))
    ax.set_yticklabels(MODEL_FOLDERS)
    ax.set_xlabel("Dependency depth")
    ax.set_title("Model Accuracy by Dependency Depth", fontweight="bold", pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Accuracy (%)")
    save_figure(fig, output_dir, "fig3_model_depth_heatmap")


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
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
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
    ax.set_title("Error Composition by Dependency Depth", fontweight="bold", pad=8)
    save_figure(fig, output_dir, "fig4_error_propagation_by_depth")


def plot_model_overall(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        df.groupby("model")
        .agg(
            total=("final_correct", "size"),
            raw_correct=("raw_correct", "sum"),
            final_correct=("final_correct", "sum"),
        )
        .reset_index()
    )
    grouped["raw_accuracy"] = grouped["raw_correct"] / grouped["total"] * 100
    grouped["final_accuracy"] = grouped["final_correct"] / grouped["total"] * 100
    grouped = grouped.set_index("model").reindex(MODEL_FOLDERS).reset_index()

    y = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(8.0, 6.3))
    ax.barh(y - 0.18, grouped["raw_accuracy"], height=0.34, color="#E69F00", label="Raw")
    ax.barh(y + 0.18, grouped["final_accuracy"], height=0.34, color="#0072B2", label="Dependency-aware")
    for idx, row in grouped.iterrows():
        ax.text(
            max(row["raw_accuracy"], row["final_accuracy"]) + 0.9,
            idx,
            f"n={int(row['total'])}",
            va="center",
            fontsize=7,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(grouped["model"])
    ax.invert_yaxis()
    ax.set_xlabel("QA accuracy (%)")
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    ax.legend(loc="lower right", frameon=True, edgecolor="#CCCCCC")
    ax.set_title("Overall QA Accuracy by Model", fontweight="bold", pad=8)
    save_figure(fig, output_dir, "fig5_model_overall_accuracy")


def plot_question_type(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        df.groupby("question_type")
        .agg(
            total=("final_correct", "size"),
            raw_correct=("raw_correct", "sum"),
            final_correct=("final_correct", "sum"),
        )
        .reset_index()
    )
    grouped["raw_accuracy"] = grouped["raw_correct"] / grouped["total"] * 100
    grouped["final_accuracy"] = grouped["final_correct"] / grouped["total"] * 100
    grouped = grouped.sort_values("final_accuracy", ascending=True)

    y = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.barh(y - 0.18, grouped["raw_accuracy"], height=0.34, color="#E69F00", label="Raw")
    ax.barh(y + 0.18, grouped["final_accuracy"], height=0.34, color="#0072B2", label="Dependency-aware")
    for idx, row in grouped.iterrows():
        y_idx = grouped.index.get_loc(idx)
        ax.text(
            max(row["raw_accuracy"], row["final_accuracy"]) + 1.0,
            y_idx,
            f"n={int(row['total'])}",
            va="center",
            fontsize=7.3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(grouped["question_type"])
    ax.set_xlabel("QA accuracy (%)")
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    ax.legend(loc="lower right", frameon=True, edgecolor="#CCCCCC")
    ax.set_title("QA Accuracy by Question Type", fontweight="bold", pad=8)
    save_figure(fig, output_dir, "fig6_question_type_accuracy")


def plot_video_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    per_video = (
        df.groupby(["model", "source_file"])
        .agg(total=("final_correct", "size"), correct=("final_correct", "sum"))
        .reset_index()
    )
    per_video["accuracy"] = per_video["correct"] / per_video["total"] * 100
    values = [
        per_video.loc[per_video["model"] == model, "accuracy"].to_numpy()
        for model in MODEL_FOLDERS
    ]

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    box = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#1A1A1A", "linewidth": 1.3},
        whiskerprops={"color": "#666666"},
        capprops={"color": "#666666"},
    )
    for patch, model in zip(box["boxes"], MODEL_FOLDERS):
        patch.set_facecolor(MODEL_COLORS.get(model, "#999999"))
        patch.set_alpha(0.62)
        patch.set_edgecolor("#555555")

    rng = np.random.default_rng(4)
    for idx, vals in enumerate(values, start=1):
        jitter = rng.normal(loc=idx, scale=0.045, size=len(vals))
        ax.scatter(jitter, vals, s=7, color="#333333", alpha=0.28, linewidths=0)

    ax.set_xticks(np.arange(1, len(MODEL_FOLDERS) + 1))
    ax.set_xticklabels(MODEL_FOLDERS, rotation=32, ha="right")
    ax.set_ylabel("Per-file QA accuracy (%)")
    ax.set_ylim(-2, 102)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.set_title("Per-File Accuracy Distribution by Model", fontweight="bold", pad=8)
    save_figure(fig, output_dir, "fig7_per_file_accuracy_distribution")


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    output_dir = args.output_dir.resolve()

    df, counts = load_question_rows(repo_dir, args.depth_cap)
    save_tables(df, counts, output_dir)

    plot_overall_depth(df, output_dir, args.depth_cap)
    plot_model_depth_lines(df, output_dir)
    plot_model_depth_heatmap(df, output_dir, args.depth_cap)
    plot_error_propagation(df, output_dir)
    plot_model_overall(df, output_dir)
    plot_question_type(df, output_dir)
    plot_video_distribution(df, output_dir)

    raw_accuracy = df["raw_correct"].mean() * 100
    final_accuracy = df["final_correct"].mean() * 100
    blocked_rate = df["dependency_blocked"].mean() * 100
    print(f"Loaded eval files: {counts['loaded_eval_file_count'].sum()}")
    print(f"Loaded question rows: {len(df)}")
    print(f"Overall raw accuracy: {raw_accuracy:.2f}%")
    print(f"Overall dependency-aware accuracy: {final_accuracy:.2f}%")
    print(f"Overall dependency-blocked rate: {blocked_rate:.2f}%")
    print(f"Figures written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
