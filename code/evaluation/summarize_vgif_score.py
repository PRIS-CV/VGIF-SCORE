from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from scoring import (
    extract_rubric_ratings_from_payload,
    mean_rubric_score,
    mean_sample_score,
    objective_score as compute_objective_score,
    sample_vgif_score,
)


MACRO_DOMAIN_ABBREVIATIONS = {
    "Commercial & Product Showcase": "Prod.",
    "Narrative & Cinematic Storytelling": "Narr.",
    "Creative & Surreal Expression": "Surr.",
    "Dynamics & Physical Interaction": "Phys.",
    "Emotion & Atmosphere Expression": "Emot.",
    "Spatial Composition & Scene Orchestration": "Spat.",
    "Performance, Sports & Embodied Motion": "Perf.",
    "Travel, Nature & Living World": "Nat.",
}

MACRO_DOMAIN_ORDER = [
    "Prod.",
    "Narr.",
    "Surr.",
    "Phys.",
    "Emot.",
    "Spat.",
    "Perf.",
    "Nat.",
]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute VGIF-Score as a 50/50 weighted average of subject QA accuracy "
            "and object autorubric score percent."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, default=None)
    parser.add_argument("--autorubric-dir", type=Path, default=None)
    parser.add_argument("--qa-pattern", default="*_qa_eval.json")
    parser.add_argument("--autorubric-pattern", default="*_autorubric_eval.json")
    parser.add_argument("--qa-weight", type=float, default=0.5)
    parser.add_argument("--autorubric-weight", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "video_name",
        "task_id",
        "macro_domain_abbr",
        "matched_macro_domain",
        "matched_micro_domain",
        "qa_accuracy_percent",
        "autorubric_percent",
        "vgif_score",
        "qa_model",
        "autorubric_model",
        "qa_eval_file",
        "autorubric_eval_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_video_key(video_name: str) -> str:
    if video_name.endswith(".mp4"):
        return video_name[:-4]
    return Path(video_name).stem


def short_macro_name(name: str | None) -> str:
    if not name:
        return "UNKNOWN"
    english = name.split(" (", 1)[0].strip()
    return MACRO_DOMAIN_ABBREVIATIONS.get(english, english)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def extract_autorubric_percent(payload: dict[str, Any]) -> float:
    ratings = extract_rubric_ratings_from_payload(payload)
    return mean_rubric_score(ratings) * 100.0


def build_qa_map(eval_dir: Path, pattern: str) -> dict[str, dict[str, Any]]:
    qa_map: dict[str, dict[str, Any]] = {}
    for path in sorted(eval_dir.rglob(pattern)):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        video_name = Path(payload.get("video_file", path.name)).name
        key = normalize_video_key(video_name)
        correct_count = payload.get("correct_count")
        question_count = payload.get("question_count")
        if isinstance(correct_count, int) and isinstance(question_count, int) and question_count > 0:
            objective_score = compute_objective_score(correct_count, question_count)
        elif isinstance(payload.get("accuracy"), (int, float)):
            objective_score = float(payload["accuracy"])
        else:
            objective_score = float(payload.get("accuracy_percent", 0.0)) / 100.0
        qa_map[key] = {
            "video_name": video_name,
            "sample_index": payload.get("sample_index"),
            "task_id": payload.get("task_id"),
            "objective_score": objective_score,
            "qa_accuracy_percent": objective_score * 100.0,
            "correct_count": correct_count,
            "question_count": question_count,
            "matched_macro_domain": payload.get("matched_macro_domain"),
            "matched_micro_domain": payload.get("matched_micro_domain"),
            "qa_model": payload.get("model"),
            "qa_eval_file": str(path),
        }
    return qa_map


def build_autorubric_map(eval_dir: Path, pattern: str) -> dict[str, dict[str, Any]]:
    autorubric_map: dict[str, dict[str, Any]] = {}
    for path in sorted(eval_dir.rglob(pattern)):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        video_name = Path(payload.get("video_file", path.name)).name
        key = normalize_video_key(video_name)
        autorubric_map[key] = {
            "video_name": video_name,
            "sample_index": payload.get("sample_index"),
            "autorubric_percent": extract_autorubric_percent(payload),
            "matched_macro_domain": payload.get("matched_macro_domain"),
            "matched_micro_domain": payload.get("matched_micro_domain"),
            "autorubric_model": payload.get("model"),
            "autorubric_model_version": payload.get("model_version"),
            "autorubric_eval_file": str(path),
        }
    return autorubric_map


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    videos_dir = run_dir / "videos"
    default_eval_dir = videos_dir if videos_dir.is_dir() else run_dir
    qa_dir = args.qa_dir.resolve() if args.qa_dir is not None else default_eval_dir
    autorubric_dir = (
        args.autorubric_dir.resolve() if args.autorubric_dir is not None else default_eval_dir
    )

    weight_sum = args.qa_weight + args.autorubric_weight
    if args.qa_weight < 0 or args.autorubric_weight < 0:
        raise SystemExit("qa-weight and autorubric-weight must be non-negative.")
    if weight_sum <= 0:
        raise SystemExit("The sum of qa-weight and autorubric-weight must be > 0.")
    qa_weight = args.qa_weight / weight_sum
    autorubric_weight = args.autorubric_weight / weight_sum

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else run_dir / "vgif_score_summary.json"
    )
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else run_dir / "vgif_score_per_video.csv"
    )

    qa_map = build_qa_map(qa_dir, args.qa_pattern)
    autorubric_map = build_autorubric_map(autorubric_dir, args.autorubric_pattern)

    qa_keys = set(qa_map)
    autorubric_keys = set(autorubric_map)
    common_keys = sorted(qa_keys & autorubric_keys)
    missing_qa = sorted(autorubric_keys - qa_keys)
    missing_autorubric = sorted(qa_keys - autorubric_keys)

    if not common_keys:
        raise SystemExit("No matched QA and autorubric evaluation pairs were found.")

    per_video: list[dict[str, Any]] = []
    by_macro_abbr: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"objective": [], "subjective": [], "vgif": []}
    )
    objective_scores: list[float] = []
    subjective_scores: list[float] = []
    vgif_scores: list[float] = []

    for key in common_keys:
        qa_row = qa_map[key]
        autorubric_row = autorubric_map[key]
        macro_domain = qa_row.get("matched_macro_domain") or autorubric_row.get("matched_macro_domain")
        micro_domain = qa_row.get("matched_micro_domain") or autorubric_row.get("matched_micro_domain")
        macro_abbr = short_macro_name(macro_domain)
        objective_score = float(qa_row["objective_score"])
        subjective_score = float(autorubric_row["autorubric_percent"]) / 100.0
        vgif_score = sample_vgif_score(
            objective_score,
            subjective_score,
            objective_weight=qa_weight,
            subjective_weight=autorubric_weight,
        )
        qa_accuracy_percent = objective_score * 100.0
        autorubric_percent = subjective_score * 100.0

        row = {
            "sample_index": qa_row.get("sample_index") or autorubric_row.get("sample_index"),
            "video_name": qa_row.get("video_name") or autorubric_row.get("video_name"),
            "task_id": qa_row.get("task_id"),
            "macro_domain_abbr": macro_abbr,
            "matched_macro_domain": macro_domain,
            "matched_micro_domain": micro_domain,
            "qa_accuracy_percent": round(qa_accuracy_percent, 4),
            "autorubric_percent": round(autorubric_percent, 4),
            "vgif_score": round(vgif_score * 100.0, 4),
            "qa_model": qa_row.get("qa_model"),
            "autorubric_model": autorubric_row.get("autorubric_model"),
            "qa_eval_file": qa_row.get("qa_eval_file"),
            "autorubric_eval_file": autorubric_row.get("autorubric_eval_file"),
        }
        per_video.append(row)
        by_macro_abbr[macro_abbr]["objective"].append(objective_score)
        by_macro_abbr[macro_abbr]["subjective"].append(subjective_score)
        by_macro_abbr[macro_abbr]["vgif"].append(vgif_score)
        objective_scores.append(objective_score)
        subjective_scores.append(subjective_score)
        vgif_scores.append(vgif_score)

    per_video.sort(
        key=lambda item: (
            item["sample_index"] if isinstance(item.get("sample_index"), int) else 10**9,
            item["video_name"],
        )
    )

    macro_summary = []
    for abbr in MACRO_DOMAIN_ORDER:
        values = by_macro_abbr.get(abbr, {"objective": [], "subjective": [], "vgif": []})
        video_count = len(values["vgif"])
        macro_summary.append(
            {
                "abbr": abbr,
                "video_count": video_count,
                "objective_score_percent": round(average(values["objective"]) * 100.0, 2),
                "subjective_score_percent": round(average(values["subjective"]) * 100.0, 2),
                "vgif_score": round(average(values["vgif"]), 6),
                "vgif_score_percent": round(average(values["vgif"]) * 100.0, 2),
            }
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "qa_dir": str(qa_dir),
        "autorubric_dir": str(autorubric_dir),
        "formula": {
            "qa_component": "objective ST-DAG QA accuracy_percent",
            "autorubric_component": "subjective mean(rating) / 5 over Cin, Pur, Mot, Phy",
            "qa_weight": qa_weight,
            "autorubric_weight": autorubric_weight,
            "vgif_score_formula": "VGIF-Score = qa_accuracy_percent * qa_weight + autorubric_percent * autorubric_weight",
            "benchmark_aggregation": "Compute Obj, Sub, and VGIF for each matched prompt-video pair, then macro-average each per-sample score.",
        },
        "video_count": len(per_video),
        "missing_qa_count": len(missing_qa),
        "missing_autorubric_count": len(missing_autorubric),
        "missing_qa": missing_qa,
        "missing_autorubric": missing_autorubric,
        "overall": {
            "objective_score": round(mean_sample_score(objective_scores), 6),
            "objective_score_percent": round(mean_sample_score(objective_scores) * 100.0, 2),
            "subjective_score": round(mean_sample_score(subjective_scores), 6),
            "subjective_score_percent": round(mean_sample_score(subjective_scores) * 100.0, 2),
            "vgif_score": round(mean_sample_score(vgif_scores), 6),
            "vgif_score_percent": round(mean_sample_score(vgif_scores) * 100.0, 2),
        },
        "by_macro_domain": macro_summary,
        "per_video": per_video,
    }

    dump_json(output_json, summary)
    dump_csv(output_csv, per_video)

    print(f"VGIF summary written to: {output_json}")
    print(f"Per-video CSV written to: {output_csv}")
    print(f"Video count: {len(per_video)}")
    print(f"Overall VGIF-Score: {summary['overall']['vgif_score_percent']}%")
    for item in macro_summary:
        print(f"{item['abbr']}: {item['vgif_score_percent']}% ({item['video_count']} videos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
