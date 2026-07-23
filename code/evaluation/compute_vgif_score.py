from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from scoring import (
    RUBRIC_DIMENSIONS,
    extract_rubric_ratings_from_payload,
    mean_sample_score,
    normalize_rating,
    objective_score as compute_objective_score,
    sample_vgif_score,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_DIR / "models" / "CogVideoX-1.5" / "results"

MACRO_ABBREVIATIONS = {
    "Commercial & Product Showcase": "Prod.",
    "Narrative & Cinematic Storytelling": "Narr.",
    "Creative & Surreal Expression": "Surr.",
    "Dynamics & Physical Interaction": "Phys.",
    "Emotion & Atmosphere Expression": "Emot.",
    "Spatial Composition & Scene Orchestration": "Spat.",
    "Performance, Sports & Embodied Motion": "Perf.",
    "Travel, Nature & Living World": "Nat.",
}
MACRO_ORDER = ["Prod.", "Narr.", "Surr.", "Phys.", "Emot.", "Spat.", "Perf.", "Nat."]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine QA accuracy and autorubric scores into per-video VGIF-Score."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--qa-suffix", default="_qa_eval_dependency_rounds.json")
    parser.add_argument("--autorubric-tag", default="gem31fullrub")
    parser.add_argument("--objective-weight", type=float, default=0.5)
    parser.add_argument("--subjective-weight", type=float, default=0.5)
    parser.add_argument("--cin-weight", type=float, default=1.0)
    parser.add_argument("--pur-weight", type=float, default=1.0)
    parser.add_argument("--mot-weight", type=float, default=1.0)
    parser.add_argument("--phy-weight", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_video_id(name: str) -> int:
    match = re.search(r"CogVideoX-(\d{3,4})", name, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"MAGI-\d+-(\d{3,4})", name, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"(\d{3,4})(?!.*\d)", name)
    if match is None:
        raise ValueError(f"Could not infer video id from: {name}")
    return int(match.group(1))


def normalize_macro_name(value: str | None) -> str:
    if not value:
        return "UNKNOWN"
    return value.split("(", 1)[0].strip()


def to_percent(value: float) -> float:
    return round(value * 100, 2)


def round6(value: float) -> float:
    return round(value, 6)


def collect_payloads(run_dir: Path, qa_suffix: str, autorubric_tag: str) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    qa_payloads: dict[int, dict[str, Any]] = {}
    autorubric_payloads: dict[int, dict[str, Any]] = {}

    for path in sorted(run_dir.glob(f"*{qa_suffix}")):
        qa_payloads[extract_video_id(path.stem)] = load_json(path)

    autorubric_suffix = f"_{autorubric_tag}_autorubric_eval.json"
    for path in sorted(run_dir.glob(f"*{autorubric_suffix}")):
        autorubric_payloads[extract_video_id(path.stem)] = load_json(path)

    return qa_payloads, autorubric_payloads


def weighted_subjective_scores(
    score_map: dict[str, float],
    *,
    cin_weight: float,
    pur_weight: float,
    mot_weight: float,
    phy_weight: float,
) -> tuple[float, float]:
    weights = {
        "Cin": cin_weight,
        "Pur": pur_weight,
        "Mot": mot_weight,
        "Phy": phy_weight,
    }
    if any(value < 0 for value in weights.values()):
        raise ValueError("Rubric dimension weights must be non-negative.")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Rubric dimension weights must sum to a positive value.")

    missing = [key for key in RUBRIC_DIMENSIONS if key not in score_map]
    if missing:
        raise ValueError(f"Missing autorubric dimensions: {', '.join(missing)}")

    for key in RUBRIC_DIMENSIONS:
        normalize_rating(score_map[key])
    weighted_rating = sum(score_map[key] * weights[key] for key in RUBRIC_DIMENSIONS) / total_weight
    weighted_score = sum(
        normalize_rating(score_map[key]) * weights[key]
        for key in RUBRIC_DIMENSIONS
    ) / total_weight
    return weighted_rating, weighted_score


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    qa_payloads, autorubric_payloads = collect_payloads(run_dir, args.qa_suffix, args.autorubric_tag)
    video_ids = sorted(set(qa_payloads) & set(autorubric_payloads))

    missing_qa = sorted(set(autorubric_payloads) - set(qa_payloads))
    missing_autorubric = sorted(set(qa_payloads) - set(autorubric_payloads))
    if missing_qa or missing_autorubric:
        raise ValueError(
            "QA / autorubric file sets do not match. "
            f"missing_qa={missing_qa[:10]} missing_autorubric={missing_autorubric[:10]}"
        )

    total_mix_weight = args.objective_weight + args.subjective_weight
    if args.objective_weight < 0 or args.subjective_weight < 0:
        raise ValueError("objective_weight and subjective_weight must be non-negative.")
    if total_mix_weight <= 0:
        raise ValueError("objective_weight + subjective_weight must be positive.")

    objective_mix_weight = args.objective_weight / total_mix_weight
    subjective_mix_weight = args.subjective_weight / total_mix_weight

    per_video_rows: list[dict[str, Any]] = []
    macro_buckets: dict[str, list[float]] = {abbr: [] for abbr in MACRO_ORDER}
    objective_values: list[float] = []
    subjective_rating_values: list[float] = []
    subjective_values: list[float] = []
    vgif_values: list[float] = []

    for video_id in video_ids:
        qa_payload = qa_payloads[video_id]
        autorubric_payload = autorubric_payloads[video_id]

        correct_count = int(qa_payload.get("correct_count", 0))
        question_count = int(qa_payload.get("question_count", 0))
        objective_score = compute_objective_score(correct_count, question_count)

        score_map = extract_rubric_ratings_from_payload(autorubric_payload)
        subjective_rating, subjective_score = weighted_subjective_scores(
            score_map,
            cin_weight=args.cin_weight,
            pur_weight=args.pur_weight,
            mot_weight=args.mot_weight,
            phy_weight=args.phy_weight,
        )
        vgif_score = sample_vgif_score(
            objective_score,
            subjective_score,
            objective_weight=objective_mix_weight,
            subjective_weight=subjective_mix_weight,
        )
        objective_values.append(objective_score)
        subjective_rating_values.append(subjective_rating)
        subjective_values.append(subjective_score)
        vgif_values.append(vgif_score)

        macro_name = normalize_macro_name(
            qa_payload.get("matched_macro_domain") or autorubric_payload.get("matched_macro_domain")
        )
        macro_abbr = MACRO_ABBREVIATIONS.get(macro_name, "UNKNOWN")
        if macro_abbr != "UNKNOWN":
            macro_buckets.setdefault(macro_abbr, []).append(vgif_score)

        video_file = qa_payload.get("video_file") or autorubric_payload.get("video_file")
        video_name = Path(str(video_file)).name if video_file else f"CogVideoX-{video_id:03d}.mp4"

        per_video_rows.append(
            {
                "video_id": video_id,
                "video_name": video_name,
                "video_file": video_file,
                "macro_domain": macro_name,
                "macro_abbr": macro_abbr,
                "objective_score": round6(objective_score),
                "objective_score_percent": to_percent(objective_score),
                "objective_correct_count": correct_count,
                "objective_question_count": question_count,
                "subjective_rating": round6(subjective_rating),
                "subjective_score": round6(subjective_score),
                "subjective_score_percent": to_percent(subjective_score),
                "cin_score": score_map["Cin"],
                "pur_score": score_map["Pur"],
                "mot_score": score_map["Mot"],
                "phy_score": score_map["Phy"],
                "vgif_score": round6(vgif_score),
                "vgif_score_percent": to_percent(vgif_score),
            }
        )

    per_video_rows.sort(key=lambda row: row["video_id"])

    if not per_video_rows:
        raise ValueError("No matched QA and AutoRubric sample pairs were found.")
    overall_vgif = mean_sample_score(vgif_values)
    overall_objective = mean_sample_score(objective_values)
    overall_subjective_rating = mean_sample_score(subjective_rating_values)
    overall_subjective = mean_sample_score(subjective_values)

    by_macro = []
    for macro_abbr in MACRO_ORDER:
        values = macro_buckets.get(macro_abbr, [])
        average = sum(values) / len(values) if values else 0.0
        by_macro.append(
            {
                "macro_abbr": macro_abbr,
                "video_count": len(values),
                "vgif_score": round6(average),
                "vgif_score_percent": to_percent(average),
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "qa_suffix": args.qa_suffix,
        "autorubric_tag": args.autorubric_tag,
        "mixing_rule": {
            "objective_weight": args.objective_weight,
            "subjective_weight": args.subjective_weight,
            "normalized_objective_weight": round6(objective_mix_weight),
            "normalized_subjective_weight": round6(subjective_mix_weight),
        },
        "aggregation_rule": "Compute objective, subjective, and VGIF per sample, then macro-average over the same matched sample set.",
        "autorubric_subjective_rule": {
            "dimensions": list(RUBRIC_DIMENSIONS),
            "cin_weight": args.cin_weight,
            "pur_weight": args.pur_weight,
            "mot_weight": args.mot_weight,
            "phy_weight": args.phy_weight,
            "normalization": "The four 1-5 ratings are averaged and divided by the maximum rating of 5.",
        },
        "overall": {
            "video_count": len(per_video_rows),
            "objective_score": round6(overall_objective),
            "objective_score_percent": to_percent(overall_objective),
            "subjective_rating": round6(overall_subjective_rating),
            "subjective_score": round6(overall_subjective),
            "subjective_score_percent": to_percent(overall_subjective),
            "vgif_score": round6(overall_vgif),
            "vgif_score_percent": to_percent(overall_vgif),
        },
        "by_macro": by_macro,
        "per_video": per_video_rows,
    }

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else run_dir / f"vgif_score_{args.autorubric_tag}.json"
    )
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else run_dir / f"vgif_score_{args.autorubric_tag}.csv"
    )

    dump_json(output_json, summary)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "video_id",
                "video_name",
                "video_file",
                "macro_abbr",
                "macro_domain",
                "objective_score",
                "objective_score_percent",
                "objective_correct_count",
                "objective_question_count",
                "subjective_rating",
                "subjective_score",
                "subjective_score_percent",
                "cin_score",
                "pur_score",
                "mot_score",
                "phy_score",
                "vgif_score",
                "vgif_score_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(per_video_rows)

    print(f"VGIF summary written to {output_json}")
    print(f"VGIF csv written to {output_csv}")
    print(f"Overall VGIF-Score: {summary['overall']['vgif_score_percent']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
