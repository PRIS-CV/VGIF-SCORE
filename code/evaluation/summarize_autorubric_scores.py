from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from scoring import (
    extract_rubric_ratings_from_payload,
    RUBRIC_DIMENSIONS,
    mean_rubric_rating,
    mean_rubric_score,
    mean_sample_score,
    normalize_rating,
)


SCORE_KEYS = list(RUBRIC_DIMENSIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate autorubric evaluation JSON files into score summaries."
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*_autorubric_eval.json")
    parser.add_argument("--expected-video-count", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    args = parse_args()
    eval_dir = args.eval_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else eval_dir / "autorubric_eval_score_summary.json"
    )

    eval_paths = sorted(path for path in eval_dir.rglob(args.pattern) if path.is_file())
    if not eval_paths:
        raise SystemExit(f"No autorubric evaluation files found under {eval_dir} matching {args.pattern}")

    dim_values: dict[str, list[float]] = defaultdict(list)
    per_video: list[dict[str, Any]] = []
    per_video_subjective_ratings: list[float] = []
    per_video_subjective_scores: list[float] = []
    incomplete_files: list[str] = []

    for path in eval_paths:
        payload = load_json(path)
        try:
            score_map = extract_rubric_ratings_from_payload(payload)
        except ValueError:
            score_map = {}
        row = {
            "video_name": Path(payload.get("video_file", path.name)).name,
            "sample_id": payload.get("sample_id"),
            "sample_index": payload.get("sample_index"),
            "score_average": payload.get("score_average"),
            "model": payload.get("model"),
            "model_version": payload.get("model_version"),
            "output_file": str(path),
        }
        for key in SCORE_KEYS:
            score = score_map.get(key)
            row[key] = score
            if isinstance(score, (int, float)):
                score_map[key] = float(score)
                dim_values[key].append(float(score))
        if set(score_map) == set(SCORE_KEYS):
            subjective_rating = mean_rubric_rating(score_map)
            subjective_score = mean_rubric_score(score_map)
            row["subjective_rating"] = round(subjective_rating, 6)
            row["subjective_score"] = round(subjective_score, 6)
            row["subjective_score_percent"] = round(subjective_score * 100.0, 2)
            per_video_subjective_ratings.append(subjective_rating)
            per_video_subjective_scores.append(subjective_score)
        else:
            row["subjective_rating"] = None
            row["subjective_score"] = None
            row["subjective_score_percent"] = None
            incomplete_files.append(str(path))
        per_video.append(row)

    per_video.sort(
        key=lambda item: (
            item["sample_index"] if isinstance(item.get("sample_index"), int) else 10**9,
            item["video_name"],
        )
    )

    summary_scores: dict[str, dict[str, Any]] = {}
    for key in SCORE_KEYS:
        values = dim_values[key]
        mean_value = average(values) if values else None
        normalized = normalize_rating(mean_value) if mean_value is not None else None
        summary_scores[key] = {
            "count": len(values),
            "average_score": round(mean_value, 4) if mean_value is not None else None,
            "normalized_score": round(normalized, 6) if normalized is not None else None,
            "normalized_percent": round(normalized * 100, 2) if normalized is not None else None,
            "min_score": min(values) if values else None,
            "max_score": max(values) if values else None,
        }

    subjective_rating = (
        mean_sample_score(per_video_subjective_ratings)
        if per_video_subjective_ratings
        else None
    )
    subjective_score = (
        mean_sample_score(per_video_subjective_scores)
        if per_video_subjective_scores
        else None
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_dir": str(eval_dir),
        "pattern": args.pattern,
        "eval_file_count": len(eval_paths),
        "valid_sample_count": len(per_video_subjective_scores),
        "incomplete_sample_count": len(incomplete_files),
        "incomplete_files": incomplete_files,
        "expected_video_count": args.expected_video_count,
        "missing_video_count": (
            max(args.expected_video_count - len(eval_paths), 0)
            if args.expected_video_count is not None
            else None
        ),
        "score_summary": summary_scores,
        "subjective_rating": round(subjective_rating, 4) if subjective_rating is not None else None,
        "subjective_score": round(subjective_score, 6) if subjective_score is not None else None,
        "subjective_score_percent": round(subjective_score * 100.0, 2) if subjective_score is not None else None,
        "aggregation_rule": "Compute the four-dimension subjective score per complete sample, then macro-average over samples.",
        "per_video": per_video,
    }
    dump_json(output_path, payload)

    print(f"Summary written to: {output_path}")
    for key in SCORE_KEYS:
        item = summary_scores[key]
        print(f"{key}: avg={item['average_score']}/5 ({item['normalized_percent']}%) count={item['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
