from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "results_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "score_normalization_audit.json"
RUBRIC_IDS = ("Cin", "Pur", "Mot", "Phy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the printed equation and reported-table normalizations.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def rubric_ratings(payload: dict[str, Any]) -> dict[str, float] | None:
    ratings: dict[str, float] = {}
    results = payload.get("results")
    if isinstance(results, list):
        for row in results:
            if isinstance(row, dict) and row.get("id") in RUBRIC_IDS and isinstance(row.get("score"), (int, float)):
                ratings[str(row["id"])] = float(row["score"])
    scores = payload.get("scores")
    if isinstance(scores, dict):
        for key in RUBRIC_IDS:
            score = (scores.get(key) or {}).get("score")
            if isinstance(score, (int, float)):
                ratings[key] = float(score)
    return ratings if set(ratings) == set(RUBRIC_IDS) else None


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest.resolve())
    by_model: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in manifest.get("rows", []):
        if not isinstance(row, dict) or not row.get("complete"):
            continue
        qa_path = REPO_ROOT / str(row["qa_file"])
        rubric_path = REPO_ROOT / str(row["rubric_file"])
        qa = load_json(qa_path)
        rubric = load_json(rubric_path)
        question_count = int(qa.get("question_count", 0))
        if question_count <= 0:
            continue
        objective = int(qa.get("correct_count", 0)) / question_count
        ratings = rubric_ratings(rubric)
        if ratings is None:
            continue
        rating = average(list(ratings.values()))
        equation_subjective = (rating - 1.0) / 4.0
        reported_subjective = rating / 5.0
        by_model[str(row["model"])].append(
            {
                "objective": objective,
                "rating": rating,
                "equation_subjective": equation_subjective,
                "reported_subjective": reported_subjective,
                "equation_vgif": 0.5 * objective + 0.5 * equation_subjective,
                "reported_vgif": 0.5 * objective + 0.5 * reported_subjective,
            }
        )

    models: dict[str, Any] = {}
    for model, rows in sorted(by_model.items()):
        models[model] = {
            "pair_count": len(rows),
            "objective_percent": round(average([row["objective"] for row in rows]) * 100, 2),
            "mean_rating_1_to_5": round(average([row["rating"] for row in rows]), 6),
            "equation_subjective_percent": round(average([row["equation_subjective"] for row in rows]) * 100, 2),
            "reported_subjective_percent": round(average([row["reported_subjective"] for row in rows]) * 100, 2),
            "equation_vgif_percent": round(average([row["equation_vgif"] for row in rows]) * 100, 2),
            "reported_vgif_percent": round(average([row["reported_vgif"] for row in rows]) * 100, 2),
        }
    output = {
        "finding": "The finalized camera-ready tables use rating / 5, while the printed equation specifies (rating - 1) / 4.",
        "canonical_reproduction_protocol": "reported_table",
        "printed_equation_formula": "mean((rating - 1) / 4)",
        "reported_table_formula": "mean(rating) / 5",
        "models": models,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(models, ensure_ascii=False, indent=2))
    print(f"Audit written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
