from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
FINAL_QA_ROOT = REPO_ROOT / "data" / "final_qa"
DEFAULT_JSON = REPO_ROOT / "data" / "results_manifest.json"
DEFAULT_CSV = REPO_ROOT / "data" / "results_manifest.csv"
EXPECTED_SAMPLES = 223
RUBRIC_IDS = {"Cin", "Pur", "Mot", "Phy"}


@dataclass(frozen=True)
class Candidate:
    path: Path
    payload: dict[str, Any]
    rank: tuple[int, int, int, int, int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical VGIF evaluation result manifest.")
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    parser.add_argument("--final-qa-root", type=Path, default=FINAL_QA_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--expected-samples", type=int, default=EXPECTED_SAMPLES)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def infer_sample_id(path: Path, payload: dict[str, Any]) -> int | None:
    name = path.name
    patterns = (
        r"^(\d{1,4})[_-]",
        r"(?:Wan2\.2|CogVideoX|MAGI(?:-1)?|LTX-2|Mochi-1|URSA|InfinityStar)[_-](\d{3,4})",
    )
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            value = int(match.group(1))
            return value if value > 0 else value + 1
    sample_index = payload.get("sample_index")
    if isinstance(sample_index, int):
        return sample_index + 1 if sample_index == 0 else sample_index
    return None


def evaluator_name(payload: dict[str, Any]) -> str:
    values = (
        payload.get("model"),
        payload.get("model_version"),
        (payload.get("model_versions") or [None])[-1],
    )
    return next((str(value) for value in values if value), "")


def is_valid_qa(payload: dict[str, Any]) -> bool:
    results = payload.get("results")
    count = payload.get("question_count")
    return (
        payload.get("success", True) is not False
        and isinstance(results, list)
        and len(results) > 0
        and isinstance(count, int)
        and count == len(results)
        and all(isinstance(row, dict) and isinstance(row.get("correct"), bool) for row in results)
    )


def rubric_score_ids(payload: dict[str, Any]) -> set[str]:
    results = payload.get("results")
    if isinstance(results, list):
        return {
            str(row.get("id"))
            for row in results
            if isinstance(row, dict) and isinstance(row.get("score"), (int, float))
        }
    scores = payload.get("scores")
    if isinstance(scores, dict):
        return {
            str(key)
            for key, value in scores.items()
            if isinstance(value, dict) and isinstance(value.get("score"), (int, float))
        }
    return set()


def is_valid_rubric(payload: dict[str, Any]) -> bool:
    return payload.get("success", True) is not False and RUBRIC_IDS <= rubric_score_ids(payload)


def candidate_rank(path: Path, payload: dict[str, Any], valid: bool, authoritative: bool) -> tuple[int, int, int, int, int, str]:
    evaluator = evaluator_name(payload).lower()
    return (
        int(valid),
        int(authoritative),
        int("3.1" in evaluator or "31pro" in path.name.lower()),
        int(payload.get("question_mode") == "dependency-rounds"),
        int("test" not in path.name.lower()),
        path.as_posix(),
    )


def collect_candidates(
    roots: list[tuple[Path, bool]],
    pattern: str,
    validator: Callable[[dict[str, Any]], bool],
) -> dict[int, list[Candidate]]:
    by_sample: dict[int, list[Candidate]] = {}
    seen: set[Path] = set()
    for root, authoritative in roots:
        if not root.exists():
            continue
        for path in root.rglob(pattern):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            payload = load_json(path)
            if payload is None:
                continue
            sample_id = infer_sample_id(path, payload)
            if sample_id is None:
                continue
            valid = validator(payload)
            by_sample.setdefault(sample_id, []).append(
                Candidate(path=resolved, payload=payload, rank=candidate_rank(path, payload, valid, authoritative))
            )
    return by_sample


def select(candidates: list[Candidate] | None) -> Candidate | None:
    if not candidates:
        return None
    chosen = max(candidates, key=lambda item: item.rank)
    return chosen if chosen.rank[0] else None


def relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_model_rows(model: str, model_root: Path, final_qa_root: Path, expected: int) -> list[dict[str, Any]]:
    qa_roots = [(final_qa_root / model, True), (model_root, False)]
    rubric_roots = [(model_root, False)]
    qa_candidates = collect_candidates(qa_roots, "*_qa_eval_dependency_rounds.json", is_valid_qa)
    rubric_candidates = collect_candidates(rubric_roots, "*_autorubric_eval.json", is_valid_rubric)
    rows: list[dict[str, Any]] = []
    for sample_id in range(1, expected + 1):
        qa = select(qa_candidates.get(sample_id))
        rubric = select(rubric_candidates.get(sample_id))
        rows.append(
            {
                "model": model,
                "sample_id": sample_id,
                "qa_status": "available" if qa else "missing",
                "qa_file": relative(qa.path if qa else None),
                "qa_evaluator": evaluator_name(qa.payload) if qa else None,
                "qa_candidate_count": len(qa_candidates.get(sample_id, [])),
                "rubric_status": "available" if rubric else "missing",
                "rubric_file": relative(rubric.path if rubric else None),
                "rubric_evaluator": evaluator_name(rubric.payload) if rubric else None,
                "rubric_candidate_count": len(rubric_candidates.get(sample_id, [])),
                "complete": bool(qa and rubric),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    models_root = args.models_root.resolve()
    final_qa_root = args.final_qa_root.resolve()
    models = sorted(path.name for path in models_root.iterdir() if path.is_dir())
    rows: list[dict[str, Any]] = []
    for model in models:
        rows.extend(build_model_rows(model, models_root / model, final_qa_root, args.expected_samples))

    by_model: dict[str, dict[str, int]] = {}
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        by_model[model] = {
            "expected": args.expected_samples,
            "qa_available": sum(row["qa_status"] == "available" for row in model_rows),
            "rubric_available": sum(row["rubric_status"] == "available" for row in model_rows),
            "complete_pairs": sum(bool(row["complete"]) for row in model_rows),
        }
    payload = {
        "selection_policy": [
            "valid structured result",
            "data/final_qa candidate for QA",
            "Gemini 3.1 evaluator",
            "dependency-round question mode",
            "non-test filename",
        ],
        "expected_samples_per_model": args.expected_samples,
        "model_count": len(models),
        "by_model": by_model,
        "rows": rows,
    }
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv.resolve(), rows)
    print(json.dumps(by_model, ensure_ascii=False, indent=2))
    print(f"Manifest JSON: {output_json}")
    print(f"Manifest CSV: {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
