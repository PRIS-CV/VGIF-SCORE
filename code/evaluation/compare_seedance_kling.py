from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

DEFAULT_KLING_SUMMARY = (
    REPO_DIR
    / "kling_t2v"
    / "outputs"
    / "kling_v3_720p_5s"
    / "20260416_134733"
    / "videos"
    / "qa_eval_batch_summary_01_08_gemini3flash.json"
)
DEFAULT_SEEDANCE_VIDEOS_DIR = (
    REPO_DIR
    / "seedance2.0"
    / "outputs"
    / "seedance_2_0_720p_5s"
    / "20260417_233131"
    / "videos"
)
DEFAULT_SELECTED_PROMPTS = (
    REPO_DIR
    / "kling_t2v"
    / "outputs"
    / "kling_v3_720p_5s"
    / "20260416_134733"
    / "selected_prompts.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Seedance and Kling QA evaluation results side by side."
    )
    parser.add_argument(
        "--kling-summary",
        type=Path,
        default=DEFAULT_KLING_SUMMARY,
        help="Kling batch summary JSON path.",
    )
    parser.add_argument(
        "--seedance-videos-dir",
        type=Path,
        default=DEFAULT_SEEDANCE_VIDEOS_DIR,
        help="Directory containing Seedance *_qa_eval_gemini-3-flash-preview.json files.",
    )
    parser.add_argument(
        "--selected-prompts",
        type=Path,
        default=DEFAULT_SELECTED_PROMPTS,
        help="selected_prompts.json used for sample metadata.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
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
    fieldnames = [
        "sample_index",
        "macro_domain",
        "micro_domain",
        "kling_video_name",
        "seedance_video_name",
        "question_count",
        "kling_correct_count",
        "seedance_correct_count",
        "kling_accuracy_percent",
        "seedance_accuracy_percent",
        "accuracy_delta_percent",
        "kling_raw_accuracy_percent",
        "seedance_raw_accuracy_percent",
        "raw_accuracy_delta_percent",
        "winner",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def dump_raw_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_index",
        "macro_domain",
        "micro_domain",
        "question_count",
        "kling_raw_correct_count",
        "seedance_raw_correct_count",
        "kling_raw_accuracy_percent",
        "seedance_raw_accuracy_percent",
        "raw_accuracy_delta_percent",
        "raw_winner",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def extract_sample_index(name: str) -> int | None:
    match = re.match(r"^(\d{2})_", name)
    if not match:
        return None
    return int(match.group(1))


def build_prompt_index(selected_prompts_path: Path) -> dict[int, dict[str, Any]]:
    payload = load_json(selected_prompts_path)
    selected_prompts = payload.get("selected_prompts", [])
    return {
        item["sample_index"]: item
        for item in selected_prompts
        if isinstance(item, dict) and isinstance(item.get("sample_index"), int)
    }


def build_kling_index(kling_summary_path: Path) -> dict[int, dict[str, Any]]:
    payload = load_json(kling_summary_path)
    index: dict[int, dict[str, Any]] = {}
    for item in payload.get("results", []):
        name = item.get("video_name")
        if not isinstance(name, str):
            continue
        sample_index = extract_sample_index(name)
        if sample_index is None:
            continue
        index[sample_index] = item
    return index


def build_seedance_index(seedance_videos_dir: Path) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for path in sorted(seedance_videos_dir.glob("*_qa_eval_gemini-3-flash-preview.json")):
        payload = load_json(path)
        sample_index = payload.get("sample_index")
        if not isinstance(sample_index, int):
            sample_index = extract_sample_index(path.name)
        if sample_index is None:
            continue
        payload["_compare_file"] = str(path)
        index[sample_index] = payload
    return index


def collect_wrong_qids(results: Any, *, use_final_correct: bool) -> list[str]:
    wrong_qids: list[str] = []
    if not isinstance(results, list):
        return wrong_qids

    for item in results:
        if not isinstance(item, dict):
            continue
        qid = item.get("id")
        if not isinstance(qid, str):
            continue

        if use_final_correct:
            is_correct = item.get("correct")
        else:
            is_correct = item.get("answer_match")

        if is_correct is False:
            wrong_qids.append(qid)

    return wrong_qids


def round_or_none(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value), 2)


def build_row(
    sample_index: int,
    prompt_item: dict[str, Any] | None,
    kling_item: dict[str, Any] | None,
    seedance_item: dict[str, Any] | None,
) -> dict[str, Any]:
    kling_accuracy = kling_item.get("accuracy_percent") if kling_item else None
    seedance_accuracy = seedance_item.get("accuracy_percent") if seedance_item else None
    kling_raw_accuracy = kling_item.get("raw_accuracy_percent") if kling_item else None
    seedance_raw_accuracy = seedance_item.get("raw_accuracy_percent") if seedance_item else None

    accuracy_delta = None
    if isinstance(kling_accuracy, (int, float)) and isinstance(seedance_accuracy, (int, float)):
        accuracy_delta = round(float(seedance_accuracy) - float(kling_accuracy), 2)

    raw_accuracy_delta = None
    if isinstance(kling_raw_accuracy, (int, float)) and isinstance(seedance_raw_accuracy, (int, float)):
        raw_accuracy_delta = round(float(seedance_raw_accuracy) - float(kling_raw_accuracy), 2)

    winner = "tie"
    if isinstance(accuracy_delta, (int, float)):
        if accuracy_delta > 0:
            winner = "seedance"
        elif accuracy_delta < 0:
            winner = "kling"

    raw_winner = "tie"
    if isinstance(raw_accuracy_delta, (int, float)):
        if raw_accuracy_delta > 0:
            raw_winner = "seedance"
        elif raw_accuracy_delta < 0:
            raw_winner = "kling"

    kling_results = load_json(Path(kling_item["eval_file"])).get("results", []) if kling_item else []
    seedance_results = seedance_item.get("results", []) if seedance_item else []

    kling_wrong_qids = collect_wrong_qids(kling_results, use_final_correct=True)
    seedance_wrong_qids = collect_wrong_qids(seedance_results, use_final_correct=True)
    kling_raw_wrong_qids = collect_wrong_qids(kling_results, use_final_correct=False)
    seedance_raw_wrong_qids = collect_wrong_qids(seedance_results, use_final_correct=False)

    return {
        "sample_index": sample_index,
        "macro_domain": prompt_item.get("macro_domain") if prompt_item else None,
        "micro_domain": prompt_item.get("micro_domain") if prompt_item else None,
        "prompt": prompt_item.get("prompt") if prompt_item else None,
        "question_count": (
            seedance_item.get("question_count")
            if seedance_item is not None
            else kling_item.get("question_count") if kling_item is not None else None
        ),
        "kling_video_name": kling_item.get("video_name") if kling_item else None,
        "kling_eval_file": kling_item.get("eval_file") if kling_item else None,
        "kling_correct_count": kling_item.get("correct_count") if kling_item else None,
        "kling_raw_correct_count": kling_item.get("raw_correct_count") if kling_item else None,
        "kling_accuracy_percent": round_or_none(kling_accuracy),
        "kling_raw_accuracy_percent": round_or_none(kling_raw_accuracy),
        "kling_wrong_qids": kling_wrong_qids,
        "kling_raw_wrong_qids": kling_raw_wrong_qids,
        "seedance_video_name": Path(seedance_item.get("video_file")).name if seedance_item else None,
        "seedance_eval_file": seedance_item.get("_compare_file") if seedance_item else None,
        "seedance_correct_count": seedance_item.get("correct_count") if seedance_item else None,
        "seedance_raw_correct_count": seedance_item.get("raw_correct_count") if seedance_item else None,
        "seedance_accuracy_percent": round_or_none(seedance_accuracy),
        "seedance_raw_accuracy_percent": round_or_none(seedance_raw_accuracy),
        "seedance_wrong_qids": seedance_wrong_qids,
        "seedance_raw_wrong_qids": seedance_raw_wrong_qids,
        "accuracy_delta_percent": accuracy_delta,
        "raw_accuracy_delta_percent": raw_accuracy_delta,
        "winner": winner,
        "raw_winner": raw_winner,
    }


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    args = parse_args()

    kling_summary_path = args.kling_summary.resolve()
    seedance_videos_dir = args.seedance_videos_dir.resolve()
    selected_prompts_path = args.selected_prompts.resolve()

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else seedance_videos_dir / "seedance_vs_kling_gemini3flash_comparison.json"
    )
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else seedance_videos_dir / "seedance_vs_kling_gemini3flash_comparison.csv"
    )
    detailed_json = seedance_videos_dir / "seedance_vs_kling_gemini3flash_detailed_qid_comparison.json"
    raw_json = seedance_videos_dir / "seedance_vs_kling_gemini3flash_raw_comparison.json"
    raw_csv = seedance_videos_dir / "seedance_vs_kling_gemini3flash_raw_comparison.csv"

    kling_index = build_kling_index(kling_summary_path)
    seedance_index = build_seedance_index(seedance_videos_dir)
    prompt_index = build_prompt_index(selected_prompts_path)

    sample_indexes = sorted(set(prompt_index) | set(kling_index) | set(seedance_index))
    rows = [
        build_row(
            sample_index=sample_index,
            prompt_item=prompt_index.get(sample_index),
            kling_item=kling_index.get(sample_index),
            seedance_item=seedance_index.get(sample_index),
        )
        for sample_index in sample_indexes
    ]

    comparable_rows = [
        row
        for row in rows
        if isinstance(row.get("kling_accuracy_percent"), (int, float))
        and isinstance(row.get("seedance_accuracy_percent"), (int, float))
    ]
    kling_better = sum(1 for row in comparable_rows if row["winner"] == "kling")
    seedance_better = sum(1 for row in comparable_rows if row["winner"] == "seedance")
    ties = sum(1 for row in comparable_rows if row["winner"] == "tie")
    raw_kling_better = sum(1 for row in comparable_rows if row["raw_winner"] == "kling")
    raw_seedance_better = sum(1 for row in comparable_rows if row["raw_winner"] == "seedance")
    raw_ties = sum(1 for row in comparable_rows if row["raw_winner"] == "tie")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_basis": "gemini-3-flash-preview final QA accuracy",
        "kling_summary": str(kling_summary_path),
        "seedance_videos_dir": str(seedance_videos_dir),
        "selected_prompts": str(selected_prompts_path),
        "row_count": len(rows),
        "comparable_count": len(comparable_rows),
        "overall": {
            "kling_average_accuracy_percent": round(
                average([row["kling_accuracy_percent"] for row in comparable_rows]), 2
            ),
            "seedance_average_accuracy_percent": round(
                average([row["seedance_accuracy_percent"] for row in comparable_rows]), 2
            ),
            "average_accuracy_delta_percent": round(
                average([row["accuracy_delta_percent"] for row in comparable_rows]),
                2,
            ),
            "kling_better_count": kling_better,
            "seedance_better_count": seedance_better,
            "tie_count": ties,
        },
        "rows": rows,
    }

    detailed_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_basis": "Per-prompt final and raw wrong q_id comparison",
        "rows": [
            {
                "sample_index": row["sample_index"],
                "macro_domain": row["macro_domain"],
                "micro_domain": row["micro_domain"],
                "question_count": row["question_count"],
                "kling_accuracy_percent": row["kling_accuracy_percent"],
                "seedance_accuracy_percent": row["seedance_accuracy_percent"],
                "accuracy_delta_percent": row["accuracy_delta_percent"],
                "kling_wrong_qids": row["kling_wrong_qids"],
                "seedance_wrong_qids": row["seedance_wrong_qids"],
                "kling_raw_wrong_qids": row["kling_raw_wrong_qids"],
                "seedance_raw_wrong_qids": row["seedance_raw_wrong_qids"],
            }
            for row in rows
        ],
    }

    raw_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_basis": "gemini-3-flash-preview raw QA accuracy only",
        "kling_summary": str(kling_summary_path),
        "seedance_videos_dir": str(seedance_videos_dir),
        "row_count": len(rows),
        "comparable_count": len(comparable_rows),
        "overall": {
            "kling_average_raw_accuracy_percent": round(
                average([row["kling_raw_accuracy_percent"] for row in comparable_rows]),
                2,
            ),
            "seedance_average_raw_accuracy_percent": round(
                average([row["seedance_raw_accuracy_percent"] for row in comparable_rows]),
                2,
            ),
            "average_raw_accuracy_delta_percent": round(
                average([row["raw_accuracy_delta_percent"] for row in comparable_rows]),
                2,
            ),
            "kling_better_count": raw_kling_better,
            "seedance_better_count": raw_seedance_better,
            "tie_count": raw_ties,
        },
        "rows": [
            {
                "sample_index": row["sample_index"],
                "macro_domain": row["macro_domain"],
                "micro_domain": row["micro_domain"],
                "question_count": row["question_count"],
                "kling_raw_correct_count": row["kling_raw_correct_count"],
                "seedance_raw_correct_count": row["seedance_raw_correct_count"],
                "kling_raw_accuracy_percent": row["kling_raw_accuracy_percent"],
                "seedance_raw_accuracy_percent": row["seedance_raw_accuracy_percent"],
                "raw_accuracy_delta_percent": row["raw_accuracy_delta_percent"],
                "raw_winner": row["raw_winner"],
            }
            for row in rows
        ],
    }

    dump_json(output_json, payload)
    dump_csv(output_csv, rows)
    dump_json(detailed_json, detailed_payload)
    dump_json(raw_json, raw_payload)
    dump_raw_csv(raw_csv, raw_payload["rows"])

    print(f"Comparison JSON written to: {output_json}")
    print(f"Comparison CSV written to: {output_csv}")
    print(f"Detailed qid JSON written to: {detailed_json}")
    print(f"Raw-only JSON written to: {raw_json}")
    print(f"Raw-only CSV written to: {raw_csv}")
    print(
        "Overall average accuracy: "
        f"Kling {payload['overall']['kling_average_accuracy_percent']}% vs "
        f"Seedance {payload['overall']['seedance_average_accuracy_percent']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
