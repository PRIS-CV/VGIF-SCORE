from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import evaluate_video_qa_accuracy as evalmod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate QA evaluation JSON files into item-level accuracy summaries."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Directory containing *_qa_eval*.json files.",
    )
    parser.add_argument(
        "--question-mode",
        choices=[
            evalmod.QUESTION_MODE_ALL_AT_ONCE,
            evalmod.QUESTION_MODE_DEPENDENCY_ROUNDS,
        ],
        default=evalmod.QUESTION_MODE_DEPENDENCY_ROUNDS,
        help="Which QA eval suffix to aggregate.",
    )
    parser.add_argument(
        "--expected-video-count",
        type=int,
        default=None,
        help="Optional expected number of evaluated videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path.",
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


def init_bucket() -> dict[str, int]:
    return {
        "question_count": 0,
        "raw_correct_count": 0,
        "correct_count": 0,
        "dependency_blocked_count": 0,
    }


def update_bucket(bucket: dict[str, int], row: dict[str, Any]) -> None:
    bucket["question_count"] += 1
    if row.get("answer_match") is True:
        bucket["raw_correct_count"] += 1
    if row.get("correct") is True:
        bucket["correct_count"] += 1
    if row.get("answer_match") is True and row.get("dependency_passed") is False:
        bucket["dependency_blocked_count"] += 1


def finalize_bucket(bucket: dict[str, int]) -> dict[str, Any]:
    question_count = bucket["question_count"]
    raw_correct_count = bucket["raw_correct_count"]
    correct_count = bucket["correct_count"]
    dependency_blocked_count = bucket["dependency_blocked_count"]
    raw_accuracy = raw_correct_count / question_count if question_count else 0.0
    accuracy = correct_count / question_count if question_count else 0.0
    dependency_blocked_rate = (
        dependency_blocked_count / question_count if question_count else 0.0
    )
    return {
        **bucket,
        "raw_accuracy": raw_accuracy,
        "raw_accuracy_percent": round(raw_accuracy * 100, 2),
        "accuracy": accuracy,
        "accuracy_percent": round(accuracy * 100, 2),
        "dependency_blocked_rate": dependency_blocked_rate,
        "dependency_blocked_rate_percent": round(dependency_blocked_rate * 100, 2),
    }


def sort_summary_map(summary_map: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return dict(
        sorted(
            summary_map.items(),
            key=lambda item: (
                -item[1]["accuracy"],
                -item[1]["question_count"],
                item[0],
            ),
        )
    )


def main() -> int:
    args = parse_args()
    eval_dir = args.eval_dir.resolve()
    suffix = evalmod.get_output_suffix(args.question_mode)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else eval_dir / f"qa_accuracy_summary_{args.question_mode}.json"
    )

    if not eval_dir.exists():
        raise SystemExit(f"Evaluation directory not found: {eval_dir}")

    eval_paths = sorted(
        path
        for path in eval_dir.rglob(f"*{suffix}")
        if path.is_file()
    )
    if not eval_paths:
        raise SystemExit(f"No evaluation files found under {eval_dir} with suffix {suffix}")

    overall = init_bucket()
    by_type: dict[str, dict[str, int]] = defaultdict(init_bucket)
    by_macro_domain: dict[str, dict[str, int]] = defaultdict(init_bucket)
    by_micro_domain: dict[str, dict[str, int]] = defaultdict(init_bucket)
    per_video: list[dict[str, Any]] = []

    for path in eval_paths:
        payload = load_json(path)
        results = payload.get("results", [])
        macro_domain = payload.get("matched_macro_domain") or "UNKNOWN"
        micro_domain = payload.get("matched_micro_domain") or "UNKNOWN"

        per_video.append(
            {
                "video_name": Path(payload.get("video_file", path.name)).name,
                "sample_index": payload.get("sample_index"),
                "question_count": payload.get("question_count", len(results)),
                "raw_correct_count": payload.get("raw_correct_count"),
                "correct_count": payload.get("correct_count"),
                "raw_accuracy": payload.get("raw_accuracy"),
                "raw_accuracy_percent": payload.get("raw_accuracy_percent"),
                "accuracy": payload.get("accuracy"),
                "accuracy_percent": payload.get("accuracy_percent"),
                "dependency_blocked_count": payload.get("dependency_blocked_count"),
                "matched_macro_domain": macro_domain,
                "matched_micro_domain": micro_domain,
                "entry_match_source": payload.get("entry_match_source"),
                "eval_file": str(path),
            }
        )

        for row in results:
            update_bucket(overall, row)
            update_bucket(by_type[row.get("type") or "UNKNOWN"], row)
            update_bucket(by_macro_domain[macro_domain], row)
            update_bucket(by_micro_domain[micro_domain], row)

    per_video.sort(
        key=lambda item: (
            item["sample_index"] if isinstance(item.get("sample_index"), int) else 10**9,
            item["video_name"],
        )
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_dir": str(eval_dir),
        "question_mode": args.question_mode,
        "suffix": suffix,
        "eval_file_count": len(eval_paths),
        "expected_video_count": args.expected_video_count,
        "missing_video_count": (
            max(args.expected_video_count - len(eval_paths), 0)
            if args.expected_video_count is not None
            else None
        ),
        "overall": finalize_bucket(overall),
        "by_type": sort_summary_map(
            {name: finalize_bucket(bucket) for name, bucket in by_type.items()}
        ),
        "by_macro_domain": sort_summary_map(
            {name: finalize_bucket(bucket) for name, bucket in by_macro_domain.items()}
        ),
        "by_micro_domain": sort_summary_map(
            {name: finalize_bucket(bucket) for name, bucket in by_micro_domain.items()}
        ),
        "per_video": per_video,
    }
    dump_json(output_path, payload)

    overall_summary = payload["overall"]
    print(f"Summary written to: {output_path}")
    print(f"Eval files: {len(eval_paths)}")
    print(
        "Overall final accuracy: "
        f"{overall_summary['correct_count']}/{overall_summary['question_count']} "
        f"= {overall_summary['accuracy_percent']}%"
    )
    print(
        "Overall raw accuracy: "
        f"{overall_summary['raw_correct_count']}/{overall_summary['question_count']} "
        f"= {overall_summary['raw_accuracy_percent']}%"
    )
    print("Accuracy by QA type:")
    for name, bucket in payload["by_type"].items():
        print(
            f"  {name}: {bucket['correct_count']}/{bucket['question_count']} "
            f"= {bucket['accuracy_percent']}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
