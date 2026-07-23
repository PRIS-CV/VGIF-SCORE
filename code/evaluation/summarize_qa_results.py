from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_VIDEOS_DIR = REPO_DIR / "hunyuan15_videos" / "dataset_720p_t2v_223"
VIDEO_LEVEL_KEYS = (
    "accuracy",
    "raw_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize QA evaluation JSON files into overall and per-dimension accuracy."
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help="Directory containing *_qa_eval*.json files.",
    )
    parser.add_argument(
        "--suffix",
        default="_qa_eval_dependency_rounds.json",
        help="Evaluation JSON suffix to summarize.",
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


def make_bucket() -> dict[str, Any]:
    return {
        "video_count": 0,
        "question_count": 0,
        "correct_count": 0,
        "raw_correct_count": 0,
        "accuracy": 0.0,
        "raw_accuracy": 0.0,
    }


def finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    question_count = bucket["question_count"]
    bucket["accuracy"] = (
        bucket["correct_count"] / question_count if question_count else 0.0
    )
    bucket["raw_accuracy"] = (
        bucket["raw_correct_count"] / question_count if question_count else 0.0
    )
    bucket["accuracy_percent"] = round(bucket["accuracy"] * 100, 2)
    bucket["raw_accuracy_percent"] = round(bucket["raw_accuracy"] * 100, 2)
    return bucket


def sort_buckets(buckets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, bucket in sorted(
        buckets.items(),
        key=lambda item: (-item[1]["accuracy"], -item[1]["question_count"], item[0]),
    ):
        rows.append({"name": name, **finalize_bucket(bucket)})
    return rows


def main() -> int:
    args = parse_args()
    videos_dir = args.videos_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else videos_dir / "qa_eval_dependency_rounds_summary_by_dimension.json"
    )

    eval_files = sorted(videos_dir.glob(f"*{args.suffix}"))
    if not eval_files:
        raise SystemExit(f"No evaluation files found in {videos_dir} with suffix {args.suffix}")

    filtered_eval_files: list[Path] = []
    for eval_path in eval_files:
        video_stem = eval_path.name[: -len(args.suffix)]
        video_path = videos_dir / f"{video_stem}.mp4"
        if video_path.exists():
            filtered_eval_files.append(eval_path)

    eval_files = filtered_eval_files
    if not eval_files:
        raise SystemExit(
            f"No evaluation files matched real video files in {videos_dir} with suffix {args.suffix}"
        )

    overall = make_bucket()
    by_type: dict[str, dict[str, Any]] = defaultdict(make_bucket)
    by_macro_domain: dict[str, dict[str, Any]] = defaultdict(make_bucket)
    by_micro_domain: dict[str, dict[str, Any]] = defaultdict(make_bucket)
    by_video: list[dict[str, Any]] = []

    average_video_metrics = {key: [] for key in VIDEO_LEVEL_KEYS}

    for eval_path in eval_files:
        payload = load_json(eval_path)
        results = payload.get("results", [])
        macro_domain = payload.get("matched_macro_domain") or "UNKNOWN"
        micro_domain = payload.get("matched_micro_domain") or "UNKNOWN"

        overall["video_count"] += 1
        by_macro_domain[macro_domain]["video_count"] += 1
        by_micro_domain[micro_domain]["video_count"] += 1

        for key in VIDEO_LEVEL_KEYS:
            value = payload.get(key)
            if isinstance(value, (int, float)):
                average_video_metrics[key].append(float(value))

        by_video.append(
            {
                "video_name": Path(payload.get("video_file", eval_path.name)).name,
                "question_count": payload.get("question_count"),
                "correct_count": payload.get("correct_count"),
                "accuracy": payload.get("accuracy"),
                "accuracy_percent": payload.get("accuracy_percent"),
                "matched_macro_domain": macro_domain,
                "matched_micro_domain": micro_domain,
                "eval_file": str(eval_path),
            }
        )

        for row in results:
            question_type = row.get("type") or "UNKNOWN"
            is_raw_correct = bool(row.get("answer_match"))
            is_correct = bool(row.get("correct"))

            overall["question_count"] += 1
            overall["raw_correct_count"] += int(is_raw_correct)
            overall["correct_count"] += int(is_correct)

            for bucket in (
                by_type[question_type],
                by_macro_domain[macro_domain],
                by_micro_domain[micro_domain],
            ):
                bucket["question_count"] += 1
                bucket["raw_correct_count"] += int(is_raw_correct)
                bucket["correct_count"] += int(is_correct)

    average_accuracy = (
        sum(average_video_metrics["accuracy"]) / len(average_video_metrics["accuracy"])
        if average_video_metrics["accuracy"]
        else 0.0
    )
    average_raw_accuracy = (
        sum(average_video_metrics["raw_accuracy"]) / len(average_video_metrics["raw_accuracy"])
        if average_video_metrics["raw_accuracy"]
        else 0.0
    )

    overall_node_micro = finalize_bucket(overall)
    overall_sample_macro = {
        "accuracy": average_accuracy,
        "accuracy_percent": round(average_accuracy * 100, 2),
        "raw_accuracy": average_raw_accuracy,
        "raw_accuracy_percent": round(average_raw_accuracy * 100, 2),
    }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "videos_dir": str(videos_dir),
        "suffix": args.suffix,
        "evaluated_video_count": len(eval_files),
        "aggregation_rules": {
            "paper_objective_score": "Compute dependency-aware QA accuracy per sample, then macro-average over samples.",
            "paper_node_type_columns": "For each semantic node type, divide total correct nodes by total evaluated nodes of that type.",
        },
        "overall_node_micro": overall_node_micro,
        "overall_sample_macro": overall_sample_macro,
        "overall_micro": overall_node_micro,
        "overall_macro_video_average": overall_sample_macro,
        "by_question_type": sort_buckets(by_type),
        "by_macro_domain": sort_buckets(by_macro_domain),
        "by_micro_domain": sort_buckets(by_micro_domain),
        "by_video": sorted(by_video, key=lambda row: row.get("video_name", "")),
    }
    dump_json(output_path, payload)

    overall_micro = payload["overall_node_micro"]
    print(f"Summary written to: {output_path}")
    print(
        "Overall QA accuracy: "
        f"{overall_micro['correct_count']}/{overall_micro['question_count']} "
        f"= {overall_micro['accuracy_percent']:.2f}%"
    )
    print(
        "Average per-video accuracy: "
        f"{payload['overall_sample_macro']['accuracy_percent']:.2f}%"
    )
    print("Per-question-type accuracy:")
    for row in payload["by_question_type"]:
        print(
            f"  {row['name']}: {row['correct_count']}/{row['question_count']} "
            f"= {row['accuracy_percent']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
