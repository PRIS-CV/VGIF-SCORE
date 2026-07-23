from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import evaluate_video_qa_accuracy as evalmod


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_VIDEOS_DIR = REPO_DIR / "hunyuan15_videos" / "dataset_720p_t2v_223"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate videos with Gemini QA accuracy concurrently."
    )
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--end-at", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--entries", type=Path, default=None)
    parser.add_argument("--metadata-root", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--format-retries", type=int, default=evalmod.DEFAULT_FORMAT_RETRIES)
    parser.add_argument("--eval-attempts", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--question-mode",
        choices=[
            evalmod.QUESTION_MODE_ALL_AT_ONCE,
            evalmod.QUESTION_MODE_DEPENDENCY_ROUNDS,
            evalmod.QUESTION_MODE_AUTORUBRIC,
        ],
        default=evalmod.QUESTION_MODE_DEPENDENCY_ROUNDS,
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_name",
        "video_path",
        "eval_file",
        "question_mode",
        "round_count",
        "question_count",
        "raw_correct_count",
        "raw_accuracy",
        "raw_accuracy_percent",
        "correct_count",
        "accuracy",
        "accuracy_percent",
        "dependency_blocked_count",
        "response_id",
        "model",
        "model_version",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def iter_target_videos(videos_dir: Path, start_from: str | None, end_at: str | None) -> list[Path]:
    targets: list[Path] = []
    for path in sorted(videos_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if start_from and path.name < start_from:
            continue
        if end_at and path.name > end_at:
            continue
        targets.append(path)
    return targets


def build_eval_json_path(video_path: Path, question_mode: str) -> Path:
    return video_path.with_name(f"{video_path.stem}{evalmod.get_output_suffix(question_mode)}")


def build_eval_args(args: argparse.Namespace, video_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        api_key=args.api_key if args.api_key is not None else evalmod.DEFAULT_API_KEY,
        base_url=args.base_url if args.base_url is not None else evalmod.DEFAULT_BASE_URL,
        model=args.model if args.model is not None else evalmod.DEFAULT_MODEL,
        video=video_path,
        entries=args.entries if args.entries is not None else evalmod.DEFAULT_ENTRIES_PATH,
        metadata_root=args.metadata_root if args.metadata_root is not None else args.videos_dir,
        timeout=args.timeout if args.timeout is not None else evalmod.DEFAULT_TIMEOUT,
        retries=args.retries if args.retries is not None else evalmod.DEFAULT_RETRIES,
        format_retries=args.format_retries,
        question_mode=args.question_mode,
        skip_model_check=args.skip_model_check,
        output=None,
        output_tag=None,
        verbose=args.verbose,
    )


def extract_result_row(video_path: Path, eval_json_path: Path) -> dict[str, Any]:
    payload = load_json(eval_json_path)
    return {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "eval_file": str(eval_json_path),
        "question_mode": payload.get("question_mode"),
        "round_count": payload.get("round_count"),
        "question_count": payload.get("question_count"),
        "raw_correct_count": payload.get("raw_correct_count"),
        "raw_accuracy": payload.get("raw_accuracy"),
        "raw_accuracy_percent": payload.get("raw_accuracy_percent"),
        "correct_count": payload.get("correct_count"),
        "accuracy": payload.get("accuracy"),
        "accuracy_percent": payload.get("accuracy_percent"),
        "dependency_blocked_count": payload.get("dependency_blocked_count"),
        "response_id": payload.get("response_id"),
        "model": payload.get("model"),
        "model_version": payload.get("model_version"),
    }


def evaluate_one(args: argparse.Namespace, video_path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    eval_json_path = build_eval_json_path(video_path, args.question_mode)
    local_logs: list[dict[str, Any]] = []

    if args.overwrite or not eval_json_path.exists():
        eval_args = build_eval_args(args, video_path)
        attempts = max(1, args.eval_attempts)
        returncode = 1
        for attempt in range(1, attempts + 1):
            returncode = evalmod.run_evaluation(eval_args)
            local_logs.append(
                {
                    "video_name": video_path.name,
                    "attempt": attempt,
                    "returncode": returncode,
                }
            )
            if returncode == 0:
                break
            if attempt < attempts:
                print(f"Retrying {video_path.name} ({attempt}/{attempts})...", file=sys.stderr)
        if returncode != 0:
            print(f"Evaluation failed: {video_path.name}", file=sys.stderr)
            return None, local_logs

    if not eval_json_path.exists():
        print(f"Missing eval output: {video_path.name}", file=sys.stderr)
        return None, local_logs

    return extract_result_row(video_path, eval_json_path), local_logs


def main() -> int:
    args = parse_args()
    args.videos_dir = args.videos_dir.resolve()
    if args.entries is None:
        args.entries = evalmod.DEFAULT_ENTRIES_PATH
    else:
        args.entries = args.entries.resolve()
    if args.metadata_root is None:
        args.metadata_root = args.videos_dir
    else:
        args.metadata_root = args.metadata_root.resolve()

    if args.max_workers <= 0:
        print("--max-workers must be > 0", file=sys.stderr)
        return 1

    default_summary_name = (
        "qa_eval_batch_summary_dependency_rounds.json"
        if args.question_mode == evalmod.QUESTION_MODE_DEPENDENCY_ROUNDS
        else "qa_eval_batch_summary.json"
    )
    summary_output = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else args.videos_dir / default_summary_name
    )
    csv_output = (
        args.csv_output.resolve()
        if args.csv_output is not None
        else summary_output.with_suffix(".csv")
    )

    if not args.videos_dir.exists():
        print(f"Videos directory not found: {args.videos_dir}", file=sys.stderr)
        return 1

    videos = iter_target_videos(args.videos_dir, args.start_from, args.end_at)
    if args.max_items is not None:
        videos = videos[: args.max_items]
    if not videos:
        print("No videos to evaluate.", file=sys.stderr)
        return 1

    summary_rows: list[dict[str, Any]] = []
    run_logs: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(evaluate_one, args, video_path): video_path
            for video_path in videos
        }
        for future in as_completed(futures):
            summary_row, local_logs = future.result()
            run_logs.extend(local_logs)
            if summary_row is not None:
                summary_rows.append(summary_row)

    summary_rows.sort(key=lambda row: row["video_name"])
    average_accuracy = (
        sum(row["accuracy"] for row in summary_rows if isinstance(row.get("accuracy"), (int, float)))
        / len(summary_rows)
        if summary_rows
        else 0.0
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "videos_dir": str(args.videos_dir),
        "entries": str(args.entries),
        "metadata_root": str(args.metadata_root),
        "question_mode": args.question_mode,
        "video_count": len(videos),
        "successful_count": len(summary_rows),
        "average_accuracy": average_accuracy,
        "average_accuracy_percent": round(average_accuracy * 100, 2),
        "max_workers": args.max_workers,
        "results": summary_rows,
        "run_logs": run_logs,
    }
    dump_json(summary_output, payload)
    dump_csv(csv_output, summary_rows)

    print(f"Batch summary written to: {summary_output}")
    print(f"CSV summary written to: {csv_output}")
    print(f"Succeeded: {len(summary_rows)}/{len(videos)}")
    print(f"Average per-video accuracy: {average_accuracy:.2%}")
    return 0 if summary_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
