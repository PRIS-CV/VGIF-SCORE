from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import evaluate_video_qa_accuracy as evalmod


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_VIDEOS_DIR = (
    REPO_DIR
    / "kling_t2v"
    / "outputs"
    / "kling_v3_720p_5s"
    / "20260416_134733"
    / "videos"
)
DEFAULT_EVAL_SCRIPT = PROJECT_DIR / "evaluate_video_qa_accuracy.py"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate videos with Gemini QA accuracy and write a summary JSON."
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help="Directory containing the videos to evaluate.",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=DEFAULT_EVAL_SCRIPT,
        help="Path to the single-video evaluation script.",
    )
    parser.add_argument(
        "--start-from",
        default=None,
        help="Only evaluate videos whose file name is >= this prefix lexicographically.",
    )
    parser.add_argument(
        "--end-at",
        default=None,
        help="Optional upper prefix bound; only evaluate videos whose name is <= this prefix.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Only process up to this many matching videos in the current run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if *_qa_eval.json already exists.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key passed through to the single-video script.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional base URL passed through to the single-video script.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model passed through to the single-video script.",
    )
    parser.add_argument(
        "--entries",
        type=Path,
        default=None,
        help="Optional all_entries_merged_final.json path passed through to the single-video script.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help="Optional metadata root passed through to the single-video script.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional request timeout passed through to the single-video script.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Optional retry count passed through to the single-video script.",
    )
    parser.add_argument(
        "--eval-attempts",
        type=int,
        default=3,
        help="How many times to retry a video when evaluation fails.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="How many videos to evaluate concurrently.",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Pass through to the single-video script.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass through to the single-video script.",
    )
    parser.add_argument(
        "--question-mode",
        choices=[
            evalmod.QUESTION_MODE_ALL_AT_ONCE,
            evalmod.QUESTION_MODE_DEPENDENCY_ROUNDS,
        ],
        default=evalmod.QUESTION_MODE_ALL_AT_ONCE,
        help="QA 提问方式：一次性提问或按 dependency 分轮提问。",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional summary output JSON path; defaults to videos_dir/qa_eval_batch_summary.json.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV summary path; defaults to the JSON summary path with a .csv suffix.",
    )
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


def iter_target_videos(videos_dir: Path, start_from: str, end_at: str | None) -> list[Path]:
    targets: list[Path] = []
    for path in sorted(videos_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        name = path.name
        if start_from and name < start_from:
            continue
        if end_at and name > end_at:
            continue
        targets.append(path)
    return targets


def build_eval_json_path(video_path: Path, question_mode: str) -> Path:
    suffix = evalmod.get_output_suffix(question_mode)
    return video_path.with_name(f"{video_path.stem}{suffix}")


def build_command(args: argparse.Namespace, eval_script: Path, video_path: Path) -> list[str]:
    command = [sys.executable, str(eval_script), "--video", str(video_path)]
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    if args.model:
        command.extend(["--model", args.model])
    if args.entries:
        command.extend(["--entries", str(args.entries)])
    if args.metadata_root:
        command.extend(["--metadata-root", str(args.metadata_root)])
    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])
    if args.retries is not None:
        command.extend(["--retries", str(args.retries)])
    if args.question_mode:
        command.extend(["--question-mode", args.question_mode])
    if args.eval_attempts is not None:
        command.extend(["--eval-attempts", str(args.eval_attempts)])
    if args.skip_model_check:
        command.append("--skip-model-check")
    if args.verbose:
        command.append("--verbose")
    return command


def build_eval_args(args: argparse.Namespace, video_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        api_key=args.api_key if args.api_key is not None else evalmod.DEFAULT_API_KEY,
        base_url=args.base_url if args.base_url is not None else evalmod.DEFAULT_BASE_URL,
        model=args.model if args.model is not None else evalmod.DEFAULT_MODEL,
        video=video_path,
        entries=args.entries if args.entries is not None else evalmod.DEFAULT_ENTRIES_PATH,
        metadata_root=(
            args.metadata_root if args.metadata_root is not None else evalmod.DEFAULT_METADATA_ROOT
        ),
        timeout=args.timeout if args.timeout is not None else evalmod.DEFAULT_TIMEOUT,
        retries=args.retries if args.retries is not None else evalmod.DEFAULT_RETRIES,
        format_retries=evalmod.DEFAULT_FORMAT_RETRIES,
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


def main() -> int:
    args = parse_args()
    videos_dir = args.videos_dir.resolve()
    eval_script = args.eval_script.resolve()
    default_summary_name = (
        "qa_eval_batch_summary_dependency_rounds.json"
        if args.question_mode == evalmod.QUESTION_MODE_DEPENDENCY_ROUNDS
        else "qa_eval_batch_summary.json"
    )
    summary_output = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else videos_dir / default_summary_name
    )
    csv_output = (
        args.csv_output.resolve()
        if args.csv_output is not None
        else summary_output.with_suffix(".csv")
    )

    if not videos_dir.exists():
        print(f"错误：找不到 videos 目录：{videos_dir}", file=sys.stderr)
        return 1
    if not eval_script.exists():
        print(f"错误：找不到评测脚本：{eval_script}", file=sys.stderr)
        return 1

    videos = iter_target_videos(videos_dir, args.start_from, args.end_at)
    if not videos:
        print("没有找到需要评测的视频。", file=sys.stderr)
        return 1

    if args.max_items is not None:
        if args.max_items <= 0:
            print("--max-items 必须是正整数。", file=sys.stderr)
            return 1
        videos = videos[: args.max_items]

    summary_rows: list[dict[str, Any]] = []
    run_logs: list[dict[str, Any]] = []
    succeeded = 0

    for video_path in videos:
        eval_json_path = build_eval_json_path(video_path, args.question_mode)
        should_run = args.overwrite or not eval_json_path.exists()

        if should_run:
            command = build_command(args, eval_script, video_path)
            eval_args = build_eval_args(args, video_path)
            returncode = 1
            attempts = max(1, args.eval_attempts)
            for attempt in range(1, attempts + 1):
                returncode = evalmod.run_evaluation(eval_args)
                run_logs.append(
                    {
                        "video_name": video_path.name,
                        "command": command,
                        "attempt": attempt,
                        "returncode": returncode,
                        "stdout": "",
                        "stderr": "",
                    }
                )
                if returncode == 0:
                    break
                if attempt < attempts:
                    print(f"Retrying {video_path.name} ({attempt}/{attempts})...", file=sys.stderr)
            if returncode != 0:
                print(f"评测失败：{video_path.name}", file=sys.stderr)
                continue

        if not eval_json_path.exists():
            print(f"警告：未生成结果文件，跳过汇总：{video_path.name}", file=sys.stderr)
            continue

        summary_rows.append(extract_result_row(video_path, eval_json_path))
        succeeded += 1

    average_accuracy = (
        sum(row["accuracy"] for row in summary_rows if isinstance(row.get("accuracy"), (int, float)))
        / len(summary_rows)
        if summary_rows
        else 0.0
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "videos_dir": str(videos_dir),
        "eval_script": str(eval_script),
        "start_from": args.start_from,
        "end_at": args.end_at,
        "entries": str(args.entries.resolve()) if args.entries is not None else str(evalmod.DEFAULT_ENTRIES_PATH),
        "metadata_root": str(args.metadata_root.resolve()) if args.metadata_root is not None else str(evalmod.DEFAULT_METADATA_ROOT),
        "question_mode": args.question_mode,
        "video_count": len(videos),
        "successful_count": succeeded,
        "average_accuracy": average_accuracy,
        "average_accuracy_percent": round(average_accuracy * 100, 2),
        "results": summary_rows,
        "run_logs": run_logs,
    }
    dump_json(summary_output, payload)
    dump_csv(csv_output, summary_rows)

    print(f"批量汇总已写入：{summary_output}")
    print(f"CSV 汇总已写入：{csv_output}")
    print(f"成功评测：{succeeded}/{len(videos)}")
    print(f"平均正确率：{average_accuracy:.2%}")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
