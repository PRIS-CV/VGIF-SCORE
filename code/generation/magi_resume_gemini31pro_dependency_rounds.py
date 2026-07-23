from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_DIR / "models" / "MAGI-1"
VIDEOS_DIR = RUN_DIR
PYTHON_EXE = Path(sys.executable)
ENTRIES = REPO_DIR / "data" / "vgif_bench" / "vgif_bench.jsonl"
MODEL = "gemini-3.1-pro-preview"
OUTPUT_TAG = MODEL
PER_VIDEO_TIMEOUT_SECONDS = 420


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume MAGI dependency-round QA evaluation with configurable parallelism."
    )
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def safe_print(text: str, *, is_stderr: bool = False) -> None:
    stream = sys.stderr if is_stderr else sys.stdout
    sanitized = text.encode(stream.encoding or "utf-8", errors="replace").decode(
        stream.encoding or "utf-8",
        errors="replace",
    )
    print(sanitized, end="" if sanitized.endswith("\n") else "\n", file=stream, flush=True)


def run_subprocess(args: list[str], timeout: int | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args,
            cwd=REPO_DIR,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "completed": result,
            "timed_out": False,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "completed": None,
            "timed_out": True,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "command": args,
            "timeout": timeout,
        }


def build_eval_command(video_path: Path) -> list[str]:
    return [
        str(PYTHON_EXE),
        "code\\evaluation\\evaluate_video_qa_accuracy.py",
        "--video",
        str(video_path),
        "--metadata-root",
        str(RUN_DIR),
        "--entries",
        str(ENTRIES),
        "--question-mode",
        "dependency-rounds",
        "--model",
        MODEL,
        "--output-tag",
        OUTPUT_TAG,
        "--skip-model-check",
        "--timeout",
        "300",
        "--retries",
        "2",
        "--format-retries",
        "1",
    ]


def evaluate_one_video(video_path: Path) -> dict[str, Any]:
    output_path = VIDEOS_DIR / f"{video_path.stem}_{OUTPUT_TAG}_qa_eval_dependency_rounds.json"
    if output_path.exists():
        return {
            "video_name": video_path.name,
            "status": "skip",
            "stdout": "",
            "stderr": "",
        }

    result = run_subprocess(build_eval_command(video_path), timeout=PER_VIDEO_TIMEOUT_SECONDS)
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")

    if output_path.exists():
        status = "done"
    elif result.get("timed_out"):
        status = "miss_timeout"
    else:
        status = "miss"

    return {
        "video_name": video_path.name,
        "status": status,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": bool(result.get("timed_out")),
        "command": result.get("command"),
        "timeout": result.get("timeout"),
    }


def main() -> int:
    args = parse_args()
    max_workers = max(1, args.max_workers)
    completed = 0
    failed = 0

    video_paths = sorted(VIDEOS_DIR.glob("*.mp4"))
    pending_video_paths = [
        path
        for path in video_paths
        if not (VIDEOS_DIR / f"{path.stem}_{OUTPUT_TAG}_qa_eval_dependency_rounds.json").exists()
    ]

    print(
        f"QUEUE total={len(video_paths)} pending={len(pending_video_paths)} max_workers={max_workers}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_one_video, video_path): video_path
            for video_path in pending_video_paths
        }
        for future in as_completed(futures):
            payload = future.result()
            video_name = payload["video_name"]
            status = payload["status"]

            if payload.get("timed_out") and payload.get("command"):
                print(
                    f"TIMEOUT after {payload.get('timeout')}s: {' '.join(payload['command'])}",
                    flush=True,
                )
            if payload.get("stdout"):
                safe_print(payload["stdout"])
            if payload.get("stderr"):
                safe_print(payload["stderr"], is_stderr=True)

            if status == "done":
                completed += 1
                print(f"DONE {video_name}", flush=True)
            elif status == "skip":
                print(f"SKIP {video_name}", flush=True)
            elif status == "miss_timeout":
                failed += 1
                print(f"MISS_TIMEOUT {video_name}", flush=True)
            else:
                failed += 1
                print(f"MISS {video_name}", flush=True)

    print(f"SEQUENTIAL_DONE completed={completed} failed={failed}", flush=True)

    summary_path = RUN_DIR / "qa_eval_dependency_rounds_summary_gemini-3.1-pro-preview.json"
    run_subprocess(
        [
            str(PYTHON_EXE),
            "gemini_test\\summarize_qa_results.py",
            "--videos-dir",
            str(VIDEOS_DIR),
            "--suffix",
            "_gemini-3.1-pro-preview_qa_eval_dependency_rounds.json",
            "--output",
            str(summary_path),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
