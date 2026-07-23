from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_DIR / "models" / "MAGI-1"
PYTHON_EXE = Path(sys.executable)
ENTRIES = REPO_DIR / "data" / "vgif_bench" / "vgif_bench.jsonl"
MODEL = "gemini-3.1-pro-preview"
OUTPUT_TAG = MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume MAGI autorubric evaluation with Gemini 3.1 Pro Preview."
    )
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--format-retries", type=int, default=0)
    parser.add_argument("--video-attempts", type=int, default=1)
    return parser.parse_args()


def run_subprocess(args: list[str]) -> int:
    result = subprocess.run(
        args,
        cwd=REPO_DIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return int(result.returncode)


def main() -> int:
    args = parse_args()
    summary_path = RUN_DIR / f"autorubric_summary_{OUTPUT_TAG}.json"

    command = [
        str(PYTHON_EXE),
        "code\\evaluation\\evaluate_kling_batch_accuracy.py",
        "--run-dir",
        str(RUN_DIR),
        "--entries",
        str(ENTRIES),
        "--question-mode",
        "autorubric",
        "--model",
        MODEL,
        "--output-tag",
        OUTPUT_TAG,
        "--skip-model-check",
        "--max-workers",
        str(max(1, args.max_workers)),
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
        "--format-retries",
        str(args.format_retries),
        "--video-attempts",
        str(args.video_attempts),
        "--reuse-existing",
        "--output",
        str(summary_path),
    ]
    return run_subprocess(command)


if __name__ == "__main__":
    raise SystemExit(main())
