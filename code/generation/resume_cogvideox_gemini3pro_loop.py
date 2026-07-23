from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(os.environ.get("VGIF_REPO_ROOT", Path(__file__).resolve().parents[2]))
RUN_DIR = Path(os.environ.get("VGIF_RUN_DIR", REPO_ROOT / "models" / "CogVideoX-1.5"))
LOG_PATH = RUN_DIR / "logs" / "cogvideox_dependency_rounds_gemini3pro_loop_py.log"
BATCH_SCRIPT = REPO_ROOT / "code" / "evaluation" / "evaluate_kling_batch_accuracy.py"
TARGET_COUNT = 223
START_INDEX = 207
SLEEP_SECONDS = 90

API_KEY = os.environ.get("VGIF_API_KEY", "")
BASE_URL = os.environ.get("VGIF_BASE_URL", "")
MODEL = "gemini-3-pro-preview"


def write_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as file_obj:
        file_obj.write(f"[{timestamp}] {message}\n")


def completed_count() -> int:
    return len(list(RUN_DIR.glob("*_qa_eval_dependency_rounds.json")))


def build_command() -> list[str]:
    return [
        sys.executable,
        str(BATCH_SCRIPT),
        "--run-dir",
        str(RUN_DIR),
        "--question-mode",
        "dependency-rounds",
        "--start-index",
        str(START_INDEX),
        "--max-workers",
        "1",
        "--video-attempts",
        "12",
        "--timeout",
        "600",
        "--retries",
        "8",
        "--skip-model-check",
        "--reuse-existing",
        "--model",
        MODEL,
        "--api-key",
        API_KEY,
        "--base-url",
        BASE_URL,
    ]


def main() -> int:
    if not API_KEY or not BASE_URL:
        raise SystemExit("Set VGIF_API_KEY and VGIF_BASE_URL before running this resume loop.")
    write_log("Loop runner started.")
    command = build_command()

    while True:
        before = completed_count()
        write_log(f"Current completed count: {before} / {TARGET_COUNT}")
        if before >= TARGET_COUNT:
            write_log("Target reached. Exiting.")
            return 0

        completed_process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stdout_text = completed_process.stdout.strip()
        stderr_text = completed_process.stderr.strip()
        if stdout_text:
            write_log("stdout:\n" + stdout_text)
        if stderr_text:
            write_log("stderr:\n" + stderr_text)
        write_log(f"Round return code: {completed_process.returncode}")

        after = completed_count()
        write_log(f"Completed count after round: {after} / {TARGET_COUNT}")
        if after >= TARGET_COUNT:
            write_log("Target reached after round. Exiting.")
            return 0

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
