from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_GEMINI_DIR = SCRIPT_DIR
DEFAULT_ENTRIES_PATH = REPO_DIR / "kling_t2v" / "all_entries_merged_final.json"
DEFAULT_METADATA_ROOT = REPO_DIR / "kling_t2v" / "outputs"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "matched_video_qas.json"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match videos in gemini_test to all_entries_merged_final.json and export their QA answers."
    )
    parser.add_argument(
        "--gemini-dir",
        type=Path,
        default=DEFAULT_GEMINI_DIR,
        help="Directory containing local test videos.",
    )
    parser.add_argument(
        "--entries",
        type=Path,
        default=DEFAULT_ENTRIES_PATH,
        help="Path to all_entries_merged_final.json.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
        help="Root directory used to locate generated metadata files for each video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Consolidated output JSON path.",
    )
    parser.add_argument(
        "--skip-per-video",
        action="store_true",
        help="Only write the consolidated JSON file.",
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


def iter_video_files(gemini_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in gemini_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def find_matching_metadata(metadata_root: Path, video_stem: str) -> Path | None:
    candidates = sorted(
        metadata_root.rglob(f"{video_stem}.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_prompt_index(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    prompt_index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        prompt = entry.get("prompt")
        if isinstance(prompt, str) and prompt not in prompt_index:
            prompt_index[prompt] = entry
    return prompt_index


def read_description_payload(video_path: Path) -> dict[str, Any] | None:
    description_path = video_path.with_name(f"{video_path.stem}_description.json")
    if not description_path.exists():
        return None
    return load_json(description_path)


def format_qa_pairs(entry: dict[str, Any]) -> list[dict[str, Any]]:
    qa_pairs = entry.get("vlm_qa_pairs", [])
    formatted: list[dict[str, Any]] = []
    for qa in qa_pairs:
        formatted.append(
            {
                "id": qa.get("id"),
                "question": qa.get("question"),
                "answer": qa.get("expected_answer"),
                "expected_answer": qa.get("expected_answer"),
                "type": qa.get("type"),
                "dependency": qa.get("dependency"),
                "node_id": qa.get("node_id"),
            }
        )
    return formatted


def build_video_result(
    video_path: Path,
    metadata_path: Path | None,
    entry: dict[str, Any] | None,
) -> dict[str, Any]:
    description_payload = read_description_payload(video_path)
    result: dict[str, Any] = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "video_stem": video_path.stem,
        "description_file": str(video_path.with_name(f"{video_path.stem}_description.json")),
        "matched": entry is not None,
        "matched_metadata_file": str(metadata_path) if metadata_path else None,
    }

    if description_payload:
        result["gemini_description"] = description_payload.get("description")
        result["gemini_response_id"] = description_payload.get("response_id")
        result["gemini_model"] = description_payload.get("model")

    if not metadata_path or not entry:
        result["qa_pairs"] = []
        return result

    metadata_payload = load_json(metadata_path)
    task = metadata_payload.get("task", {})

    result.update(
        {
            "task_id": task.get("task_id"),
            "sample_index": task.get("sample_index"),
            "macro_domain": entry.get("domain_info", {}).get("macro_domain"),
            "micro_domain": entry.get("domain_info", {}).get("micro_domain"),
            "prompt": entry.get("prompt"),
            "qa_pairs": format_qa_pairs(entry),
        }
    )
    return result


def main() -> int:
    args = parse_args()

    entries = load_json(args.entries)
    prompt_index = build_prompt_index(entries)
    results: list[dict[str, Any]] = []

    for video_path in iter_video_files(args.gemini_dir):
        metadata_path = find_matching_metadata(args.metadata_root, video_path.stem)
        entry: dict[str, Any] | None = None

        if metadata_path:
            metadata_payload = load_json(metadata_path)
            prompt = metadata_payload.get("task", {}).get("prompt")
            if isinstance(prompt, str):
                entry = prompt_index.get(prompt)

        result = build_video_result(video_path, metadata_path, entry)
        results.append(result)

        if not args.skip_per_video:
            per_video_path = video_path.with_name(f"{video_path.stem}_qa_answers.json")
            dump_json(per_video_path, result)

    consolidated_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gemini_dir": str(args.gemini_dir),
        "entries_path": str(args.entries),
        "metadata_root": str(args.metadata_root),
        "video_count": len(results),
        "matched_count": sum(1 for item in results if item["matched"]),
        "results": results,
    }
    dump_json(args.output, consolidated_payload)

    print(
        f"Processed {len(results)} video(s); matched {consolidated_payload['matched_count']} entry/entries."
    )
    print(f"Consolidated output: {args.output}")
    if not args.skip_per_video:
        print("Per-video QA files were also written next to each video.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
