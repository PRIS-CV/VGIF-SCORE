from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_io import load_entries_file
from scoring import RUBRIC_DIMENSIONS, mean_rubric_rating, mean_rubric_score, to_percent

from test_gemini_video import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    build_request_payload,
    build_session,
    encode_file_to_base64,
    fetch_available_models,
    guess_mime_type,
    parse_json_response,
    print_model_check,
    send_generate_content_request,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_AUTORUBRIC_DATASET_DIR = REPO_DIR / "data" / "vgif_bench"
DEFAULT_ENTRIES_PATH = DEFAULT_AUTORUBRIC_DATASET_DIR / "vgif_bench.jsonl"
DEFAULT_METADATA_ROOT = REPO_DIR / "models"
DEFAULT_VIDEOS_DIR = REPO_DIR / "models" / "Wan-2.7"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_FORMAT_RETRIES = 2
DEFAULT_AUTORUBRIC_MODEL = "gemini-3-pro-preview"
AUTORUBRIC_DATASET_FILENAMES = (
    "vgif_bench.jsonl",
    "all_autorubric_gpt52_v2.json",
    "all_autorubric_enhanced_qa.json",
)

SCORE_DIMENSIONS = [
    ("Cin", "cinematography", "Cinematography"),
    ("Pur", "purity", "Purity"),
    ("Mot", "motion_smoothness", "Motion smoothness"),
    ("Phy", "physics_adherence", "Physics adherence"),
]
ALL_SCORE_KEYS = list(RUBRIC_DIMENSIONS)
DIMENSION_KEY_ALIASES = {
    "cinematography": ["cinematography"],
    "purity": ["purity", "visual_purity"],
    "motion_smoothness": ["motion_smoothness"],
    "physics_adherence": ["physics_adherence"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score videos with vgif_autorubric_full 1-5 auto-rubric dimensions."
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_AUTORUBRIC_MODEL)
    parser.add_argument("--video", type=Path, default=None, help="Evaluate one video.")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help="Evaluate videos from this directory when --video is omitted.",
    )
    parser.add_argument("--entries", type=Path, default=DEFAULT_ENTRIES_PATH)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument(
        "--progress-jsonl",
        type=Path,
        default=None,
        help="Optional progress_success.jsonl used to map video names to prompts directly.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Single-video output path.")
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional suffix inserted before _autorubric_eval.json for batch-friendly runs.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Batch summary JSON path.",
    )
    parser.add_argument("--csv-output", type=Path, default=None, help="Batch summary CSV path.")
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--end-at", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--format-retries", type=int, default=DEFAULT_FORMAT_RETRIES)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def dataset_rank(entries: list[dict[str, Any]], source_path: Path) -> tuple[int, int, int]:
    sample = entries[: min(len(entries), 5)]
    richness = 0
    if any(isinstance(item.get("autorubric"), dict) for item in sample):
        richness += 2
    if any(isinstance(item.get("original_qa_pairs"), list) and item.get("original_qa_pairs") for item in sample):
        richness += 1
    if any(
        isinstance(item.get("enhanced_qa"), dict)
        and isinstance(item.get("enhanced_qa", {}).get("enhanced_qa_pairs"), list)
        and item.get("enhanced_qa", {}).get("enhanced_qa_pairs")
        for item in sample
    ):
        richness += 1

    source_priority = 0
    if source_path.name == "vgif_bench.jsonl":
        source_priority = 4
    elif source_path.name == "all_autorubric_gpt52_v2.json":
        source_priority = 3
    elif source_path.name == "all_autorubric_enhanced_qa.json":
        source_priority = 2
    elif source_path.is_dir():
        source_priority = 1
    return len(entries), richness, source_priority


def load_entries_dataset(entries_path: Path) -> tuple[list[dict[str, Any]], Path]:
    resolved = entries_path.resolve()

    def load_list_file(path: Path) -> list[dict[str, Any]] | None:
        return load_entries_file(path)

    def load_sample_entries(directory: Path) -> tuple[list[dict[str, Any]], Path] | None:
        sample_files = sorted(directory.glob("sample_*_complete.json"))
        if len(sample_files) < 200:
            return None
        merged_entries: list[dict[str, Any]] = []
        for path in sample_files:
            payload = load_json(path)
            if isinstance(payload, dict):
                merged_entries.append(payload)
        if len(merged_entries) >= 200:
            return merged_entries, directory
        return None

    def choose_best_dataset(
        candidates: list[tuple[list[dict[str, Any]], Path]],
    ) -> tuple[list[dict[str, Any]], Path] | None:
        if not candidates:
            return None
        return max(candidates, key=lambda item: dataset_rank(item[0], item[1]))

    if resolved.is_file():
        entries = load_list_file(resolved)
        if entries is None:
            raise ValueError(f"Entries file is not a JSON list: {resolved}")
        if len(entries) >= 200:
            return entries, resolved

        if resolved.name in AUTORUBRIC_DATASET_FILENAMES:
            candidates: list[tuple[list[dict[str, Any]], Path]] = [(entries, resolved)]
            sample_entries = load_sample_entries(resolved.parent)
            if sample_entries is not None:
                candidates.append(sample_entries)
            for filename in AUTORUBRIC_DATASET_FILENAMES:
                candidate_path = resolved.parent / filename
                if candidate_path == resolved:
                    continue
                candidate_entries = load_list_file(candidate_path)
                if candidate_entries:
                    candidates.append((candidate_entries, candidate_path))
            best_dataset = choose_best_dataset(candidates)
            if best_dataset is not None:
                return best_dataset
        return entries, resolved

    if resolved.is_dir():
        candidates: list[tuple[list[dict[str, Any]], Path]] = []
        sample_entries = load_sample_entries(resolved)
        if sample_entries is not None:
            candidates.append(sample_entries)

        for filename in AUTORUBRIC_DATASET_FILENAMES:
            candidate = resolved / filename
            entries = load_list_file(candidate)
            if entries:
                candidates.append((entries, candidate))

        best_dataset = choose_best_dataset(candidates)
        if best_dataset is not None:
            return best_dataset

    raise FileNotFoundError(f"Could not load autorubric dataset from: {resolved}")


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_name",
        "output_file",
        "sample_id",
        "sample_index",
        "Cin",
        "Pur",
        "Mot",
        "Phy",
        "score_average",
        "subjective_score",
        "subjective_score_percent",
        "model",
        "model_version",
        "response_id",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def iter_target_videos(videos_dir: Path, start_from: str | None, end_at: str | None) -> list[Path]:
    videos: list[Path] = []
    for path in sorted(videos_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if start_from and path.name < start_from:
            continue
        if end_at and path.name > end_at:
            continue
        videos.append(path)
    return videos


def load_progress_prompt_map(progress_jsonl: Path | None) -> dict[str, dict[str, Any]]:
    if progress_jsonl is None:
        return {}

    prompt_map: dict[str, dict[str, Any]] = {}
    with progress_jsonl.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            output_name = payload.get("output_name")
            if isinstance(output_name, str) and output_name:
                prompt_map[output_name] = payload
    return prompt_map


def find_entry_by_prompt(
    entries: list[dict[str, Any]],
    prompt: str,
) -> tuple[dict[str, Any], int, int]:
    matches: list[tuple[int, dict[str, Any]]] = [
        (index, entry)
        for index, entry in enumerate(entries)
        if entry.get("prompt") == prompt
    ]
    if not matches:
        raise ValueError("Prompt not found in autorubric entries dataset.")
    if len(matches) > 1:
        raise ValueError(f"Prompt is duplicated in autorubric entries dataset: {prompt[:160]}...")
    entry_index, entry = matches[0]
    return entry, len(matches), entry_index


def find_matching_metadata(metadata_root: Path, video_stem: str) -> Path | None:
    candidates = sorted(
        (
            path
            for pattern in (f"{video_stem}.json", f"{video_stem}_config.json")
            for path in metadata_root.rglob(pattern)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def extract_metadata_prompt(metadata_payload: dict[str, Any]) -> str | None:
    task_payload = metadata_payload.get("task")
    candidates = [
        task_payload.get("prompt") if isinstance(task_payload, dict) else None,
        metadata_payload.get("prompt"),
        metadata_payload.get("final_prompt"),
        metadata_payload.get("submission_payload", {}).get("prompt"),
        metadata_payload.get("retry_prompt"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def find_matching_entry(
    entries: list[dict[str, Any]],
    metadata_payload: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    prompt = extract_metadata_prompt(metadata_payload)
    if prompt:
        matches = [entry for entry in entries if entry.get("prompt") == prompt]
        if len(matches) == 1:
            return matches[0], 1
        if len(matches) > 1:
            raise ValueError(f"Multiple entries matched prompt: {prompt[:160]}...")

    sample_index_candidates = [
        metadata_payload.get("task", {}).get("sample_index")
        if isinstance(metadata_payload.get("task"), dict)
        else None,
        metadata_payload.get("sample_index"),
        metadata_payload.get("index"),
        metadata_payload.get("global_index"),
    ]
    for sample_index in sample_index_candidates:
        if not isinstance(sample_index, int):
            continue

        # Some metadata uses 1-based sample_index, while other exporters keep 0-based global_index/index.
        if 1 <= sample_index <= len(entries):
            return entries[sample_index - 1], 0
        if 0 <= sample_index < len(entries):
            return entries[sample_index], 0

    raise ValueError("Could not match metadata prompt or sample_index to autorubric entry.")


def infer_sample_index_from_video_name(video_path: Path) -> int | None:
    match = re.search(r"(\d{3,4})(?!.*\d)", video_path.stem)
    if match is None:
        return None
    sample_index = int(match.group(1))
    return sample_index if sample_index > 0 else None


def find_entry_by_sample_index(
    entries: list[dict[str, Any]],
    sample_index: int,
) -> tuple[dict[str, Any], int]:
    zero_based_index = sample_index - 1
    if 0 <= zero_based_index < len(entries):
        return entries[zero_based_index], 0
    raise ValueError(
        f"sample_index {sample_index} is out of range for autorubric entries ({len(entries)} total)."
    )


def resolve_entry_for_video(
    *,
    entries: list[dict[str, Any]],
    metadata_root: Path,
    progress_prompt_map: dict[str, dict[str, Any]],
    video_path: Path,
) -> tuple[dict[str, Any], int, Path | None, dict[str, Any], int | None, str]:
    progress_record = progress_prompt_map.get(video_path.name)
    if progress_record is not None:
        prompt = progress_record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing prompt in progress jsonl for {video_path.name}")
        matched_entry, matched_prompt_occurrences, entry_index = find_entry_by_prompt(entries, prompt)
        progress_index = progress_record.get("index")
        sample_index = progress_index + 1 if isinstance(progress_index, int) else entry_index + 1
        return (
            matched_entry,
            matched_prompt_occurrences,
            None,
            {"task": {"task_id": None, "sample_index": sample_index}},
            sample_index,
            "progress_jsonl_prompt",
        )

    metadata_path = find_matching_metadata(metadata_root, video_path.stem)
    metadata_payload: dict[str, Any] = {}
    metadata_error: str | None = None

    if metadata_path is not None:
        metadata_payload = load_json(metadata_path)
        try:
            matched_entry, matched_prompt_occurrences = find_matching_entry(entries, metadata_payload)
            sample_index = (
                metadata_payload.get("task", {}).get("sample_index")
                if isinstance(metadata_payload.get("task"), dict)
                else metadata_payload.get("sample_index")
            )
            if isinstance(sample_index, int):
                if 0 <= sample_index < len(entries):
                    sample_index += 1
            else:
                sample_index = infer_sample_index_from_video_name(video_path)
            return (
                matched_entry,
                matched_prompt_occurrences,
                metadata_path,
                metadata_payload,
                sample_index,
                "metadata",
            )
        except ValueError as exc:
            metadata_error = str(exc)

    sample_index = infer_sample_index_from_video_name(video_path)
    if sample_index is not None:
        matched_entry, matched_prompt_occurrences = find_entry_by_sample_index(entries, sample_index)
        return (
            matched_entry,
            matched_prompt_occurrences,
            metadata_path,
            metadata_payload,
            sample_index,
            "video_filename_sample_index",
        )

    if metadata_error:
        raise ValueError(
            f"{metadata_error}; also could not infer sample index from video name {video_path.name}."
        )
    raise ValueError(
        f"No metadata match and could not infer sample index from video name {video_path.name}."
    )


def extract_first_text(response_payload: dict[str, Any]) -> str | None:
    for candidate in response_payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                return text
    return None


def extract_json_text(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def compact_qa_checklist(entry: dict[str, Any]) -> str:
    qa_pairs = entry.get("original_qa_pairs", [])
    if not isinstance(qa_pairs, list) or not qa_pairs:
        return "None"

    lines: list[str] = []
    for qa in qa_pairs:
        qa_id = qa.get("id")
        question = qa.get("question")
        dependency = qa.get("dependency")
        if qa_id and question:
            lines.append(f"{qa_id} | dependency={dependency} | {question}")
    return "\n".join(lines) if lines else "None"


def resolve_dimension_payload(entry: dict[str, Any], dimension_key: str) -> dict[str, Any]:
    dimensions = entry.get("autorubric", {}).get("dimensions", {})
    if not isinstance(dimensions, dict):
        return {}

    for candidate_key in DIMENSION_KEY_ALIASES.get(dimension_key, [dimension_key]):
        payload = dimensions.get(candidate_key)
        if isinstance(payload, dict):
            return payload
    return {}


def format_dimension_block(code: str, dimension_key: str, display_name: str, entry: dict[str, Any]) -> str:
    dimension = resolve_dimension_payload(entry, dimension_key)
    anchors = dimension.get("score_anchors", {})
    anchors_blob = "\n".join(
        f"    {score}: {text}"
        for score, text in sorted(anchors.items(), key=lambda item: int(item[0]))
    )
    focus_points = dimension.get("focus_points", [])
    focus_blob = ", ".join(focus_points) if isinstance(focus_points, list) else str(focus_points)

    return (
        f"{code} ({display_name})\n"
        f"  Goal: {dimension.get('dimension_goal')}\n"
        f"  Scoring prompt: {dimension.get('vlm_scoring_prompt')}\n"
        f"  Focus points: {focus_blob}\n"
        "  Score anchors:\n"
        f"{anchors_blob}"
    )


def build_autorubric_prompt(entry: dict[str, Any]) -> str:
    dimension_blocks = "\n\n".join(
        format_dimension_block(code, dimension_key, display_name, entry)
        for code, dimension_key, display_name in SCORE_DIMENSIONS
    )
    qa_checklist = compact_qa_checklist(entry)

    return (
        "You are a strict video auto-rubric evaluator for text-to-video generation.\n"
        "Inspect the video carefully and assign integer scores from 1 to 5 for each requested dimension.\n"
        "Use only visible evidence in the video. Do not give credit for events that are only implied by the prompt.\n"
        "Be conservative: if evidence is ambiguous, lower the score.\n"
        "\n"
        "Original text-to-video prompt:\n"
        f"{entry.get('prompt')}\n"
        "\n"
        "QA checklist from the dataset. Use this as supporting context for prompt-specific requirements:\n"
        f"{qa_checklist}\n"
        "\n"
        "Score these four dimensions:\n"
        f"{dimension_blocks}\n\n"
        "\n"
        "Return JSON only with this exact schema:\n"
        "{\n"
        '  "scores": {\n'
        '    "Cin": {"score": 1, "reason": "short visual reason"},\n'
        '    "Pur": {"score": 1, "reason": "short visual reason"},\n'
        '    "Mot": {"score": 1, "reason": "short visual reason"},\n'
        '    "Phy": {"score": 1, "reason": "short visual reason"}\n'
        "  },\n"
        '  "summary": "one sentence summary"\n'
        "}\n"
        "Every score must be an integer 1, 2, 3, 4, or 5. Keep each reason under 24 words."
    )


def normalize_score(value: Any) -> int:
    if isinstance(value, int):
        score = value
    elif isinstance(value, float) and value.is_integer():
        score = int(value)
    elif isinstance(value, str):
        match = re.search(r"\b[1-5]\b", value)
        if not match:
            raise ValueError(f"Invalid score value: {value}")
        score = int(match.group(0))
    else:
        raise ValueError(f"Invalid score value: {value}")

    if score < 1 or score > 5:
        raise ValueError(f"Score out of range: {score}")
    return score


def normalize_scores(model_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_scores = model_payload.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("Missing scores object.")

    normalized: dict[str, dict[str, Any]] = {}
    for key in ALL_SCORE_KEYS:
        payload = raw_scores.get(key)
        if not isinstance(payload, dict):
            raise ValueError(f"Missing score payload for {key}.")
        normalized[key] = {
            "score": normalize_score(payload.get("score")),
            "reason": str(payload.get("reason", "")).strip(),
        }
    return normalized


def request_scores(
    *,
    session: Any,
    args: argparse.Namespace,
    prompt: str,
    mime_type: str,
    video_base64: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    max_attempts = max(1, int(args.format_retries) + 1)
    last_error: Exception | None = None
    last_raw_text: str | None = None

    for attempt in range(1, max_attempts + 1):
        request_payload = build_request_payload(
            prompt=prompt,
            model=args.model,
            mime_type=mime_type,
            video_base64=video_base64,
        )
        response = send_generate_content_request(
            session=session,
            base_url=args.base_url,
            model=args.model,
            payload=request_payload,
            timeout=args.timeout,
            retries=args.retries,
        )
        response_payload = parse_json_response(response)
        if response_payload is None:
            last_error = ValueError(f"HTTP response was not JSON: {response.text[:500]}")
            continue
        if not response.ok:
            last_error = ValueError(f"HTTP {response.status_code}: {response.text[:500]}")
            continue

        raw_text = extract_first_text(response_payload)
        if not raw_text:
            last_error = ValueError("Model response did not contain text.")
            continue
        last_raw_text = raw_text

        try:
            model_payload = extract_json_text(raw_text)
            scores = normalize_scores(model_payload)
            return (
                {
                    "scores": scores,
                    "summary": model_payload.get("summary"),
                    "raw_model_text": raw_text,
                },
                {
                    "model_version": response_payload.get("modelVersion"),
                    "response_id": response_payload.get("responseId"),
                    "http_status": response.status_code,
                    "usage_metadata": response_payload.get("usageMetadata"),
                    "format_attempt": attempt,
                },
            )
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    assert last_error is not None
    if last_raw_text:
        raise ValueError(f"{last_error}\nRAW_MODEL_TEXT:\n{last_raw_text}") from last_error
    raise last_error


def build_output_path(
    video_path: Path,
    custom_output: Path | None,
    output_tag: str | None = None,
) -> Path:
    if custom_output is not None:
        return custom_output.resolve()
    if output_tag:
        safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", output_tag.strip()).strip("-")
        if safe_tag:
            return video_path.with_name(f"{video_path.stem}_{safe_tag}_autorubric_eval.json")
    return video_path.with_name(f"{video_path.stem}_autorubric_eval.json")


def score_average(scores: dict[str, dict[str, Any]]) -> float:
    return round(mean_rubric_rating(scores), 3)


def evaluate_one_video(
    *,
    args: argparse.Namespace,
    session: Any,
    entries: list[dict[str, Any]],
    progress_prompt_map: dict[str, dict[str, Any]],
    resolved_entries_path: Path,
    video_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    (
        matched_entry,
        matched_prompt_occurrences,
        metadata_path,
        metadata_payload,
        resolved_sample_index,
        entry_match_source,
    ) = resolve_entry_for_video(
        entries=entries,
        metadata_root=args.metadata_root.resolve(),
        progress_prompt_map=progress_prompt_map,
        video_path=video_path,
    )
    eval_prompt = build_autorubric_prompt(matched_entry)

    video_base64 = encode_file_to_base64(video_path)
    mime_type = guess_mime_type(video_path)
    score_payload, execution = request_scores(
        session=session,
        args=args,
        prompt=eval_prompt,
        mime_type=mime_type,
        video_base64=video_base64,
    )

    scores = score_payload["scores"]
    normalized_score = mean_rubric_score(scores)
    result_payload = {
        "success": True,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "video_file": str(video_path),
        "output_file": str(output_path),
        "metadata_file": str(metadata_path) if metadata_path is not None else None,
        "entries_file": str(resolved_entries_path),
        "sample_id": matched_entry.get("sample_id"),
        "sample_index": resolved_sample_index,
        "matched_entry_index": matched_entry.get("index"),
        "matched_prompt_occurrences": matched_prompt_occurrences,
        "entry_match_source": entry_match_source,
        "model": args.model,
        "model_version": execution.get("model_version"),
        "response_id": execution.get("response_id"),
        "http_status": execution.get("http_status"),
        "usage_metadata": execution.get("usage_metadata"),
        "format_attempt": execution.get("format_attempt"),
        "score_scale": "1-5 integer, higher is better",
        "normalization": "mean(rating) / 5 over Cin, Pur, Mot, and Phy",
        "score_mapping": {
            "Cin": "cinematography",
            "Pur": "visual_purity",
            "Mot": "motion_smoothness",
            "Phy": "physics_adherence",
        },
        "scores": scores,
        "score_average": score_average(scores),
        "subjective_score": round(normalized_score, 6),
        "subjective_score_percent": to_percent(normalized_score),
        "summary": score_payload.get("summary"),
        "prompt": matched_entry.get("prompt"),
        "eval_prompt": eval_prompt,
        "raw_model_text": score_payload.get("raw_model_text"),
    }
    dump_json(output_path, result_payload)
    return result_payload


def result_row(video_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    scores = payload.get("scores", {})
    row = {
        "video_name": video_path.name,
        "output_file": payload.get("output_file"),
        "sample_id": payload.get("sample_id"),
        "sample_index": payload.get("sample_index"),
        "score_average": payload.get("score_average"),
        "subjective_score": payload.get("subjective_score"),
        "subjective_score_percent": payload.get("subjective_score_percent"),
        "model": payload.get("model"),
        "model_version": payload.get("model_version"),
        "response_id": payload.get("response_id"),
    }
    for key in ALL_SCORE_KEYS:
        score_payload = scores.get(key, {}) if isinstance(scores, dict) else {}
        row[key] = score_payload.get("score")
    return row


def main() -> int:
    args = parse_args()
    try:
        entries, resolved_entries_path = load_entries_dataset(args.entries)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    progress_jsonl = args.progress_jsonl.resolve() if args.progress_jsonl is not None else None
    try:
        progress_prompt_map = load_progress_prompt_map(progress_jsonl)
    except Exception as exc:
        print(f"Error: failed to load progress jsonl: {exc}", file=sys.stderr)
        return 1

    session = build_session(args.api_key)
    if not args.skip_model_check:
        try:
            models = fetch_available_models(session, args.base_url, args.timeout)
            print_model_check(args.model, models)
        except Exception as exc:
            print(f"Warning: model check failed: {exc}", file=sys.stderr)

    if args.video is not None:
        video_path = args.video.resolve()
        output_path = build_output_path(video_path, args.output, args.output_tag)
        try:
            result = evaluate_one_video(
                args=args,
                session=session,
                entries=entries,
                progress_prompt_map=progress_prompt_map,
                resolved_entries_path=resolved_entries_path,
                video_path=video_path,
                output_path=output_path,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        scores = result["scores"]
        score_line = ", ".join(f"{key}={scores[key]['score']}" for key in ALL_SCORE_KEYS)
        print(f"Autorubric result written: {output_path}")
        print(f"Scores: {score_line}; avg={result['score_average']}")
        return 0

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.exists():
        print(f"Error: videos directory not found: {videos_dir}", file=sys.stderr)
        return 1

    videos = iter_target_videos(videos_dir, args.start_from, args.end_at)
    if args.max_items is not None:
        if args.max_items <= 0:
            print("Error: --max-items must be positive.", file=sys.stderr)
            return 1
        videos = videos[: args.max_items]
    if not videos:
        print("Error: no target videos found.", file=sys.stderr)
        return 1

    summary_output = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else videos_dir / "autorubric_eval_batch_summary.json"
    )
    csv_output = (
        args.csv_output.resolve()
        if args.csv_output is not None
        else summary_output.with_suffix(".csv")
    )

    rows: list[dict[str, Any]] = []
    run_logs: list[dict[str, Any]] = []
    for video_path in videos:
        output_path = build_output_path(video_path, None, args.output_tag)
        if output_path.exists() and not args.overwrite:
            payload = load_json(output_path)
            rows.append(result_row(video_path, payload))
            run_logs.append({"video_name": video_path.name, "status": "skipped_existing"})
            continue

        try:
            payload = evaluate_one_video(
                args=args,
                session=session,
                entries=entries,
                progress_prompt_map=progress_prompt_map,
                resolved_entries_path=resolved_entries_path,
                video_path=video_path,
                output_path=output_path,
            )
            rows.append(result_row(video_path, payload))
            run_logs.append({"video_name": video_path.name, "status": "success"})
            print(f"Scored {video_path.name}: avg={payload['score_average']}")
        except Exception as exc:
            run_logs.append({"video_name": video_path.name, "status": "failed", "error": str(exc)})
            print(f"Failed {video_path.name}: {exc}", file=sys.stderr)

    summary_payload = {
        "success": bool(rows),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "videos_dir": str(videos_dir),
        "entries_file": str(resolved_entries_path),
        "metadata_root": str(args.metadata_root.resolve()),
        "progress_jsonl": str(progress_jsonl) if progress_jsonl is not None else None,
        "model": args.model,
        "requested_count": len(videos),
        "succeeded_count": len(rows),
        "score_mapping": {
            "Cin": "cinematography",
            "Pur": "visual_purity",
            "Mot": "motion_smoothness",
            "Phy": "physics_adherence",
        },
        "rows": rows,
        "run_logs": run_logs,
    }
    dump_json(summary_output, summary_payload)
    dump_csv(csv_output, rows)
    print(f"Batch summary written: {summary_output}")
    print(f"Batch CSV written: {csv_output}")
    print(f"Succeeded {len(rows)}/{len(videos)} video(s).")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
