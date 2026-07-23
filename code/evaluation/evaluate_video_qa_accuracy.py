from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dataset_io import load_entries_file

from scoring import normalize_rating, to_percent as normalized_percent

from test_gemini_video import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
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
DEFAULT_ENTRIES_PATH = REPO_DIR / "data" / "vgif_bench" / "vgif_bench.jsonl"
DEFAULT_AUTORUBRIC_DATASET_DIR = REPO_DIR / "data" / "autorubric"
DEFAULT_METADATA_ROOT = REPO_DIR / "models"
DEFAULT_VIDEO_PATH = REPO_DIR / "video.mp4"
DEFAULT_FORMAT_RETRIES = 2
QUESTION_MODE_ALL_AT_ONCE = "all-at-once"
QUESTION_MODE_DEPENDENCY_ROUNDS = "dependency-rounds"
QUESTION_MODE_AUTORUBRIC = "autorubric"
AUTORUBRIC_BASE_DIMENSIONS = [
    ("Cin", "cinematography"),
    ("Pur", "purity"),
    ("Mot", "motion_smoothness"),
    ("Phy", "physics_adherence"),
]
AUTORUBRIC_DIMENSIONS = AUTORUBRIC_BASE_DIMENSIONS
AUTORUBRIC_DISPLAY_NAMES = {
    "Cin": "Cinematography",
    "Pur": "Purity",
    "Mot": "Motion smoothness",
    "Phy": "Physics adherence",
}
AUTORUBRIC_COMPLETE_THRESHOLD = 200
AUTORUBRIC_DATASET_FILENAMES = (
    "vgif_bench.jsonl",
    "all_autorubric_gpt52_v2.json",
    "all_autorubric_enhanced_qa.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a video against its QA pairs and compute Gemini accuracy."
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="代理接口使用的 API Key。")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="代理接口的基础地址，不要以斜杠结尾。",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="要调用的模型 ID。")
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="待评测视频文件路径。",
    )
    parser.add_argument(
        "--entries",
        type=Path,
        default=DEFAULT_ENTRIES_PATH,
        help="all_entries_merged_final.json 路径。",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
        help="用于按文件名定位 metadata 的根目录。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="单次 HTTP 请求超时时间，单位为秒。",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="主请求失败后额外重试次数。",
    )
    parser.add_argument(
        "--format-retries",
        type=int,
        default=DEFAULT_FORMAT_RETRIES,
        help="模型返回 JSON 格式异常时额外重试次数。",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="跳过 /v1/models 预检查。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="评测结果输出路径；不传时默认写到视频同目录。",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional tag appended to the default per-video output filename.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印更多调试信息。",
    )
    parser.add_argument(
        "--question-mode",
        choices=[
            QUESTION_MODE_ALL_AT_ONCE,
            QUESTION_MODE_DEPENDENCY_ROUNDS,
            QUESTION_MODE_AUTORUBRIC,
        ],
        default=QUESTION_MODE_ALL_AT_ONCE,
        help="QA 提问方式：一次性提问或按 dependency 分轮提问。",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_entries_dataset(entries_path: Path) -> tuple[list[dict[str, Any]], Path]:
    resolved = entries_path.resolve()

    def load_list_file(path: Path) -> list[dict[str, Any]] | None:
        return load_entries_file(path)

    if resolved.is_file():
        entries = load_list_file(resolved)
        if entries is None:
            raise ValueError(f"entries 文件不是 JSON list：{resolved}")
        if len(entries) >= AUTORUBRIC_COMPLETE_THRESHOLD:
            return entries, resolved

        if resolved.name in {"all_autorubric_enhanced_qa.json", "all_autorubric_gpt52_v2.json"}:
            sample_files = sorted(resolved.parent.glob("sample_*_complete.json"))
            if len(sample_files) >= AUTORUBRIC_COMPLETE_THRESHOLD:
                merged_entries: list[dict[str, Any]] = []
                for path in sample_files:
                    payload = load_json(path)
                    if isinstance(payload, dict):
                        merged_entries.append(payload)
                if len(merged_entries) >= AUTORUBRIC_COMPLETE_THRESHOLD:
                    return merged_entries, resolved.parent

            for filename in ("all_autorubric_gpt52_v2.json", "all_autorubric_enhanced_qa.json"):
                candidate = resolved.parent / filename
                if candidate == resolved:
                    continue
                candidate_entries = load_list_file(candidate)
                if candidate_entries and len(candidate_entries) > len(entries):
                    return candidate_entries, candidate
        return entries, resolved

    if resolved.is_dir():
        candidate_files = [
            resolved / "all_autorubric_enhanced_qa.json",
            resolved / "all_autorubric_gpt52_v2.json",
            resolved / "all_entries_merged_final.json",
        ]
        for candidate in candidate_files:
            entries = load_list_file(candidate)
            if entries and len(entries) >= AUTORUBRIC_COMPLETE_THRESHOLD:
                return entries, candidate

        sample_files = sorted(resolved.glob("sample_*_complete.json"))
        if len(sample_files) >= AUTORUBRIC_COMPLETE_THRESHOLD:
            merged_entries = [
                load_json(path)
                for path in sample_files
                if isinstance(load_json(path), dict)
            ]
            if len(merged_entries) >= AUTORUBRIC_COMPLETE_THRESHOLD:
                return merged_entries, resolved

        for candidate in candidate_files:
            entries = load_list_file(candidate)
            if entries:
                return entries, candidate

    raise FileNotFoundError(f"无法从 entries 路径加载数据集：{resolved}")


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
        if len(sample_files) < AUTORUBRIC_COMPLETE_THRESHOLD:
            return None
        merged_entries: list[dict[str, Any]] = []
        for path in sample_files:
            payload = load_json(path)
            if isinstance(payload, dict):
                merged_entries.append(payload)
        if len(merged_entries) >= AUTORUBRIC_COMPLETE_THRESHOLD:
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
        if len(entries) >= AUTORUBRIC_COMPLETE_THRESHOLD:
            return entries, resolved

        if resolved.name in AUTORUBRIC_DATASET_FILENAMES:
            candidates: list[tuple[list[dict[str, Any]], Path]] = [(entries, resolved)]
            sample_entries = load_sample_entries(resolved.parent)
            if sample_entries is not None:
                candidates.append(sample_entries)
            for filename in AUTORUBRIC_DATASET_FILENAMES:
                candidate = resolved.parent / filename
                if candidate == resolved:
                    continue
                candidate_entries = load_list_file(candidate)
                if candidate_entries:
                    candidates.append((candidate_entries, candidate))
            best_dataset = choose_best_dataset(candidates)
            if best_dataset is not None:
                return best_dataset
        return entries, resolved

    if resolved.is_dir():
        candidates: list[tuple[list[dict[str, Any]], Path]] = []
        sample_entries = load_sample_entries(resolved)
        if sample_entries is not None:
            candidates.append(sample_entries)

        candidate_files = [resolved / filename for filename in AUTORUBRIC_DATASET_FILENAMES]
        candidate_files.append(resolved / "all_entries_merged_final.json")
        for candidate in candidate_files:
            entries = load_list_file(candidate)
            if entries:
                candidates.append((entries, candidate))

        best_dataset = choose_best_dataset(candidates)
        if best_dataset is not None:
            return best_dataset

    raise FileNotFoundError(f"Could not load entries dataset from: {resolved}")


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


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


def find_matching_entries(entries_path: Path, metadata_path: Path) -> tuple[str, list[dict[str, Any]]]:
    metadata_payload = load_json(metadata_path)
    task_payload = metadata_payload.get("task")
    prompt_candidates = [
        task_payload.get("prompt") if isinstance(task_payload, dict) else None,
        metadata_payload.get("prompt"),
        metadata_payload.get("submission_payload", {}).get("prompt"),
        metadata_payload.get("retry_prompt"),
    ]
    prompt = next(
        (
            candidate.strip()
            for candidate in prompt_candidates
            if isinstance(candidate, str) and candidate.strip()
        ),
        None,
    )
    if not isinstance(prompt, str):
        raise ValueError(f"metadata 中缺少 prompt：{metadata_path}")

    entries, _ = load_entries_dataset(entries_path)
    matches = [entry for entry in entries if entry.get("prompt") == prompt]
    return prompt, matches


def find_matching_entry(entries_path: Path, metadata_path: Path) -> tuple[dict[str, Any], int]:
    prompt, matches = find_matching_entries(entries_path, metadata_path)

    if not matches:
        metadata_payload = load_json(metadata_path)
        task_payload = metadata_payload.get("task")
        sample_index = metadata_payload.get("sample_index")
        if sample_index is None and isinstance(task_payload, dict):
            sample_index = task_payload.get("sample_index")
        if isinstance(sample_index, int) and sample_index >= 0:
            entries, _ = load_entries_dataset(entries_path)
            if sample_index < len(entries):
                return entries[sample_index], 0

        raise ValueError(
            "未在 all_entries_merged_final.json 中通过 metadata prompt 或 sample_index 找到对应条目。"
        )

    if len(matches) > 1:
        raise ValueError(
            "all_entries_merged_final.json 中存在多个 prompt 对应条目："
            f"{prompt[:160]}..."
        )

    return matches[0], len(matches)


def get_entry_qa_pairs(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    for field_name in ("vlm_qa_pairs", "original_qa_pairs"):
        value = entry.get(field_name)
        if isinstance(value, list) and value:
            return value, field_name
    return [], None


def get_entry_autorubric(entry: dict[str, Any]) -> dict[str, Any]:
    value = entry.get("autorubric")
    return value if isinstance(value, dict) else {}


def resolve_entries_path_for_mode(entries_path: Path, question_mode: str) -> Path:
    resolved = entries_path.resolve()
    if (
        question_mode == QUESTION_MODE_AUTORUBRIC
        and resolved == DEFAULT_ENTRIES_PATH.resolve()
        and DEFAULT_AUTORUBRIC_DATASET_DIR.exists()
    ):
        return DEFAULT_AUTORUBRIC_DATASET_DIR.resolve()
    return resolved


def get_metadata_value(metadata_payload: dict[str, Any], key: str) -> Any:
    value = metadata_payload.get(key)
    if value is not None:
        return value

    task_payload = metadata_payload.get("task")
    if isinstance(task_payload, dict):
        return task_payload.get(key)

    return None


def infer_sample_index_from_video_name(video_path: Path) -> int | None:
    match = re.search(r"(\d{3,4})(?!.*\d)", video_path.stem)
    if match is None:
        return None
    sample_index = int(match.group(1))
    return sample_index if sample_index > 0 else None


def find_entry_by_sample_index(
    entries_path: Path,
    sample_index: int,
) -> tuple[dict[str, Any], int]:
    entries, _ = load_entries_dataset(entries_path)
    zero_based_index = sample_index - 1
    if 0 <= zero_based_index < len(entries):
        return entries[zero_based_index], 0
    raise ValueError(
        f"sample_index {sample_index} 超出 all_entries_merged_final.json 范围（共 {len(entries)} 条）。"
    )


def resolve_entry_for_video(
    *,
    entries_path: Path,
    metadata_root: Path,
    video_path: Path,
) -> tuple[dict[str, Any], int, Path | None, dict[str, Any], int | None, str]:
    metadata_path = find_matching_metadata(metadata_root, video_path.stem)
    metadata_payload: dict[str, Any] = {}
    metadata_error: str | None = None

    if metadata_path is not None:
        metadata_payload = load_json(metadata_path)
        try:
            matched_entry, matched_prompt_occurrences = find_matching_entry(
                entries_path,
                metadata_path,
            )
            sample_index = get_metadata_value(metadata_payload, "sample_index")
            if isinstance(sample_index, int) and sample_index >= 0:
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
        matched_entry, matched_prompt_occurrences = find_entry_by_sample_index(
            entries_path,
            sample_index,
        )
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
            f"{metadata_error}；并且无法从视频文件名 {video_path.name} 推断 sample_index。"
        )
    raise ValueError(
        f"未找到对应 metadata，且无法从视频文件名 {video_path.name} 推断 sample_index。"
    )


def normalize_answer(value: str | None) -> str:
    if not value:
        return "UNKNOWN"

    cleaned = value.strip().lower()
    cleaned = re.sub(r"[^a-z]", "", cleaned)
    if cleaned in {"yes", "y", "true"}:
        return "Yes"
    if cleaned in {"no", "n", "false"}:
        return "No"
    return "UNKNOWN"


def build_eval_prompt(qa_pairs: list[dict[str, Any]]) -> str:
    questions_blob = build_questions_blob(qa_pairs)

    return (
        "You are a strict video QA auditor for text-to-video evaluation.\n"
        "Your job is to inspect the video carefully and judge whether each question is supported by visible evidence in the video itself.\n"
        "You are not a creative writer, and you must not assume missing events happened just because they are plausible.\n"
        "\n"
        "Evaluation procedure:\n"
        "1. Identify the main entities, scene layout, and visible attributes.\n"
        "2. Track each action over time across the whole clip, not just one frame.\n"
        "3. Verify state changes only if the before/after change is visually supported.\n"
        "4. Verify causal questions only if both the triggering event and the resulting event are visibly shown in the correct order.\n"
        "5. If an event is partially visible, ambiguous, implied, off-screen, or too uncertain, answer No.\n"
        "6. Use the dependency/type annotations as structure hints for careful checking, but answer from video evidence rather than copying the dependency.\n"
        "\n"
        "Answer every question using only Yes or No.\n"
        "Be conservative: uncertain evidence must be labeled No.\n"
        "Return JSON only, with this exact schema:\n"
        '{"answers":[{"id":"q1","answer":"Yes","reason":"short reason"}]}\n'
        "You must include every question id exactly once and preserve the original ids.\n"
        "Keep each reason under 18 words and mention the visual evidence briefly.\n"
        "Questions:\n"
        f"{questions_blob}"
    )


def build_dependency_round_prompt(
    *,
    round_index: int,
    total_rounds: int,
    current_round_qas: list[dict[str, Any]],
    locked_answers: dict[str, dict[str, str]],
) -> str:
    previous_lines = [
        f"{qa_id} = {payload.get('answer', 'UNKNOWN')} | reason={payload.get('reason', '')}"
        for qa_id, payload in sorted(locked_answers.items())
    ]
    previous_blob = "\n".join(previous_lines) if previous_lines else "None"
    questions_blob = build_questions_blob(current_round_qas)

    return (
        "You are a strict video QA auditor for text-to-video evaluation.\n"
        "Inspect the video itself carefully and answer only the current round questions.\n"
        "Do not revise previous rounds. Previous rounds are locked context only.\n"
        "If evidence is uncertain, partially visible, implied, off-screen, or ambiguous, answer No.\n"
        "\n"
        f"This is round {round_index} of {total_rounds}.\n"
        "Previous locked answers:\n"
        f"{previous_blob}\n"
        "\n"
        "Return JSON only, with this exact schema:\n"
        '{"answers":[{"id":"q1","answer":"Yes","reason":"short reason"}]}\n'
        "You must include every current-round question id exactly once.\n"
        "Use only Yes or No.\n"
        "Keep each reason under 18 words and mention brief visual evidence.\n"
        "Current round questions:\n"
        f"{questions_blob}"
    )


def build_autorubric_prompt(autorubric: dict[str, Any]) -> str:
    prompt_text = str(autorubric.get("prompt", "")).strip()
    dimensions = autorubric.get("dimensions", {})

    dimension_blocks: list[str] = []
    for short_name, dimension_key in AUTORUBRIC_BASE_DIMENSIONS:
        dimension_payload = dimensions.get(dimension_key, {})
        if dimension_key == "purity" and not dimension_payload:
            dimension_payload = dimensions.get("visual_purity", {})
        if not isinstance(dimension_payload, dict):
            continue
        focus_points = dimension_payload.get("focus_points", [])
        focus_blob = (
            "\n".join(f"- {item}" for item in focus_points if isinstance(item, str))
            if isinstance(focus_points, list) and focus_points
            else "- Follow the dimension goal and score anchors."
        )
        anchors = dimension_payload.get("score_anchors", {})
        anchor_blob = (
            "\n".join(
                f"- {score}: {text}"
                for score, text in sorted(anchors.items(), key=lambda item: item[0])
            )
            if isinstance(anchors, dict) and anchors
            else "- 1 is worst, 5 is best."
        )
        dimension_blocks.append(
            "\n".join(
                [
                    f"{short_name} ({dimension_key})",
                    f"Goal: {dimension_payload.get('dimension_goal', '')}",
                    f"Scoring prompt: {dimension_payload.get('vlm_scoring_prompt', '')}",
                    "Focus points:",
                    focus_blob,
                    "Score anchors:",
                    anchor_blob,
                ]
            )
        )

    dimension_blob = "\n\n".join(dimension_blocks)
    return (
        "You are a strict text-to-video autorubric judge.\n"
        "Inspect the video itself carefully and score four dimensions from 1 to 5.\n"
        "Use only visible evidence from the video. Be conservative and avoid assuming missing events happened.\n"
        "A score of 1 means very poor fulfillment, and 5 means excellent fulfillment.\n"
        "Return JSON only with this exact schema:\n"
        '{"scores":[{"id":"Cin","score":4,"reason":"short reason"}]}\n'
        "You must include exactly these four ids once each: Cin, Pur, Mot, Phy.\n"
        "Keep each reason under 24 words and mention the visible evidence briefly.\n"
        "\n"
        "Prompt:\n"
        f"{prompt_text}\n"
        "\n"
        "Rubric dimensions:\n"
        f"{dimension_blob}\n"
    )


def build_entry_autorubric_prompt(entry: dict[str, Any], autorubric: dict[str, Any]) -> str:
    qa_pairs, _ = get_entry_qa_pairs(entry)
    qa_lines: list[str] = []
    for qa in qa_pairs:
        qa_id = qa.get("id")
        question = qa.get("question")
        dependency = qa.get("dependency")
        if qa_id and question:
            qa_lines.append(f"{qa_id} | dependency={dependency or 'None'} | {question}")
    qa_blob = "\n".join(qa_lines) if qa_lines else "None"

    return (
        f"{build_autorubric_prompt(autorubric).rstrip()}\n"
        "\n"
        "Prompt-specific QA checklist from the dataset:\n"
        f"{qa_blob}\n"
    )


def build_dependency_rounds(qa_pairs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    qa_by_id = {
        qa["id"]: qa
        for qa in qa_pairs
        if isinstance(qa.get("id"), str)
    }
    visiting: set[str] = set()
    cache: dict[str, int] = {}

    def compute_level(qa_id: str) -> int:
        if qa_id in cache:
            return cache[qa_id]
        if qa_id in visiting:
            raise ValueError(f"Dependency cycle detected at {qa_id}")

        qa = qa_by_id.get(qa_id)
        if qa is None:
            return 0

        visiting.add(qa_id)
        dependency_ids = extract_dependency_ids(qa.get("dependency"))
        if not dependency_ids:
            level = 0
        else:
            level = max(compute_level(dep_id) for dep_id in dependency_ids) + 1
        visiting.remove(qa_id)
        cache[qa_id] = level
        return level

    grouped: dict[int, list[dict[str, Any]]] = {}
    for qa in qa_pairs:
        qa_id = qa.get("id")
        if not isinstance(qa_id, str):
            continue
        level = compute_level(qa_id)
        grouped.setdefault(level, []).append(qa)

    return [grouped[level] for level in sorted(grouped)]


def extract_first_text(response_payload: dict[str, Any]) -> str | None:
    for candidate in response_payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                return text
    return None


def extract_autorubric_payload_from_text(raw_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    scores: list[dict[str, Any]] = []

    for short_name, _ in AUTORUBRIC_DIMENSIONS:
        display_name = AUTORUBRIC_DISPLAY_NAMES.get(short_name, short_name)
        score_value: int | None = None
        reason_value = ""

        aliases = [short_name, display_name]
        alias_patterns = [re.escape(alias) for alias in aliases]
        combined_alias_pattern = "|".join(alias_patterns)

        for line in lines:
            normalized_line = re.sub(r"[*_`]", "", line)
            if not re.search(combined_alias_pattern, normalized_line, flags=re.IGNORECASE):
                continue
            digit_match = re.search(
                r"(?:score(?:\s+of)?|scoring(?:\s+a)?|giving it|give it|i(?:'m| am) giving it|rated?|is|:\s*)[^0-9]{0,24}\b([1-5])\b",
                normalized_line,
                flags=re.IGNORECASE,
            )
            if digit_match:
                score_value = int(digit_match.group(1))
                reason_value = normalized_line
                break
            loose_digit_match = re.search(r"\b([1-5])\b", normalized_line)
            if loose_digit_match:
                score_value = int(loose_digit_match.group(1))
                reason_value = normalized_line
                break

        if score_value is None:
            patterns = [
                rf"(?:{combined_alias_pattern}).{{0,220}}?\b([1-5])\b",
                rf"\((?:{combined_alias_pattern})\).{{0,220}}?\b([1-5])\b",
            ]
            for pattern in patterns:
                match = re.search(pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
                if match:
                    score_value = int(match.group(1))
                    snippet_start = max(0, match.start() - 40)
                    snippet_end = min(len(raw_text), match.end() + 120)
                    reason_value = " ".join(raw_text[snippet_start:snippet_end].split())
                    break

        if score_value is None:
            raise ValueError(f"Could not extract autorubric score for {short_name}")

        scores.append(
            {
                "id": short_name,
                "score": score_value,
                "reason": reason_value[:240].strip(),
            }
        )

    summary = " ".join(lines[:2])[:240].strip()
    return {
        "scores": scores,
        "summary": summary,
    }


def parse_autorubric_model_payload(model_text: str) -> dict[str, Any]:
    try:
        return extract_json_text(model_text)
    except Exception:
        return extract_autorubric_payload_from_text(model_text)


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


def index_model_answers(model_payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    answers = model_payload.get("answers", [])
    indexed: dict[str, dict[str, str]] = {}
    for item in answers:
        answer_id = item.get("id")
        if isinstance(answer_id, str):
            indexed[answer_id] = {
                "answer": normalize_answer(item.get("answer")),
                "reason": str(item.get("reason", "")).strip(),
            }
    return indexed


def normalize_rubric_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 5 else None
    if isinstance(value, float) and value.is_integer():
        integer_value = int(value)
        return integer_value if 1 <= integer_value <= 5 else None
    if isinstance(value, str):
        match = re.search(r"[1-5]", value)
        if match:
            return int(match.group(0))
    return None


def compute_rub_score(
    rubric_scores: dict[str, dict[str, Any]],
) -> tuple[float | None, str]:
    base_scores: list[float] = []
    for short_name, _ in AUTORUBRIC_BASE_DIMENSIONS:
        score_value = rubric_scores.get(short_name, {}).get("score")
        if isinstance(score_value, (int, float)) and not isinstance(score_value, bool):
            base_scores.append(float(score_value))

    if not base_scores:
        return None, ""

    return round(sum(base_scores) / len(base_scores), 2), (
        "Computed as average of Cin, Pur, Mot, and Phy."
    )


def normalize_rubric_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = re.sub(r"[^a-z]", "", value.lower())
    alias_map = {
        "cin": "Cin",
        "cinematography": "Cin",
        "pur": "Pur",
        "purity": "Pur",
        "mot": "Mot",
        "motion": "Mot",
        "motionsmoothness": "Mot",
        "motionsmoothnessscore": "Mot",
        "phy": "Phy",
        "physics": "Phy",
        "physicsadherence": "Phy",
        "physicaladherence": "Phy",
    }
    return alias_map.get(cleaned)


def index_rubric_scores(model_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    scores = model_payload.get("scores", [])

    if isinstance(scores, dict):
        for score_id, payload in scores.items():
            normalized_id = normalize_rubric_id(score_id)
            if normalized_id is None or not isinstance(payload, dict):
                continue
            indexed[normalized_id] = {
                "score": normalize_rubric_score(payload.get("score")),
                "reason": str(payload.get("reason", "")).strip(),
            }
        return indexed

    if isinstance(scores, list):
        for item in scores:
            if not isinstance(item, dict):
                continue
            score_id = normalize_rubric_id(item.get("id"))
            if isinstance(score_id, str):
                indexed[score_id] = {
                    "score": normalize_rubric_score(item.get("score")),
                    "reason": str(item.get("reason", "")).strip(),
                }
    return indexed


def validate_rubric_scores(rubric_scores: dict[str, dict[str, Any]]) -> None:
    missing_ids = [
        short_name
        for short_name, _ in AUTORUBRIC_DIMENSIONS
        if not isinstance(rubric_scores.get(short_name, {}).get("score"), (int, float))
    ]
    if missing_ids:
        raise ValueError(
            "Autorubric response missing valid scores for: "
            + ", ".join(missing_ids)
        )


def find_missing_answer_ids(
    qa_pairs: list[dict[str, Any]],
    model_answers: dict[str, dict[str, str]],
) -> list[str]:
    expected_ids = [qa.get("id") for qa in qa_pairs if isinstance(qa.get("id"), str)]
    return [qa_id for qa_id in expected_ids if qa_id not in model_answers]


def get_output_suffix(question_mode: str) -> str:
    if question_mode == QUESTION_MODE_AUTORUBRIC:
        return "_autorubric_eval.json"
    if question_mode == QUESTION_MODE_DEPENDENCY_ROUNDS:
        return "_qa_eval_dependency_rounds.json"
    return "_qa_eval.json"


def build_output_path(
    video_path: Path,
    custom_output: Path | None,
    question_mode: str = QUESTION_MODE_ALL_AT_ONCE,
    output_tag: str | None = None,
) -> Path:
    if custom_output is not None:
        return custom_output.resolve()
    suffix = get_output_suffix(question_mode)
    tag = f"_{output_tag}" if output_tag else ""
    return video_path.with_name(f"{video_path.stem}{tag}{suffix}")


def build_questions_blob(qa_pairs: list[dict[str, Any]]) -> str:
    return "\n".join(
        f'{qa["id"]} | type={qa.get("type")} | dependency={qa.get("dependency")} | question={qa["question"]}'
        for qa in qa_pairs
        if qa.get("id") and qa.get("question")
    )


def extract_dependency_ids(dependency: str | None) -> list[str]:
    if not dependency or dependency == "None":
        return []
    return re.findall(r"q\d+", dependency)


def evaluate_dependency_expression(
    dependency: str | None,
    is_question_correct: Any,
) -> tuple[bool, list[str]]:
    if not dependency or dependency == "None":
        return True, []

    tokens = re.findall(r"q\d+|AND|OR|\(|\)", dependency, flags=re.IGNORECASE)
    normalized_tokens = [token.upper() if token.upper() in {"AND", "OR", "(", ")"} else token for token in tokens]
    position = 0

    def parse_or() -> bool:
        nonlocal position
        value = parse_and()
        while position < len(normalized_tokens) and normalized_tokens[position] == "OR":
            position += 1
            rhs = parse_and()
            value = value or rhs
        return value

    def parse_and() -> bool:
        nonlocal position
        value = parse_primary()
        while position < len(normalized_tokens) and normalized_tokens[position] == "AND":
            position += 1
            rhs = parse_primary()
            value = value and rhs
        return value

    def parse_primary() -> bool:
        nonlocal position
        if position >= len(normalized_tokens):
            raise ValueError(f"依赖表达式不完整：{dependency}")

        token = normalized_tokens[position]
        if token == "(":
            position += 1
            value = parse_or()
            if position >= len(normalized_tokens) or normalized_tokens[position] != ")":
                raise ValueError(f"依赖表达式括号不匹配：{dependency}")
            position += 1
            return value
        if re.fullmatch(r"q\d+", token, flags=re.IGNORECASE):
            position += 1
            return bool(is_question_correct(token))
        raise ValueError(f"无法解析 dependency token: {token}")

    passed = parse_or()
    if position != len(normalized_tokens):
        raise ValueError(f"依赖表达式存在多余 token：{dependency}")

    failed_ids = [
        question_id
        for question_id in extract_dependency_ids(dependency)
        if not is_question_correct(question_id)
    ]
    return passed, failed_ids


def evaluate_predictions(
    qa_pairs: list[dict[str, Any]],
    model_answers: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], int, int, int]:
    evaluated: list[dict[str, Any]] = []
    qa_by_id: dict[str, dict[str, Any]] = {}

    for qa in qa_pairs:
        qa_id = qa.get("id")
        if not isinstance(qa_id, str):
            continue

        predicted = model_answers.get(qa_id, {})
        predicted_answer = predicted.get("answer", "UNKNOWN")
        expected_answer = normalize_answer(qa.get("expected_answer"))
        answer_match = predicted_answer == expected_answer

        row = {
            "id": qa_id,
            "question": qa.get("question"),
            "type": qa.get("type"),
            "dependency": qa.get("dependency"),
            "dependency_ids": extract_dependency_ids(qa.get("dependency")),
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "answer_match": answer_match,
            "dependency_passed": True,
            "dependency_failed_ids": [],
            "correct": False,
            "reason": predicted.get("reason", ""),
        }
        evaluated.append(row)
        qa_by_id[qa_id] = row

    @lru_cache(maxsize=None)
    def is_question_correct(question_id: str) -> bool:
        if question_id in visiting:
            raise ValueError(f"QA dependency cycle detected at {question_id}")
        row = qa_by_id.get(question_id)
        if row is None:
            return False

        visiting.add(question_id)
        try:
            dependency_passed, failed_ids = evaluate_dependency_expression(
                row["dependency"],
                is_question_correct,
            )
            row["dependency_passed"] = dependency_passed
            row["dependency_failed_ids"] = failed_ids
            row["correct"] = bool(row["answer_match"] and dependency_passed)
            return row["correct"]
        finally:
            visiting.remove(question_id)

    visiting: set[str] = set()
    for qa_id in qa_by_id:
        is_question_correct(qa_id)

    raw_correct_count = sum(1 for row in evaluated if row["answer_match"])
    correct_count = sum(1 for row in evaluated if row["correct"])
    dependency_blocked_count = sum(
        1 for row in evaluated if row["answer_match"] and not row["dependency_passed"]
    )
    return evaluated, raw_correct_count, correct_count, dependency_blocked_count


def evaluate_autorubric_scores(
    autorubric: dict[str, Any],
    rubric_scores: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    dimensions = autorubric.get("dimensions", {})
    overall_assessment = autorubric.get("overall_assessment", {})
    rows: list[dict[str, Any]] = []
    derived_rub_score, derived_rub_reason = compute_rub_score(rubric_scores)

    for short_name, dimension_key in AUTORUBRIC_DIMENSIONS:
        predicted = rubric_scores.get(short_name, {})
        score_value = predicted.get("score")
        reason_value = predicted.get("reason", "")
        if dimension_key == "overall_rubric_adherence":
            goal = (
                str(overall_assessment.get("emotional_narrative_alignment", "")).strip()
                if isinstance(overall_assessment, dict)
                else ""
            )
            if not goal:
                goal = "Evaluate the video's overall adherence to the full autorubric."
            score_value = derived_rub_score
            if derived_rub_reason:
                reason_value = derived_rub_reason
        else:
            dimension_payload = dimensions.get(dimension_key, {})
            goal = (
                dimension_payload.get("dimension_goal", "")
                if isinstance(dimension_payload, dict)
                else ""
            )
        rows.append(
            {
                "id": short_name,
                "dimension_key": dimension_key,
                "dimension_goal": goal,
                "score": score_value,
                "reason": reason_value,
            }
        )

    if not isinstance(derived_rub_score, (int, float)):
        raise ValueError("Autorubric response is missing valid Cin/Pur/Mot/Phy scores.")
    average_score = float(derived_rub_score)
    return rows, average_score


def main() -> int:
    args = parse_args()
    return run_evaluation(args)


def request_and_parse_model_answers(
    *,
    session: Any,
    args: argparse.Namespace,
    request_payload: dict[str, Any],
    response_parser: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any], str, dict[str, Any]]:
    max_attempts = max(1, int(getattr(args, "format_retries", DEFAULT_FORMAT_RETRIES)) + 1)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
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
            last_error = ValueError("接口返回的不是合法 JSON。")
            if attempt < max_attempts:
                print(
                    f"警告：第 {attempt}/{max_attempts} 次返回不是合法 JSON，正在重试格式请求。",
                    file=sys.stderr,
                )
                continue
            raise last_error

        if isinstance(response_payload.get("error"), dict):
            error_payload = response_payload["error"]
            message = error_payload.get("message") or response.text[:1000]
            code = error_payload.get("code") or response.status_code
            last_error = ValueError(f"API error {code}: {message}")
            if attempt < max_attempts:
                print(
                    f"警告：第 {attempt}/{max_attempts} 次 API 返回错误：{last_error}；正在重试。",
                    file=sys.stderr,
                )
                continue
            raise last_error

        model_text = extract_first_text(response_payload)
        if not model_text:
            last_error = ValueError("响应中未找到模型文本输出。")
            if attempt < max_attempts:
                print(
                    f"警告：第 {attempt}/{max_attempts} 次未提取到文本输出，正在重试格式请求。",
                    file=sys.stderr,
                )
                continue
            raise last_error

        try:
            parser = response_parser or extract_json_text
            model_answer_payload = parser(model_text)
            return response, response_payload, model_text, model_answer_payload
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                print(
                    f"警告：第 {attempt}/{max_attempts} 次模型回答 JSON 解析失败：{exc}；正在重试格式请求。",
                    file=sys.stderr,
                )
                continue
            raise ValueError(f"无法解析模型回答 JSON：{exc}\n{model_text}") from exc

    raise last_error or ValueError("无法解析模型回答 JSON。")


def execute_all_at_once_evaluation(
    *,
    session: Any,
    args: argparse.Namespace,
    qa_pairs: list[dict[str, Any]],
    mime_type: str,
    video_base64: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    eval_prompt = build_eval_prompt(qa_pairs)
    request_payload = build_request_payload(
        prompt=eval_prompt,
        model=args.model,
        mime_type=mime_type,
        video_base64=video_base64,
    )
    response, response_payload, model_text, model_answer_payload = request_and_parse_model_answers(
        session=session,
        args=args,
        request_payload=request_payload,
        response_parser=parse_autorubric_model_payload,
    )
    model_answers = index_model_answers(model_answer_payload)

    return model_answers, {
        "question_mode": QUESTION_MODE_ALL_AT_ONCE,
        "eval_prompt": eval_prompt,
        "request_payload": request_payload,
        "response_payload": response_payload,
        "response_id": response_payload.get("responseId"),
        "response_ids": [response_payload.get("responseId")],
        "model_version": response_payload.get("modelVersion"),
        "model_versions": [response_payload.get("modelVersion")],
        "http_status": response.status_code,
        "http_statuses": [response.status_code],
        "raw_model_text": model_text,
        "round_count": 1,
        "rounds": [
            {
                "round_index": 1,
                "question_ids": [
                    qa.get("id")
                    for qa in qa_pairs
                    if isinstance(qa.get("id"), str)
                ],
                "response_id": response_payload.get("responseId"),
                "model_version": response_payload.get("modelVersion"),
                "http_status": response.status_code,
                "eval_prompt": eval_prompt,
                "raw_model_text": model_text,
                "missing_answer_ids": find_missing_answer_ids(qa_pairs, model_answers),
                "locked_answer_count_before_round": 0,
            }
        ],
    }


def execute_dependency_round_evaluation(
    *,
    session: Any,
    args: argparse.Namespace,
    qa_pairs: list[dict[str, Any]],
    mime_type: str,
    video_base64: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    rounds = build_dependency_rounds(qa_pairs)
    aggregated_answers: dict[str, dict[str, str]] = {}
    round_payloads: list[dict[str, Any]] = []
    response_ids: list[str] = []
    model_versions: list[str] = []
    http_statuses: list[int] = []

    for round_index, round_qas in enumerate(rounds, start=1):
        locked_before_round = len(aggregated_answers)
        eval_prompt = build_dependency_round_prompt(
            round_index=round_index,
            total_rounds=len(rounds),
            current_round_qas=round_qas,
            locked_answers=aggregated_answers,
        )
        request_payload = build_request_payload(
            prompt=eval_prompt,
            model=args.model,
            mime_type=mime_type,
            video_base64=video_base64,
        )
        response, response_payload, model_text, model_answer_payload = request_and_parse_model_answers(
            session=session,
            args=args,
            request_payload=request_payload,
        )
        round_answers = index_model_answers(model_answer_payload)
        missing_answer_ids = find_missing_answer_ids(round_qas, round_answers)

        for qa in round_qas:
            qa_id = qa.get("id")
            if isinstance(qa_id, str):
                aggregated_answers[qa_id] = round_answers.get(
                    qa_id,
                    {"answer": "UNKNOWN", "reason": ""},
                )

        response_id = response_payload.get("responseId")
        model_version = response_payload.get("modelVersion")
        response_ids.append(response_id)
        model_versions.append(model_version)
        http_statuses.append(response.status_code)
        round_payloads.append(
            {
                "round_index": round_index,
                "question_ids": [
                    qa.get("id")
                    for qa in round_qas
                    if isinstance(qa.get("id"), str)
                ],
                "response_id": response_id,
                "model_version": model_version,
                "http_status": response.status_code,
                "eval_prompt": eval_prompt,
                "raw_model_text": model_text,
                "missing_answer_ids": missing_answer_ids,
                "locked_answer_count_before_round": locked_before_round,
            }
        )

    return aggregated_answers, {
        "question_mode": QUESTION_MODE_DEPENDENCY_ROUNDS,
        "eval_prompt": None,
        "request_payload": None,
        "response_payload": None,
        "response_id": response_ids[-1] if len(response_ids) == 1 else "multiple",
        "response_ids": response_ids,
        "model_version": model_versions[-1] if model_versions else None,
        "model_versions": model_versions,
        "http_status": http_statuses[-1] if http_statuses else None,
        "http_statuses": http_statuses,
        "raw_model_text": None,
        "round_count": len(rounds),
        "rounds": round_payloads,
    }


def execute_autorubric_evaluation(
    *,
    session: Any,
    args: argparse.Namespace,
    entry: dict[str, Any],
    autorubric: dict[str, Any],
    mime_type: str,
    video_base64: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    eval_prompt = build_entry_autorubric_prompt(entry, autorubric)
    request_payload = build_request_payload(
        prompt=eval_prompt,
        model=args.model,
        mime_type=mime_type,
        video_base64=video_base64,
    )
    response, response_payload, model_text, model_answer_payload = request_and_parse_model_answers(
        session=session,
        args=args,
        request_payload=request_payload,
        response_parser=parse_autorubric_model_payload,
    )
    rubric_scores = index_rubric_scores(model_answer_payload)
    validate_rubric_scores(rubric_scores)
    return rubric_scores, {
        "question_mode": QUESTION_MODE_AUTORUBRIC,
        "eval_prompt": eval_prompt,
        "request_payload": request_payload,
        "response_payload": response_payload,
        "response_id": response_payload.get("responseId"),
        "response_ids": [response_payload.get("responseId")],
        "model_version": response_payload.get("modelVersion"),
        "model_versions": [response_payload.get("modelVersion")],
        "http_status": response.status_code,
        "http_statuses": [response.status_code],
        "raw_model_text": model_text,
        "round_count": 1,
        "rounds": [
            {
                "round_index": 1,
                "dimension_ids": [short_name for short_name, _ in AUTORUBRIC_DIMENSIONS],
                "response_id": response_payload.get("responseId"),
                "model_version": response_payload.get("modelVersion"),
                "http_status": response.status_code,
                "eval_prompt": eval_prompt,
                "raw_model_text": model_text,
            }
        ],
    }


def run_evaluation(args: argparse.Namespace) -> int:
    video_path = args.video.resolve()
    output_path = build_output_path(video_path, args.output, args.question_mode, args.output_tag)

    if not video_path.exists():
        print(f"错误：找不到视频文件：{video_path}", file=sys.stderr)
        return 1

    metadata_root = args.metadata_root.resolve()
    entries_path = resolve_entries_path_for_mode(args.entries, args.question_mode)

    try:
        (
            matched_entry,
            matched_prompt_occurrences,
            metadata_path,
            metadata_payload,
            resolved_sample_index,
            entry_match_source,
        ) = resolve_entry_for_video(
            entries_path=entries_path,
            metadata_root=metadata_root,
            video_path=video_path,
        )
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    qa_pairs, qa_source_field = get_entry_qa_pairs(matched_entry)
    autorubric = get_entry_autorubric(matched_entry)
    if args.question_mode == QUESTION_MODE_AUTORUBRIC:
        if not autorubric:
            print("Error: matched entry has no autorubric.", file=sys.stderr)
            return 1
    elif not qa_pairs:
        print("Error: matched entry has neither vlm_qa_pairs nor original_qa_pairs.", file=sys.stderr)
        return 1

    session = build_session(args.api_key)
    if not args.skip_model_check:
        try:
            models = fetch_available_models(session, args.base_url, args.timeout)
            print_model_check(args.model, models)
        except Exception as exc:
            print(f"警告：获取模型列表失败：{exc}", file=sys.stderr)

    try:
        video_base64 = encode_file_to_base64(video_path)
    except OSError as exc:
        print(f"错误：读取视频失败：{exc}", file=sys.stderr)
        return 1

    mime_type = guess_mime_type(video_path)

    if args.verbose:
        print(f"Video: {video_path}")
        print(f"Metadata: {metadata_path if metadata_path is not None else 'None'}")
        print(f"Entry match source: {entry_match_source}")
        if args.question_mode == QUESTION_MODE_AUTORUBRIC:
            print("Autorubric dimensions: 4")
        else:
            print(f"Questions: {len(qa_pairs)}")
        print(f"Question mode: {args.question_mode}")

    try:
        if args.question_mode == QUESTION_MODE_AUTORUBRIC:
            rubric_scores, execution = execute_autorubric_evaluation(
                session=session,
                args=args,
                entry=matched_entry,
                autorubric=autorubric,
                mime_type=mime_type,
                video_base64=video_base64,
            )
        elif args.question_mode == QUESTION_MODE_DEPENDENCY_ROUNDS:
            model_answers, execution = execute_dependency_round_evaluation(
                session=session,
                args=args,
                qa_pairs=qa_pairs,
                mime_type=mime_type,
                video_base64=video_base64,
            )
        else:
            model_answers, execution = execute_all_at_once_evaluation(
                session=session,
                args=args,
                qa_pairs=qa_pairs,
                mime_type=mime_type,
                video_base64=video_base64,
            )
    except Exception as exc:
        print(f"错误：评测请求失败：{exc}", file=sys.stderr)
        return 1

    if args.question_mode == QUESTION_MODE_AUTORUBRIC:
        evaluated_rows, average_score = evaluate_autorubric_scores(autorubric, rubric_scores)
        result_payload = {
            "success": True,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "video_file": str(video_path),
            "output_file": str(output_path),
            "metadata_file": str(metadata_path) if metadata_path is not None else None,
            "task_id": get_metadata_value(metadata_payload, "task_id"),
            "sample_index": resolved_sample_index,
            "entry_match_source": entry_match_source,
            "model": args.model,
            "question_mode": args.question_mode,
            "model_version": execution.get("model_version"),
            "model_versions": execution.get("model_versions"),
            "response_id": execution.get("response_id"),
            "response_ids": execution.get("response_ids"),
            "http_status": execution.get("http_status"),
            "http_statuses": execution.get("http_statuses"),
            "round_count": execution.get("round_count"),
            "dimension_count": len(evaluated_rows),
            "average_score": average_score,
            "subjective_score": round(normalize_rating(average_score), 6),
            "average_score_percent": normalized_percent(normalize_rating(average_score)),
            "prompt": matched_entry.get("prompt"),
            "matched_macro_domain": matched_entry.get("domain_info", {}).get("macro_domain"),
            "matched_micro_domain": matched_entry.get("domain_info", {}).get("micro_domain"),
            "eval_prompt": execution.get("eval_prompt"),
            "raw_model_text": execution.get("raw_model_text"),
            "rounds": execution.get("rounds"),
            "results": evaluated_rows,
        }
        dump_json(output_path, result_payload)
        print(f"评测结果已写入：{output_path}")
        print(f"Autorubric average: {average_score:.2f}/5")
        return 0

    (
        evaluated_rows,
        raw_correct_count,
        correct_count,
        dependency_blocked_count,
    ) = evaluate_predictions(qa_pairs, model_answers)
    total_count = len(evaluated_rows)
    raw_accuracy = raw_correct_count / total_count if total_count else 0.0
    accuracy = correct_count / total_count if total_count else 0.0

    result_payload = {
        "success": True,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "video_file": str(video_path),
        "output_file": str(output_path),
        "metadata_file": str(metadata_path) if metadata_path is not None else None,
        "task_id": get_metadata_value(metadata_payload, "task_id"),
        "sample_index": resolved_sample_index,
        "entry_match_source": entry_match_source,
        "model": args.model,
        "question_mode": args.question_mode,
        "model_version": execution.get("model_version"),
        "model_versions": execution.get("model_versions"),
        "response_id": execution.get("response_id"),
        "response_ids": execution.get("response_ids"),
        "http_status": execution.get("http_status"),
        "http_statuses": execution.get("http_statuses"),
        "round_count": execution.get("round_count"),
        "question_count": total_count,
        "matched_entry_question_count": len(qa_pairs),
        "matched_entry_question_source": qa_source_field,
        "matched_prompt_occurrences": matched_prompt_occurrences,
        "raw_correct_count": raw_correct_count,
        "raw_accuracy": raw_accuracy,
        "raw_accuracy_percent": round(raw_accuracy * 100, 2),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "accuracy_percent": round(accuracy * 100, 2),
        "dependency_blocked_count": dependency_blocked_count,
        "prompt": matched_entry.get("prompt"),
        "matched_macro_domain": matched_entry.get("domain_info", {}).get("macro_domain"),
        "matched_micro_domain": matched_entry.get("domain_info", {}).get("micro_domain"),
        "eval_prompt": execution.get("eval_prompt"),
        "raw_model_text": execution.get("raw_model_text"),
        "rounds": execution.get("rounds"),
        "results": evaluated_rows,
    }
    dump_json(output_path, result_payload)

    print(f"评测结果已写入：{output_path}")
    print(
        f"正确率：raw {raw_correct_count}/{total_count} = {raw_accuracy:.2%}, "
        f"final {correct_count}/{total_count} = {accuracy:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
