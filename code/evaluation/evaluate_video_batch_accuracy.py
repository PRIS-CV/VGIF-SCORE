from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scoring import (
    extract_rubric_ratings_from_payload,
    mean_rubric_rating,
    mean_rubric_score,
    normalize_rating,
)

from evaluate_video_qa_accuracy import (
    AUTORUBRIC_DIMENSIONS,
    AUTORUBRIC_BASE_DIMENSIONS,
    DEFAULT_ENTRIES_PATH,
    QUESTION_MODE_ALL_AT_ONCE,
    QUESTION_MODE_AUTORUBRIC,
    QUESTION_MODE_DEPENDENCY_ROUNDS,
    build_output_path,
    build_session,
    dump_json,
    encode_file_to_base64,
    evaluate_autorubric_scores,
    evaluate_predictions,
    execute_all_at_once_evaluation,
    execute_autorubric_evaluation,
    execute_dependency_round_evaluation,
    fetch_available_models,
    find_matching_entry,
    find_matching_metadata,
    get_entry_autorubric,
    get_metadata_value,
    get_entry_qa_pairs,
    infer_sample_index_from_video_name,
    guess_mime_type,
    load_entries_dataset,
    load_json,
    print_model_check,
    resolve_entries_path_for_mode,
)
from test_gemini_video import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    REPO_DIR
    / "kling_t2v"
    / "outputs"
    / "kling_v3_720p_5s"
    / "20260419_172612_all-entries-223"
)
DEFAULT_FORMAT_RETRIES = 2


@dataclass
class Counter:
    correct: int = 0
    total: int = 0

    def add(self, correct: int, total: int) -> None:
        self.correct += correct
        self.total += total

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-evaluate a video directory against QA pairs and aggregate overall and "
            "per-dimension accuracy."
        )
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Run directory that contains videos/ and optional metadata JSONs.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Direct video directory. If set, videos are loaded from here instead of run-dir/videos.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help="Optional metadata root for matching by video stem when JSON metadata files exist.",
    )
    parser.add_argument(
        "--progress-jsonl",
        type=Path,
        default=None,
        help="Optional progress_success.jsonl used to map video names to prompts directly.",
    )
    parser.add_argument("--entries", type=Path, default=DEFAULT_ENTRIES_PATH)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--format-retries", type=int, default=DEFAULT_FORMAT_RETRIES)
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--question-mode",
        choices=[
            QUESTION_MODE_ALL_AT_ONCE,
            QUESTION_MODE_DEPENDENCY_ROUNDS,
            QUESTION_MODE_AUTORUBRIC,
        ],
        default=QUESTION_MODE_ALL_AT_ONCE,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based start index (inclusive) after sorting all mp4 files.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="1-based end index (inclusive) after sorting all mp4 files.",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--eval-attempts",
        type=int,
        default=3,
        help="How many times to retry a video if evaluation raises an exception.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse per-video eval JSON if it already exists.",
    )
    return parser.parse_args()


def to_percent(value: float) -> float:
    return round(value * 100, 2)


def collect_video_files(videos_dir: Path) -> list[Path]:
    return sorted(videos_dir.glob("*.mp4"))


def slice_video_files(
    video_files: list[Path],
    *,
    start_index: int,
    end_index: int | None,
    limit: int | None,
) -> list[Path]:
    if start_index < 1:
        raise ValueError("--start-index must be >= 1")
    if end_index is not None and end_index < start_index:
        raise ValueError("--end-index must be >= --start-index")

    sliced = video_files[start_index - 1 :]
    if end_index is not None:
        sliced = sliced[: end_index - start_index + 1]
    if limit:
        sliced = sliced[:limit]
    return sliced


def resolve_videos_dir(args: argparse.Namespace) -> Path:
    if args.videos_dir is not None:
        return args.videos_dir.resolve()
    return (args.run_dir.resolve() / "videos")


def resolve_metadata_root(args: argparse.Namespace) -> Path:
    if args.metadata_root is not None:
        return args.metadata_root.resolve()
    return args.run_dir.resolve()


def resolve_progress_jsonl(args: argparse.Namespace, videos_dir: Path) -> Path | None:
    if args.progress_jsonl is not None:
        return args.progress_jsonl.resolve()
    candidate = videos_dir / "progress_success.jsonl"
    return candidate if candidate.exists() else None


def build_default_output_path(videos_dir: Path, question_mode: str) -> Path:
    safe_mode = question_mode.replace("-", "_")
    return videos_dir / f"qa_eval_{safe_mode}_summary.json"


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
        raise ValueError("Prompt not found in entries dataset.")
    if len(matches) > 1:
        raise ValueError(f"Prompt is duplicated in entries dataset: {prompt[:160]}...")
    entry_index, entry = matches[0]
    return entry, len(matches), entry_index


def build_eval_args(
    batch_args: argparse.Namespace,
    video_path: Path,
    metadata_root: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        api_key=batch_args.api_key,
        base_url=batch_args.base_url,
        model=batch_args.model,
        video=video_path,
        entries=batch_args.entries,
        metadata_root=metadata_root,
        timeout=batch_args.timeout,
        retries=batch_args.retries,
        format_retries=batch_args.format_retries,
        skip_model_check=True,
        output=None,
        output_tag=batch_args.output_tag,
        verbose=batch_args.verbose,
        question_mode=batch_args.question_mode,
    )


def resolve_video_entry_context(
    *,
    video_path: Path,
    entries_path: Path,
    entries: list[dict[str, Any]],
    metadata_root: Path,
    progress_prompt_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    progress_record = progress_prompt_map.get(video_path.name)
    if progress_record is not None:
        prompt = progress_record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing prompt in progress jsonl for {video_path.name}")
        matched_entry, matched_prompt_occurrences, entry_index = find_entry_by_prompt(entries, prompt)
        progress_index = progress_record.get("index")
        sample_index = progress_index + 1 if isinstance(progress_index, int) else entry_index + 1
        return {
            "metadata_path": None,
            "metadata_payload": {"task": {"task_id": None, "sample_index": sample_index}},
            "matched_entry": matched_entry,
            "matched_prompt_occurrences": matched_prompt_occurrences,
        }

    metadata_path = find_matching_metadata(metadata_root, video_path.stem)
    if metadata_path is None:
        raise FileNotFoundError(f"No metadata found for {video_path.stem}")

    matched_entry, matched_prompt_occurrences = find_matching_entry(entries_path, metadata_path)
    return {
        "metadata_path": metadata_path,
        "metadata_payload": load_json(metadata_path),
        "matched_entry": matched_entry,
        "matched_prompt_occurrences": matched_prompt_occurrences,
    }


def normalize_reused_autorubric_payload(
    *,
    payload: dict[str, Any],
    output_path: Path,
    metadata_path: Path | None,
    metadata_payload: dict[str, Any],
    matched_entry: dict[str, Any],
    matched_prompt_occurrences: int,
) -> dict[str, Any]:
    autorubric = get_entry_autorubric(matched_entry)
    if not autorubric:
        return payload

    normalized = False
    rubric_scores: dict[str, dict[str, Any]] = {}
    legacy_scores = payload.get("scores", {})
    if isinstance(legacy_scores, dict) and legacy_scores:
        rubric_scores = legacy_scores
    elif isinstance(payload.get("results"), list):
        for row in payload["results"]:
            if not isinstance(row, dict):
                continue
            score_id = row.get("id")
            if isinstance(score_id, str):
                rubric_scores[score_id] = {
                    "score": row.get("score"),
                    "reason": str(row.get("reason", "")).strip(),
                }

    if rubric_scores:
        evaluated_rows, computed_average = evaluate_autorubric_scores(autorubric, rubric_scores)
        if payload.get("results") != evaluated_rows:
            payload["results"] = evaluated_rows
            normalized = True
        if payload.get("dimension_count") != len(evaluated_rows):
            payload["dimension_count"] = len(evaluated_rows)
            normalized = True
        if payload.get("average_score") != computed_average:
            payload["average_score"] = computed_average
            normalized = True
        computed_percent = to_percent(normalize_rating(computed_average))
        if payload.get("average_score_percent") != computed_percent:
            payload["average_score_percent"] = computed_percent
            normalized = True

    if payload.get("question_mode") != QUESTION_MODE_AUTORUBRIC:
        payload["question_mode"] = QUESTION_MODE_AUTORUBRIC
        normalized = True

    if payload.get("output_file") != str(output_path):
        payload["output_file"] = str(output_path)
        normalized = True

    metadata_value = str(metadata_path) if metadata_path is not None else None
    if payload.get("metadata_file") != metadata_value:
        payload["metadata_file"] = metadata_value
        normalized = True

    sample_index = get_metadata_value(metadata_payload, "sample_index")
    if payload.get("sample_index") != sample_index:
        payload["sample_index"] = sample_index
        normalized = True

    if payload.get("matched_prompt_occurrences") != matched_prompt_occurrences:
        payload["matched_prompt_occurrences"] = matched_prompt_occurrences
        normalized = True

    prompt = matched_entry.get("prompt")
    if payload.get("prompt") != prompt:
        payload["prompt"] = prompt
        normalized = True

    macro_domain = matched_entry.get("domain_info", {}).get("macro_domain")
    if payload.get("matched_macro_domain") != macro_domain:
        payload["matched_macro_domain"] = macro_domain
        normalized = True

    micro_domain = matched_entry.get("domain_info", {}).get("micro_domain")
    if payload.get("matched_micro_domain") != micro_domain:
        payload["matched_micro_domain"] = micro_domain
        normalized = True

    if normalized:
        dump_json(output_path, payload)

    return payload


def autorubric_payload_has_valid_scores(payload: dict[str, Any]) -> bool:
    results = payload.get("results")
    if not isinstance(results, list):
        return False

    score_by_id: dict[str, Any] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        if isinstance(row_id, str):
            score_by_id[row_id] = row.get("score")

    expected_ids = [short_name for short_name, _ in AUTORUBRIC_DIMENSIONS]
    return all(
        isinstance(score_by_id.get(score_id), (int, float))
        for score_id in expected_ids
    )


def run_single_evaluation(
    *,
    batch_args: argparse.Namespace,
    entries_path: Path,
    entries: list[dict[str, Any]],
    metadata_root: Path,
    progress_prompt_map: dict[str, dict[str, Any]],
    video_path: Path,
) -> dict[str, Any]:
    session = build_session(batch_args.api_key)
    eval_args = build_eval_args(batch_args, video_path, metadata_root)
    output_path = build_output_path(video_path, None, batch_args.question_mode, batch_args.output_tag)

    context = resolve_video_entry_context(
        video_path=video_path,
        entries_path=entries_path,
        entries=entries,
        metadata_root=metadata_root,
        progress_prompt_map=progress_prompt_map,
    )
    metadata_path = context["metadata_path"]
    metadata_payload = context["metadata_payload"]
    matched_entry = context["matched_entry"]
    matched_prompt_occurrences = context["matched_prompt_occurrences"]
    resolved_sample_index = get_metadata_value(metadata_payload, "sample_index")
    if not isinstance(resolved_sample_index, int):
        resolved_sample_index = infer_sample_index_from_video_name(video_path)

    if batch_args.reuse_existing and output_path.exists():
        payload = load_json(output_path)
        if batch_args.question_mode != QUESTION_MODE_AUTORUBRIC:
            return payload
        payload = normalize_reused_autorubric_payload(
            payload=payload,
            output_path=output_path,
            metadata_path=metadata_path,
            metadata_payload=metadata_payload,
            matched_entry=matched_entry,
            matched_prompt_occurrences=matched_prompt_occurrences,
        )
        if payload.get("model") == batch_args.model and autorubric_payload_has_valid_scores(payload):
            return payload

    qa_pairs, qa_source_field = get_entry_qa_pairs(matched_entry)
    autorubric = get_entry_autorubric(matched_entry)
    if batch_args.question_mode == QUESTION_MODE_AUTORUBRIC:
        if not autorubric:
            raise ValueError(f"No autorubric for {video_path.name}")
    elif not qa_pairs:
        raise ValueError(f"No QA pairs for {video_path.name}")

    video_base64 = encode_file_to_base64(video_path)
    mime_type = guess_mime_type(video_path)

    if batch_args.question_mode == QUESTION_MODE_AUTORUBRIC:
        rubric_scores, execution = execute_autorubric_evaluation(
            session=session,
            args=eval_args,
            entry=matched_entry,
            autorubric=autorubric,
            mime_type=mime_type,
            video_base64=video_base64,
        )
        evaluated_rows, average_score = evaluate_autorubric_scores(autorubric, rubric_scores)
        result_payload = {
            "success": True,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "video_file": str(video_path),
            "output_file": str(output_path),
            "metadata_file": str(metadata_path) if metadata_path is not None else None,
            "task_id": get_metadata_value(metadata_payload, "task_id"),
            "sample_index": resolved_sample_index,
            "model": batch_args.model,
            "question_mode": batch_args.question_mode,
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
            "average_score_percent": to_percent(normalize_rating(average_score)),
            "matched_prompt_occurrences": matched_prompt_occurrences,
            "prompt": matched_entry.get("prompt"),
            "matched_macro_domain": matched_entry.get("domain_info", {}).get("macro_domain"),
            "matched_micro_domain": matched_entry.get("domain_info", {}).get("micro_domain"),
            "results": evaluated_rows,
        }
        dump_json(output_path, result_payload)
        return result_payload

    if batch_args.question_mode == QUESTION_MODE_DEPENDENCY_ROUNDS:
        model_answers, execution = execute_dependency_round_evaluation(
            session=session,
            args=eval_args,
            qa_pairs=qa_pairs,
            mime_type=mime_type,
            video_base64=video_base64,
        )
    else:
        model_answers, execution = execute_all_at_once_evaluation(
            session=session,
            args=eval_args,
            qa_pairs=qa_pairs,
            mime_type=mime_type,
            video_base64=video_base64,
        )

    evaluated_rows, raw_correct_count, correct_count, dependency_blocked_count = evaluate_predictions(
        qa_pairs,
        model_answers,
    )
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
        "model": batch_args.model,
        "question_mode": batch_args.question_mode,
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
        "raw_accuracy_percent": to_percent(raw_accuracy),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "accuracy_percent": to_percent(accuracy),
        "dependency_blocked_count": dependency_blocked_count,
        "prompt": matched_entry.get("prompt"),
        "matched_macro_domain": matched_entry.get("domain_info", {}).get("macro_domain"),
        "matched_micro_domain": matched_entry.get("domain_info", {}).get("micro_domain"),
        "results": evaluated_rows,
    }
    dump_json(output_path, result_payload)
    return result_payload


def run_single_evaluation_with_retries(
    *,
    batch_args: argparse.Namespace,
    entries_path: Path,
    entries: list[dict[str, Any]],
    metadata_root: Path,
    progress_prompt_map: dict[str, dict[str, Any]],
    video_path: Path,
) -> dict[str, Any]:
    attempts = max(1, int(batch_args.eval_attempts))
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return run_single_evaluation(
                batch_args=batch_args,
                entries_path=entries_path,
                entries=entries,
                metadata_root=metadata_root,
                progress_prompt_map=progress_prompt_map,
                video_path=video_path,
            )
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                print(
                    f"  retrying {video_path.name} ({attempt}/{attempts}) after error: {exc}",
                    file=sys.stderr,
                )

    assert last_error is not None
    raise last_error


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if results and results[0].get("question_mode") == QUESTION_MODE_AUTORUBRIC:
        macro_counters: dict[str, Counter] = defaultdict(Counter)
        micro_counters: dict[str, Counter] = defaultdict(Counter)
        dimension_counters: dict[str, Counter] = defaultdict(Counter)
        sample_records: list[dict[str, Any]] = []
        sample_average_scores: list[float] = []

        for payload in results:
            macro = payload.get("matched_macro_domain") or "UNKNOWN"
            micro = payload.get("matched_micro_domain") or "UNKNOWN"
            sample_index = payload.get("sample_index")
            video_file = payload.get("video_file")
            ratings = extract_rubric_ratings_from_payload(payload)
            average_score = mean_rubric_rating(ratings)
            average_percent = to_percent(mean_rubric_score(ratings))
            sample_average_scores.append(average_score)

            macro_counters[macro].add(average_score, 1)
            micro_counters[micro].add(average_score, 1)

            for key, score in ratings.items():
                dimension_counters[key].add(score, 1)

            sample_records.append(
                {
                    "sample_index": sample_index,
                    "video_file": video_file,
                    "macro_domain": macro,
                    "micro_domain": micro,
                    "average_score": round(average_score, 6),
                    "average_score_percent": average_percent,
                    "scores": ratings,
                }
            )

        overall_total = sum(sample_average_scores)
        overall_count = len(sample_average_scores)

        def serialize_average(
            counter_map: dict[str, Counter],
            *,
            scale_to_100: bool,
        ) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for key in sorted(counter_map):
                counter = counter_map[key]
                average_value = counter.correct / counter.total if counter.total else 0.0
                rows.append(
                    {
                        "name": key,
                        "average_score": round(average_value, 6),
                        "video_count": counter.total,
                        "average_score_percent": (
                            to_percent(normalize_rating(average_value)) if scale_to_100 else None
                        ),
                    }
                )
            return rows

        dimension_rows: list[dict[str, Any]] = []
        order = [short_name for short_name, _ in AUTORUBRIC_DIMENSIONS]
        for key in order:
            counter = dimension_counters.get(key, Counter())
            average_value = counter.correct / counter.total if counter.total else 0.0
            dimension_rows.append(
                {
                    "name": key,
                    "average_score": round(average_value, 6),
                    "rating_count": counter.total,
                    "average_score_percent": to_percent(normalize_rating(average_value)),
                }
            )

        return {
            "overall": {
                "average_score": round(overall_total / overall_count, 6) if overall_count else 0.0,
                "video_count": overall_count,
                "average_score_percent": (
                    to_percent(normalize_rating(overall_total / overall_count)) if overall_count else 0.0
                ),
            },
            "by_macro_domain": serialize_average(macro_counters, scale_to_100=True),
            "by_micro_domain": serialize_average(micro_counters, scale_to_100=True),
            "by_rubric_dimension": dimension_rows,
            "per_video": sorted(sample_records, key=lambda item: (item["sample_index"] or 0)),
        }

    overall = Counter()
    macro_counters: dict[str, Counter] = defaultdict(Counter)
    micro_counters: dict[str, Counter] = defaultdict(Counter)
    type_counters: dict[str, Counter] = defaultdict(Counter)
    sample_records: list[dict[str, Any]] = []

    for payload in results:
        correct = int(payload.get("correct_count", 0))
        total = int(payload.get("question_count", 0))
        macro = payload.get("matched_macro_domain") or "UNKNOWN"
        micro = payload.get("matched_micro_domain") or "UNKNOWN"
        sample_index = payload.get("sample_index")
        video_file = payload.get("video_file")

        overall.add(correct, total)
        macro_counters[macro].add(correct, total)
        micro_counters[micro].add(correct, total)

        for row in payload.get("results", []):
            qa_type = row.get("type") or "UNKNOWN"
            type_counters[qa_type].add(1 if row.get("correct") else 0, 1)

        sample_records.append(
            {
                "sample_index": sample_index,
                "video_file": video_file,
                "macro_domain": macro,
                "micro_domain": micro,
                "correct_count": correct,
                "question_count": total,
                "accuracy": round(payload.get("accuracy", 0.0), 6),
                "accuracy_percent": payload.get("accuracy_percent", 0.0),
            }
        )

    def serialize(counter_map: dict[str, Counter]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in sorted(counter_map):
            counter = counter_map[key]
            rows.append(
                {
                    "name": key,
                    "correct_count": counter.correct,
                    "question_count": counter.total,
                    "accuracy": round(counter.accuracy, 6),
                    "accuracy_percent": to_percent(counter.accuracy),
                }
            )
        return rows

    return {
        "overall": {
            "correct_count": overall.correct,
            "question_count": overall.total,
            "accuracy": round(overall.accuracy, 6),
            "accuracy_percent": to_percent(overall.accuracy),
        },
        "by_macro_domain": serialize(macro_counters),
        "by_micro_domain": serialize(micro_counters),
        "by_qa_type": serialize(type_counters),
        "per_video": sorted(sample_records, key=lambda item: (item["sample_index"] or 0)),
    }


def main() -> int:
    args = parse_args()
    entries_path = resolve_entries_path_for_mode(args.entries, args.question_mode)
    entries, entries_source = load_entries_dataset(entries_path)
    videos_dir = resolve_videos_dir(args)
    metadata_root = resolve_metadata_root(args)
    progress_jsonl = resolve_progress_jsonl(args, videos_dir)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else build_default_output_path(videos_dir, args.question_mode)
    )

    if not videos_dir.exists():
        print(f"Error: videos dir not found: {videos_dir}", file=sys.stderr)
        return 1

    video_files = collect_video_files(videos_dir)
    video_files = slice_video_files(
        video_files,
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
    )
    if not video_files:
        print(f"Error: no mp4 files found under {videos_dir}", file=sys.stderr)
        return 1

    progress_prompt_map = load_progress_prompt_map(progress_jsonl)

    if not args.skip_model_check:
        try:
            session = build_session(args.api_key)
            models = fetch_available_models(session, args.base_url, args.timeout)
            print_model_check(args.model, models)
        except Exception as exc:
            print(f"Warning: failed to fetch model list: {exc}", file=sys.stderr)

    all_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    future_to_video: dict[Any, tuple[int, Path]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        for index, video_path in enumerate(video_files, start=1):
            future = executor.submit(
                run_single_evaluation_with_retries,
                batch_args=args,
                entries_path=entries_path,
                entries=entries,
                metadata_root=metadata_root,
                progress_prompt_map=progress_prompt_map,
                video_path=video_path,
            )
            future_to_video[future] = (index, video_path)

        for future in as_completed(future_to_video):
            index, video_path = future_to_video[future]
            print(f"[{index}/{len(video_files)}] Evaluated {video_path.name}")
            try:
                payload = future.result()
                all_results.append(payload)
                if args.question_mode == QUESTION_MODE_AUTORUBRIC:
                    print(
                        f"  average_score={payload['average_score']:.2f}/5 "
                        f"({payload['average_score_percent']}%)"
                    )
                else:
                    print(
                        f"  accuracy={payload['correct_count']}/{payload['question_count']} "
                        f"({payload['accuracy_percent']}%)"
                    )
            except Exception as exc:
                failures.append(
                    {
                        "video_file": str(video_path),
                        "error": str(exc),
                    }
                )
                print(f"  failed: {exc}", file=sys.stderr)

    summary = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(args.run_dir.resolve()),
        "videos_dir": str(videos_dir),
        "metadata_root": str(metadata_root),
        "progress_jsonl": str(progress_jsonl) if progress_jsonl is not None else None,
        "entries_file": str(entries_source),
        "question_mode": args.question_mode,
        "model": args.model,
        "video_count": len(video_files),
        "success_count": len(all_results),
        "failure_count": len(failures),
        "failures": failures,
        "aggregate": aggregate_results(all_results),
    }

    dump_json(output_path, summary)
    overall = summary["aggregate"]["overall"]
    print(f"\nSummary written to {output_path}")
    if args.question_mode == QUESTION_MODE_AUTORUBRIC:
        print(
            f"Overall autorubric average: {overall['average_score']}/5 "
            f"= {overall['average_score_percent']}%"
        )
    else:
        print(
            f"Overall accuracy: {overall['correct_count']}/{overall['question_count']} "
            f"= {overall['accuracy_percent']}%"
        )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
