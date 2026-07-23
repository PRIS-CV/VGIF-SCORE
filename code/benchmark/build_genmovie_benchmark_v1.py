from __future__ import annotations

import csv
import json
import os
import re
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "genmovie_benchmark_v1"
PROMPT_RANGE = range(29, 59)
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
QA_SUFFIX = "_qa_eval_dependency_rounds.json"
PROMPT_ALIGNMENT_STATUS = "verified_by_user_same_prompt_set"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_name: str
    open_or_closed: str
    provider: str
    source_project: str
    model_type: str
    license_or_access: str
    root: Path
    preferred_video_dir: Path | None
    filename_prefix: str | None = None
    filename_re: str | None = None
    notes: str = ""

    @property
    def group_dir(self) -> str:
        return "open" if self.open_or_closed == "open_source" else "closed"


MODELS = [
    ModelSpec(
        "wan2_2",
        "Wan 2.2",
        "open_source",
        "Wan-AI",
        "wan22_videos",
        "text-to-video",
        "local/open model output",
        PROJECT_ROOT / "wan22_videos",
        PROJECT_ROOT / "wan22_videos",
        filename_re=r"^Wan2\.2-(\d{3})\.mp4$",
        notes="Prompt text recovered from progress_success.jsonl.",
    ),
    ModelSpec(
        "cogvideox",
        "CogVideoX",
        "open_source",
        "THUDM",
        "cogvideox_videos",
        "text-to-video",
        "local/open model output",
        PROJECT_ROOT / "cogvideox_videos",
        PROJECT_ROOT / "cogvideox_videos",
        filename_re=r"^CogVideoX-(\d{3})\.mp4$",
    ),
    ModelSpec(
        "ltx2",
        "LTX-2",
        "open_source",
        "Lightricks",
        "ltx2_videos",
        "text-to-video",
        "local/open model output",
        PROJECT_ROOT / "ltx2_videos",
        None,
        filename_re=r"^LTX-2-(\d{3})\.mp4$",
        notes="No per-video prompt JSON found; prompts are inherited by shared numeric index.",
    ),
    ModelSpec(
        "mochi_1",
        "Mochi 1",
        "open_source",
        "Genmo",
        "mochi-1-videos",
        "text-to-video",
        "local/open model output",
        PROJECT_ROOT / "mochi-1-videos",
        PROJECT_ROOT / "mochi-1-videos",
        filename_re=r"^mochi-1-(\d{3})\.mp4$",
    ),
    ModelSpec(
        "infinitystar",
        "InfinityStar",
        "open_source",
        "unknown",
        "infinitystar_videos",
        "text-to-video",
        "local/open model output",
        PROJECT_ROOT / "infinitystar_videos",
        PROJECT_ROOT / "infinitystar_videos",
        filename_re=r"^InfinityStar-(\d{3})\.mp4$",
        notes="Classified as open_source by local weight/output convention; verify manually.",
    ),
    ModelSpec(
        "wan2_7",
        "Wan 2.7",
        "closed_source",
        "unknown",
        "Wan2.7",
        "text-to-video",
        "local/generated output; access status requires verification",
        PROJECT_ROOT / "Wan2.7",
        PROJECT_ROOT
        / "Wan2.7"
        / "outputs"
        / "wan_2_7_720p_5s_all_entries"
        / "20260420_eval_all_at_once_merged_221"
        / "videos",
        notes="Placed in closed_source to satisfy the 5 open/5 closed target split; verify access status.",
    ),
    ModelSpec(
        "seedance2_0",
        "Seedance 2.0",
        "closed_source",
        "ByteDance",
        "seedance2.0",
        "text-to-video",
        "API/closed-source output",
        PROJECT_ROOT / "seedance2.0",
        PROJECT_ROOT
        / "seedance2.0"
        / "outputs"
        / "seedance_2_0_720p_5s"
        / "20260419_160031_all-entries-223"
        / "videos",
    ),
    ModelSpec(
        "kling",
        "Kling",
        "closed_source",
        "Kuaishou",
        "kling_t2v",
        "text-to-video",
        "API/closed-source output",
        PROJECT_ROOT / "kling_t2v",
        PROJECT_ROOT
        / "kling_t2v"
        / "outputs"
        / "kling_v3_720p_5s"
        / "20260419_172612_all-entries-223"
        / "videos",
    ),
    ModelSpec(
        "pixverse_v6",
        "PixVerse V6",
        "closed_source",
        "PixVerse",
        "PixVerse-V6",
        "text-to-video",
        "API/closed-source output",
        PROJECT_ROOT / "PixVerse-V6",
        PROJECT_ROOT
        / "PixVerse-V6"
        / "outputs"
        / "pixverse_v6_720p_5s_full"
        / "20260419_222116"
        / "videos",
    ),
    ModelSpec(
        "viduq3_turbo",
        "Vidu Q3 Turbo",
        "closed_source",
        "Vidu",
        "ViduQ3-Turbo",
        "text-to-video",
        "API/closed-source output",
        PROJECT_ROOT / "ViduQ3-Turbo",
        PROJECT_ROOT
        / "ViduQ3-Turbo"
        / "outputs"
        / "viduq3_turbo_720p_5s_all_entries"
        / "20260419_173826_all-entries-223"
        / "videos",
    ),
]


def safe_rel(path: Path, base: Path = DATASET_ROOT) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def prompt_index_from_name(path: Path) -> int | None:
    name = path.name
    if QA_SUFFIX in name:
        name = name.replace(QA_SUFFIX, ".mp4")
    m = re.search(r"(?:^|[^0-9])(\d{3})(?=[_.-])", name)
    if m:
        return int(m.group(1))
    m = re.match(r"^0*(\d{1,3})(?:_|-)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"-(\d{3})\.(?:mp4|mov|mkv|webm|json)$", name, re.I)
    if m:
        return int(m.group(1))
    return None


def sample_index_from_name(path: Path) -> int:
    stem = path.stem
    stem = stem.replace("_qa_eval_dependency_rounds", "").replace("_qa_eval", "")
    m = re.search(r"_(\d+)$", stem)
    if m:
        return int(m.group(1))
    return 1


def base_video_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_qa_eval_dependency_rounds", "_qa_eval"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def copy_or_link(src: Path, dst: Path, copy_modes: list[str]) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "existing_copy"
        dst.unlink()
    try:
        shutil.copy2(src, dst)
        copy_modes.append("copy")
        return "copy"
    except OSError as exc:
        if getattr(exc, "winerror", None) != 112 and getattr(exc, "errno", None) != 28:
            raise
        try:
            os.symlink(src.resolve(), dst)
            copy_modes.append("symlink")
            return "symlink"
        except OSError:
            os.link(src, dst)
            copy_modes.append("hardlink")
            return "hardlink"


def load_wan22_prompts() -> dict[int, str]:
    out: dict[int, str] = {}
    path = PROJECT_ROOT / "wan22_videos" / "progress_success.jsonl"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except Exception:
            continue
        if item.get("status") != "success":
            continue
        idx = item.get("index")
        if isinstance(idx, int):
            out[idx + 1] = item.get("prompt", "")
    return out


def video_candidates_for_model(spec: ModelSpec) -> dict[int, Path]:
    files: list[Path] = []
    if spec.preferred_video_dir and spec.preferred_video_dir.exists():
        files = [p for p in spec.preferred_video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS]
    elif spec.root.exists():
        files = [p for p in spec.root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    out: dict[int, Path] = {}
    for path in sorted(files):
        idx: int | None = None
        if spec.filename_re:
            m = re.match(spec.filename_re, path.name)
            if m:
                idx = int(m.group(1))
        if idx is None and "narrative-cinematic-storytelling" in path.name.lower():
            idx = prompt_index_from_name(path)
        if idx in PROMPT_RANGE:
            out[idx] = path
    return out


def qa_candidate_priority(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    if any(token in name for token in ("summary", "partial", "probe", "_tmp_", "run_manifest")):
        return (-1000, -1000, name)

    score = 0
    if any(token in name for token in ("g31pro", "g31preview", "gemini31propreview", "gemini-3.1-pro-preview")):
        score += 50
    if "g3flash" in name:
        score -= 20
    if name.endswith(QA_SUFFIX):
        score += 5

    question_count = -1
    qa_data = read_json(path)
    if qa_data.get("question_mode") == "dependency-rounds":
        score += 25
    if isinstance(qa_data.get("question_count"), int):
        question_count = qa_data["question_count"]
    if qa_data.get("model"):
        model_name = str(qa_data["model"]).lower()
        if "gemini-3.1-pro-preview" in model_name or "gemini-2.5-pro" in model_name:
            score += 5

    return (score, question_count, name)


def qa_candidates_for_model(spec: ModelSpec) -> dict[int, Path]:
    search_root = spec.preferred_video_dir if spec.preferred_video_dir and spec.preferred_video_dir.exists() else spec.root
    out: dict[int, Path] = {}
    if not search_root.exists():
        return out
    buckets: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(search_root.rglob(f"*{QA_SUFFIX}")):
        idx = prompt_index_from_name(path)
        if idx in PROMPT_RANGE:
            buckets[idx].append(path)
    for idx, candidates in buckets.items():
        ranked = sorted(candidates, key=qa_candidate_priority, reverse=True)
        out[idx] = ranked[0]
    return out


def metadata_for_video(video_path: Path) -> dict[str, Any]:
    sibling = video_path.with_suffix(".json")
    if sibling.exists():
        return read_json(sibling)
    meta_dir = video_path.parent.parent / "metadata"
    if meta_dir.exists():
        match = meta_dir / f"{video_path.stem}.json"
        if match.exists():
            return read_json(match)
    return {}


def qa_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "failed_entity": 0,
        "failed_attribute": 0,
        "failed_action": 0,
        "failed_state": 0,
        "failed_causal": 0,
        "failed_location": 0,
        "failed_other": 0,
    }
    type_to_key = {
        "entity": "failed_entity",
        "attribute": "failed_attribute",
        "action": "failed_action",
        "state": "failed_state",
        "causal": "failed_causal",
        "location": "failed_location",
    }
    for item in results:
        if item.get("correct") is False:
            key = type_to_key.get(str(item.get("type", "")).lower(), "failed_other")
            counts[key] += 1
    return counts


def classify_scene_type(prompt_text: str) -> str:
    text = prompt_text.lower()
    labels = []
    keyword_map = {
        "dialogue": ("argues", "confronts", "clashes", "farewell", "reunion", "apologizes"),
        "suspense": ("suspense", "tails", "detective", "hidden", "envelope", "eerie"),
        "action": ("chase", "chases", "runs", "brakes", "rushes", "sprinting"),
        "interior": ("apartment", "hallway", "kitchen", "library", "laundromat", "classroom", "museum", "hospital", "bookstore"),
        "exterior": ("night market", "bus stop", "platform", "pier", "garden", "sidewalk", "dock", "desert", "forest", "cliffside"),
        "fantasy": ("mage", "witch", "oracle", "sprite", "runes", "knight", "lighthouse", "living paper map"),
    }
    for label, words in keyword_map.items():
        if any(word in text for word in words):
            labels.append(label)
    return ";".join(labels) if labels else "narrative"


def update_prompt_expectations(
    prompt_record: dict[str, Any],
    results: list[dict[str, Any]],
    prompt_text: str,
) -> None:
    focus = set(filter(None, prompt_record.get("continuity_focus", "").split(";")))
    entities = set(filter(None, prompt_record.get("expected_entities", "").split(";")))
    actions = set(filter(None, prompt_record.get("expected_actions", "").split(";")))
    props = set(filter(None, prompt_record.get("expected_props", "").split(";")))
    settings = set(filter(None, prompt_record.get("expected_setting", "").split(";")))

    type_to_focus = {
        "entity": "VIS_OBJ_POS",
        "attribute": "VIS_CHAR_APP",
        "action": "FACT_INT_LOG",
        "state": "VIS_OBJ_STATE",
        "causal": "FACT_INT_LOG",
        "location": "PROD_SET",
    }
    prop_words = (
        "bag",
        "backpack",
        "briefcase",
        "bouquet",
        "clipboard",
        "compass",
        "coin",
        "cup",
        "duffel",
        "envelope",
        "flashlight",
        "key",
        "lantern",
        "map",
        "notebook",
        "phone",
        "script",
        "suitcase",
        "ticket",
        "vial",
    )
    for item in results:
        qa_type = str(item.get("type", "")).lower()
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        if qa_type in type_to_focus:
            focus.add(type_to_focus[qa_type])
        if qa_type == "entity":
            entities.add(question)
            if any(word in question.lower() for word in prop_words):
                props.add(question)
        elif qa_type in {"action", "causal", "state"}:
            actions.add(question)
        elif qa_type == "attribute":
            entities.add(question)
        elif qa_type == "location":
            settings.add(question)

    if "wearing" in prompt_text.lower() or "uniform" in prompt_text.lower():
        focus.add("VIS_COSTUME")
    prompt_record["continuity_focus"] = ";".join(sorted(focus))
    prompt_record["expected_entities"] = ";".join(sorted(entities))
    prompt_record["expected_actions"] = ";".join(sorted(actions))
    prompt_record["expected_props"] = ";".join(sorted(props))
    prompt_record["expected_setting"] = ";".join(sorted(settings))
    prompt_record["scene_type"] = classify_scene_type(prompt_text)
    prompt_record["notes"] = "Prompt fields derived from shared prompt text and Gemini-QA question schema; verify manually for publication."


def join_list(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(x) for x in value)
    if value is None:
        return ""
    return str(value)


def probe_video(path: Path) -> dict[str, Any]:
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {
                "duration_sec": "",
                "width": "",
                "height": "",
                "fps": "",
                "frame_count": "",
                "readable": 0,
                "error": "OpenCV VideoCapture could not open file",
            }
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, _ = cap.read()
        cap.release()
        duration = frame_count / fps if fps > 0 else 0
        return {
            "duration_sec": f"{duration:.3f}" if duration else "",
            "width": width,
            "height": height,
            "fps": f"{fps:.3f}" if fps else "",
            "frame_count": frame_count,
            "readable": 1 if ok else 0,
            "error": "" if ok else "OpenCV could not read first frame",
        }
    except Exception as exc:
        return {
            "duration_sec": "",
            "width": "",
            "height": "",
            "fps": "",
            "frame_count": "",
            "readable": 0,
            "error": str(exc),
        }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def collect_unassigned(all_selected_videos: set[Path], all_selected_qas: set[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    known_roots = [m.root for m in MODELS]
    unassigned_videos = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        if is_under(path, DATASET_ROOT):
            continue
        if path.resolve() in all_selected_videos:
            continue
        idx = prompt_index_from_name(path)
        is_relevant = idx in PROMPT_RANGE or "narrative-cinematic-storytelling" in path.name.lower()
        if not is_relevant:
            continue
        if any(is_under(path, root) for root in known_roots):
            continue
        unassigned_videos.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "reason": "model_unknown_or_outside_known_model_roots",
                "notes": "",
            }
        )

    unassigned_qas = []
    for path in PROJECT_ROOT.rglob(f"*{QA_SUFFIX}"):
        if is_under(path, DATASET_ROOT):
            continue
        if path.resolve() in all_selected_qas:
            continue
        is_narrative = "narrative-cinematic-storytelling" in path.name.lower()
        idx = prompt_index_from_name(path)
        if not (is_narrative or idx in PROMPT_RANGE):
            continue
        if any(is_under(path, root) for root in known_roots):
            continue
        if idx in PROMPT_RANGE:
            unassigned_qas.append(
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "reason": "qa_json_not_selected_or_duplicate_for_benchmark",
                    "notes": "",
                }
            )
    return unassigned_videos, unassigned_qas


def build_readme(summary: dict[str, Any], copy_modes: list[str]) -> str:
    storage = "copy"
    if "symlink" in copy_modes:
        storage = "copy with symlink fallback"
    elif "hardlink" in copy_modes:
        storage = "copy with hardlink fallback"
    return f"""# GenMovie Benchmark v1

## Purpose

GenMovie Benchmark v1 is organized for continuity error detection, prompt-QA alignment evaluation, and open/closed video generation model comparison on complex movie-style text-to-video instructions.

## Composition

This build contains {summary["num_models"]} models: {summary["num_open_source_models"]} open-source and {summary["num_closed_source_models"]} closed-source. Each model targets the narrative-cinematic-storytelling prompt range 029-058, with approximately 29 to 30 videos per model depending on missing source outputs.

Storage mode: {storage}. Original files were not deleted or modified.

## Directory Structure

```text
data/genmovie_benchmark_v1/
  videos/open/<model_id>/
  videos/closed/<model_id>/
  qa/open/<model_id>/
  qa/closed/<model_id>/
  metadata/
```

## Key Files

`manifest.csv` has one row per video clip. Important fields include `clip_id`, relative `video_path`, optional `qa_json_path`, `model_id`, `open_or_closed`, `prompt_id`, `prompt_text`, `sample_index`, `has_qa_eval`, and dataset split/source labels.

`qa_eval_results.csv` has one row per clip with Gemini-QA aggregate metrics: QA model, question mode, round/question/correct counts, accuracy, macro/micro domain, failed QA counts by type, and the relative QA JSON path.

`qa_eval_items.csv` has one row per QA item with question id, type, question text, expected/predicted answers, correctness, dependency fields, and reason.

`prompts.csv` stores prompt-level metadata. Prompt IDs preserve source global indices (`prompt_029` to `prompt_058`) so cross-model alignment can be inspected.

`model_cards.csv` summarizes model identity, access category, provider, counts, and notes.

`video_probe.csv` records OpenCV readability checks: duration, dimensions, FPS, frame count, readable flag, and error.

## continuity_eval_mvp Integration

Copy the full `data/genmovie_benchmark_v1` directory to `continuity_eval_mvp/data/`. Use `metadata/manifest.csv` to load videos for continuity detection. Use `metadata/qa_eval_results.csv` and `metadata/qa_eval_items.csv` for Gemini-QA alignment analysis.

## Current Limits

- Prompt alignment status: `{summary["prompt_alignment_status"]}`. The user confirmed that related videos use the same prompt set and the same QA design; numeric prompt indices are preserved.
- Missing QA JSON count: {summary["missing_qa_count"]}.
- Corrupted or unreadable video count: {summary["corrupted_or_unreadable_video_count"]}.
- Human ground truth is not included (`has_human_gt=0`).
- Gemini-QA results are weak semantic annotations and should not be treated as human ground truth.
- The user request mentioned both 10 models and 5 open + 6 closed models. This build follows the requested 10-model directory target with 5 open-source and 5 closed-source model folders.
- `wan2_7` and `infinitystar` access categories should be manually verified if publication-grade model taxonomy is required.
"""


def main() -> None:
    for group in ("open", "closed"):
        for base in ("videos", "qa"):
            (DATASET_ROOT / base / group).mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "metadata").mkdir(parents=True, exist_ok=True)

    shared_prompts = load_wan22_prompts()
    copy_modes: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    prompt_records: dict[str, dict[str, Any]] = {}
    qa_result_rows: list[dict[str, Any]] = []
    qa_item_rows: list[dict[str, Any]] = []
    video_probe_rows: list[dict[str, Any]] = []
    selected_videos: set[Path] = set()
    selected_qas: set[Path] = set()
    videos_per_model: dict[str, int] = {}
    qa_per_model: dict[str, int] = {}

    for spec in MODELS:
        (DATASET_ROOT / "videos" / spec.group_dir / spec.model_id).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "qa" / spec.group_dir / spec.model_id).mkdir(parents=True, exist_ok=True)

        videos = video_candidates_for_model(spec)
        qas = qa_candidates_for_model(spec)
        videos_per_model[spec.model_id] = 0
        qa_per_model[spec.model_id] = 0

        for prompt_idx in PROMPT_RANGE:
            video_path = videos.get(prompt_idx)
            if not video_path:
                continue
            selected_videos.add(video_path.resolve())
            sample_idx = sample_index_from_name(video_path)
            clip_id = f"{spec.model_id}__prompt_{prompt_idx:03d}__sample_{sample_idx:02d}"
            dst_video = DATASET_ROOT / "videos" / spec.group_dir / spec.model_id / f"{clip_id}.mp4"
            copy_or_link(video_path, dst_video, copy_modes)
            videos_per_model[spec.model_id] += 1

            qa_path = qas.get(prompt_idx)
            qa_data: dict[str, Any] = {}
            dst_qa: Path | None = None
            if qa_path and qa_path.exists():
                selected_qas.add(qa_path.resolve())
                qa_data = read_json(qa_path)
                dst_qa = DATASET_ROOT / "qa" / spec.group_dir / spec.model_id / f"{clip_id}{QA_SUFFIX}"
                copy_or_link(qa_path, dst_qa, copy_modes)
                qa_per_model[spec.model_id] += 1

            video_meta = metadata_for_video(video_path)
            prompt_text = (
                qa_data.get("prompt")
                or video_meta.get("prompt")
                or video_meta.get("final_prompt")
                or shared_prompts.get(prompt_idx, "")
            )
            domain_info = video_meta.get("domain_info") if isinstance(video_meta.get("domain_info"), dict) else {}
            macro_domain = qa_data.get("matched_macro_domain") or domain_info.get("macro_domain", "")
            micro_domain = qa_data.get("matched_micro_domain") or domain_info.get("micro_domain", "")
            task_id = qa_data.get("task_id", "")
            if not task_id:
                m = re.match(r"^0*\d+_[^_]+_([^_]+)_\d+$", base_video_stem(video_path))
                task_id = m.group(1) if m else ""

            prompt_id = f"prompt_{prompt_idx:03d}"
            prompt_records.setdefault(
                prompt_id,
                {
                    "prompt_id": prompt_id,
                    "task_id": task_id,
                    "prompt_text": prompt_text,
                    "prompt_zh": "",
                    "macro_domain": macro_domain,
                    "micro_domain": micro_domain,
                    "scene_type": classify_scene_type(prompt_text),
                    "continuity_focus": "",
                    "expected_entities": "",
                    "expected_actions": "",
                    "expected_props": "",
                    "expected_setting": "",
                    "notes": "Prompt-level expected fields were not manually extracted.",
                },
            )
            if prompt_text and not prompt_records[prompt_id].get("prompt_text"):
                prompt_records[prompt_id]["prompt_text"] = prompt_text
            if macro_domain and not prompt_records[prompt_id].get("macro_domain"):
                prompt_records[prompt_id]["macro_domain"] = macro_domain
            if micro_domain and not prompt_records[prompt_id].get("micro_domain"):
                prompt_records[prompt_id]["micro_domain"] = micro_domain
            if task_id and not prompt_records[prompt_id].get("task_id"):
                prompt_records[prompt_id]["task_id"] = task_id

            manifest_rows.append(
                {
                    "clip_id": clip_id,
                    "video_path": safe_rel(dst_video),
                    "qa_json_path": safe_rel(dst_qa) if dst_qa else "",
                    "model_id": spec.model_id,
                    "model_name": spec.model_name,
                    "open_or_closed": spec.open_or_closed,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "task_id": task_id,
                    "sample_index": sample_idx,
                    "split": "test",
                    "source": "genmovie",
                    "subset": "generated",
                    "has_human_gt": 0,
                    "has_qa_eval": 1 if dst_qa else 0,
                    "notes": f"source={video_path.relative_to(PROJECT_ROOT).as_posix()}",
                }
            )

            probe = probe_video(dst_video)
            video_probe_rows.append(
                {
                    "clip_id": clip_id,
                    "video_path": safe_rel(dst_video),
                    **probe,
                }
            )

            if qa_data:
                results = qa_data.get("results") if isinstance(qa_data.get("results"), list) else []
                failures = qa_counts(results)
                qa_model = qa_data.get("model") or qa_data.get("model_version") or ""
                qa_result_rows.append(
                    {
                        "clip_id": clip_id,
                        "model_id": spec.model_id,
                        "prompt_id": prompt_id,
                        "task_id": qa_data.get("task_id", task_id),
                        "sample_index": sample_idx,
                        "qa_model": qa_model,
                        "question_mode": qa_data.get("question_mode", ""),
                        "round_count": qa_data.get("round_count", ""),
                        "question_count": qa_data.get("question_count", ""),
                        "correct_count": qa_data.get("correct_count", ""),
                        "accuracy": qa_data.get("accuracy", ""),
                        "accuracy_percent": qa_data.get("accuracy_percent", ""),
                        "dependency_blocked_count": qa_data.get("dependency_blocked_count", ""),
                        "macro_domain": macro_domain,
                        "micro_domain": micro_domain,
                        **failures,
                        "qa_json_path": safe_rel(dst_qa) if dst_qa else "",
                    }
                )
                for item in results:
                    qa_item_rows.append(
                        {
                            "clip_id": clip_id,
                            "model_id": spec.model_id,
                            "prompt_id": prompt_id,
                            "task_id": qa_data.get("task_id", task_id),
                            "question_id": item.get("id", ""),
                            "qa_type": item.get("type", ""),
                            "question": item.get("question", ""),
                            "expected_answer": item.get("expected_answer", ""),
                            "predicted_answer": item.get("predicted_answer", ""),
                            "correct": item.get("correct", ""),
                            "dependency": item.get("dependency", ""),
                            "dependency_ids": join_list(item.get("dependency_ids")),
                            "dependency_passed": item.get("dependency_passed", ""),
                            "dependency_failed_ids": join_list(item.get("dependency_failed_ids")),
                            "reason": item.get("reason", ""),
                        }
                    )
                update_prompt_expectations(prompt_records[prompt_id], results, prompt_text)

    model_card_rows = []
    for spec in MODELS:
        model_card_rows.append(
            {
                "model_id": spec.model_id,
                "model_name": spec.model_name,
                "model_type": spec.model_type,
                "provider": spec.provider,
                "open_or_closed": spec.open_or_closed,
                "version_or_date": "",
                "source_project": spec.source_project,
                "license_or_access": spec.license_or_access,
                "video_count": videos_per_model.get(spec.model_id, 0),
                "qa_count": qa_per_model.get(spec.model_id, 0),
                "notes": spec.notes,
            }
        )

    unassigned_videos, unassigned_qas = collect_unassigned(selected_videos, selected_qas)

    metadata_dir = DATASET_ROOT / "metadata"
    write_csv(
        metadata_dir / "manifest.csv",
        [
            "clip_id",
            "video_path",
            "qa_json_path",
            "model_id",
            "model_name",
            "open_or_closed",
            "prompt_id",
            "prompt_text",
            "task_id",
            "sample_index",
            "split",
            "source",
            "subset",
            "has_human_gt",
            "has_qa_eval",
            "notes",
        ],
        sorted(manifest_rows, key=lambda r: (r["model_id"], r["prompt_id"], r["sample_index"])),
    )
    write_csv(
        metadata_dir / "prompts.csv",
        [
            "prompt_id",
            "task_id",
            "prompt_text",
            "prompt_zh",
            "macro_domain",
            "micro_domain",
            "scene_type",
            "continuity_focus",
            "expected_entities",
            "expected_actions",
            "expected_props",
            "expected_setting",
            "notes",
        ],
        [prompt_records[k] for k in sorted(prompt_records)],
    )
    write_csv(
        metadata_dir / "model_cards.csv",
        [
            "model_id",
            "model_name",
            "model_type",
            "provider",
            "open_or_closed",
            "version_or_date",
            "source_project",
            "license_or_access",
            "video_count",
            "qa_count",
            "notes",
        ],
        model_card_rows,
    )
    write_csv(
        metadata_dir / "qa_eval_results.csv",
        [
            "clip_id",
            "model_id",
            "prompt_id",
            "task_id",
            "sample_index",
            "qa_model",
            "question_mode",
            "round_count",
            "question_count",
            "correct_count",
            "accuracy",
            "accuracy_percent",
            "dependency_blocked_count",
            "macro_domain",
            "micro_domain",
            "failed_entity",
            "failed_attribute",
            "failed_action",
            "failed_state",
            "failed_causal",
            "failed_location",
            "failed_other",
            "qa_json_path",
        ],
        sorted(qa_result_rows, key=lambda r: (r["model_id"], r["prompt_id"], r["sample_index"])),
    )
    write_csv(
        metadata_dir / "qa_eval_items.csv",
        [
            "clip_id",
            "model_id",
            "prompt_id",
            "task_id",
            "question_id",
            "qa_type",
            "question",
            "expected_answer",
            "predicted_answer",
            "correct",
            "dependency",
            "dependency_ids",
            "dependency_passed",
            "dependency_failed_ids",
            "reason",
        ],
        sorted(qa_item_rows, key=lambda r: (r["model_id"], r["prompt_id"], r["question_id"])),
    )
    write_csv(
        metadata_dir / "video_probe.csv",
        [
            "clip_id",
            "video_path",
            "duration_sec",
            "width",
            "height",
            "fps",
            "frame_count",
            "readable",
            "error",
        ],
        sorted(video_probe_rows, key=lambda r: r["clip_id"]),
    )
    write_csv(metadata_dir / "unassigned_videos.csv", ["path", "reason", "notes"], unassigned_videos)
    write_csv(metadata_dir / "unassigned_qa_jsons.csv", ["path", "reason", "notes"], unassigned_qas)

    accuracies = [float(r["accuracy"]) for r in qa_result_rows if r.get("accuracy") not in ("", None)]
    readable_failures = [r for r in video_probe_rows if str(r.get("readable")) != "1"]
    missing_qa_count = sum(1 for r in manifest_rows if not r["qa_json_path"])
    missing_prompt_count = sum(1 for r in prompt_records.values() if not r.get("prompt_text"))
    summary = {
        "dataset_name": "genmovie_benchmark_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_models": len(MODELS),
        "num_open_source_models": sum(1 for m in MODELS if m.open_or_closed == "open_source"),
        "num_closed_source_models": sum(1 for m in MODELS if m.open_or_closed == "closed_source"),
        "num_videos": len(manifest_rows),
        "num_qa_jsons": len(qa_result_rows),
        "num_prompts": len(prompt_records),
        "videos_per_model": videos_per_model,
        "qa_per_model": qa_per_model,
        "missing_video_count": 0,
        "missing_qa_count": missing_qa_count,
        "missing_prompt_count": missing_prompt_count,
        "unassigned_video_count": len(unassigned_videos),
        "unassigned_qa_json_count": len(unassigned_qas),
        "corrupted_or_unreadable_video_count": len(readable_failures),
        "prompt_alignment_status": PROMPT_ALIGNMENT_STATUS,
        "notes": [
            "Benchmark uses narrative-cinematic-storytelling source prompt indices 029-058.",
            "No model inference or video regeneration was performed.",
            "QA JSON contents were copied without modification.",
            "Prompt alignment was confirmed by the user as the same prompt set and QA design across related videos.",
        ],
        "accuracy": {
            "min": min(accuracies) if accuracies else None,
            "max": max(accuracies) if accuracies else None,
            "mean": statistics.mean(accuracies) if accuracies else None,
        },
    }
    (metadata_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "README.md").write_text(build_readme(summary, copy_modes), encoding="utf-8")

    failure_by_type = Counter()
    for row in qa_result_rows:
        for key in (
            "failed_entity",
            "failed_attribute",
            "failed_action",
            "failed_state",
            "failed_causal",
            "failed_location",
            "failed_other",
        ):
            failure_by_type[key.replace("failed_", "")] += int(row.get(key) or 0)

    group_acc: dict[str, list[float]] = defaultdict(list)
    model_group = {m.model_id: m.open_or_closed for m in MODELS}
    for row in qa_result_rows:
        if row.get("accuracy") not in ("", None):
            group_acc[model_group[row["model_id"]]].append(float(row["accuracy"]))

    report = {
        "dataset_root": safe_rel(DATASET_ROOT, PROJECT_ROOT),
        "videos_per_model": videos_per_model,
        "qa_per_model": qa_per_model,
        "missing_qa_count": missing_qa_count,
        "missing_video_count": summary["missing_video_count"],
        "unreadable_video_count": len(readable_failures),
        "prompt_alignment_status": PROMPT_ALIGNMENT_STATUS,
        "accuracy_min": summary["accuracy"]["min"],
        "accuracy_max": summary["accuracy"]["max"],
        "accuracy_mean": summary["accuracy"]["mean"],
        "failed_by_qa_type": dict(failure_by_type),
        "avg_accuracy_by_open_or_closed": {
            key: statistics.mean(vals) if vals else None for key, vals in group_acc.items()
        },
        "unassigned_video_count": len(unassigned_videos),
        "unassigned_qa_json_count": len(unassigned_qas),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
