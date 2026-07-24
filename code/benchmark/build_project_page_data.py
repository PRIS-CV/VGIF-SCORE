from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[2]
MODEL_ORDER = [
    "Kling-V3",
    "Seedance-2.0",
    "Wan-2.7",
    "ViduQ3-Turbo",
    "PixVerse-V6",
    "LTX-2.0",
    "Wan2.2-A14B",
    "HyVideo-1.5",
    "LongCat-Video",
    "Mochi-1",
    "CogVideoX-1.5",
    "MAGI-1",
    "URSA",
    "InfinityStar",
]
COMMERCIAL_MODELS = set(MODEL_ORDER[:5])

MACRO_DOMAINS = [
    ("product", "Commercial & Product Showcase", "Product"),
    ("narrative", "Narrative & Cinematic Storytelling", "Narrative"),
    ("surreal", "Creative & Surreal Expression", "Surreal"),
    ("physics", "Dynamics & Physical Interaction", "Physics"),
    ("emotion", "Emotion & Atmosphere Expression", "Emotion"),
    ("spatial", "Spatial Composition & Scene Orchestration", "Spatial"),
    ("performance", "Performance, Sports & Embodied Motion", "Performance"),
    ("nature", "Travel, Nature & Living World", "Nature"),
]

# Camera-ready Table 3 values. These remain the canonical public macro results.
PAPER_MACRO_SCORES = {
    "Kling-V3": [46.13, 43.12, 44.85, 43.25, 50.39, 48.78, 49.31, 45.46],
    "Seedance-2.0": [47.26, 46.14, 48.82, 40.93, 53.18, 49.66, 49.21, 46.83],
    "Wan-2.7": [49.77, 43.03, 53.17, 50.62, 65.48, 58.20, 61.60, 58.06],
    "ViduQ3-Turbo": [46.45, 38.63, 42.77, 42.61, 51.47, 46.50, 50.06, 45.97],
    "PixVerse-V6": [46.20, 42.74, 46.05, 44.17, 50.24, 47.24, 48.29, 54.59],
    "LTX-2.0": [33.00, 34.78, 34.65, 34.03, 37.70, 40.18, 38.79, 40.20],
    "Wan2.2-A14B": [38.16, 37.65, 41.59, 39.57, 32.23, 39.96, 39.95, 47.58],
    "HyVideo-1.5": [40.43, 29.47, 35.06, 35.62, 47.67, 39.36, 43.47, 45.26],
    "LongCat-Video": [37.73, 34.09, 33.23, 36.56, 44.95, 40.12, 38.95, 43.86],
    "Mochi-1": [33.87, 27.40, 34.39, 30.42, 39.44, 34.72, 32.14, 34.80],
    "CogVideoX-1.5": [28.81, 29.87, 29.71, 26.00, 36.37, 35.92, 30.43, 30.43],
    "MAGI-1": [27.55, 24.46, 25.87, 28.55, 29.39, 29.39, 24.06, 26.80],
    "URSA": [33.19, 25.23, 32.70, 30.64, 31.97, 29.61, 30.66, 34.12],
    "InfinityStar": [33.44, 27.06, 32.93, 33.26, 41.65, 40.47, 34.46, 42.02],
}

CASE_LIBRARY = [
    {"id": "product", "sample_number": 5, "models": ["Seedance-2.0", "ViduQ3-Turbo"]},
    {"id": "narrative", "sample_number": 34, "models": ["ViduQ3-Turbo", "MAGI-1"]},
    {"id": "surreal", "sample_number": 219, "models": ["Kling-V3", "CogVideoX-1.5"]},
    {"id": "physics", "sample_number": 89, "models": ["Wan-2.7", "Kling-V3"]},
    {"id": "emotion", "sample_number": 122, "models": ["InfinityStar", "Seedance-2.0"]},
    {"id": "spatial", "sample_number": 145, "models": ["ViduQ3-Turbo", "CogVideoX-1.5"]},
    {"id": "performance", "sample_number": 172, "models": ["Wan2.2-A14B", "CogVideoX-1.5"]},
    {"id": "nature", "sample_number": 203, "models": ["Wan-2.7", "InfinityStar"]},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aggregate data for the VGIF-Score project page.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_DIR / "data" / "results_manifest.csv",
    )
    parser.add_argument(
        "--category-output",
        type=Path,
        default=REPO_DIR / "docs" / "data" / "category_scores.json",
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        default=REPO_DIR / "docs" / "data" / "case_study.json",
    )
    parser.add_argument(
        "--candidates-output",
        type=Path,
        default=REPO_DIR / "tmp" / "project_page_collage_candidates.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def english_name(value: Any) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    return value.split("(", 1)[0].strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def rubric_scores(payload: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for result in payload.get("results", []):
        key = result.get("id")
        score = result.get("score")
        if key in {"Cin", "Pur", "Mot", "Phy"} and isinstance(score, (int, float)):
            scores[key] = float(score)
    if set(scores) != {"Cin", "Pur", "Mot", "Phy"}:
        raise ValueError("AutoRubric payload does not contain all four paper dimensions")
    return scores


def score_pair(qa_payload: dict[str, Any], rubric_payload: dict[str, Any]) -> tuple[float, float, float]:
    correct = int(qa_payload["correct_count"])
    total = int(qa_payload["question_count"])
    if total <= 0:
        raise ValueError("QA question_count must be positive")
    objective = correct / total
    ratings = rubric_scores(rubric_payload)
    subjective = sum(ratings.values()) / (4.0 * 5.0)
    vgif = 0.5 * objective + 0.5 * subjective
    return objective, subjective, vgif


def video_index(model: str) -> dict[str, Path]:
    root = REPO_DIR / "models" / model / "videos"
    if not root.is_dir():
        return {}
    return {path.name: path for path in root.rglob("*.mp4")}


def build_category_data(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    macro_id_by_name = {name: domain_id for domain_id, name, _ in MACRO_DOMAINS}
    buckets: dict[str, dict[str, dict[str, list[float]]]] = {
        "micro": defaultdict(lambda: defaultdict(list)),
    }
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    micro_to_macro: dict[str, str] = {}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    indexes = {model: video_index(model) for model in MODEL_ORDER}

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        rows = csv.DictReader(file_obj)
        for row in rows:
            if row.get("complete", "").lower() != "true":
                continue
            model = row["model"]
            if model not in MODEL_ORDER:
                continue
            qa_path = REPO_DIR / row["qa_file"]
            rubric_path = REPO_DIR / row["rubric_file"]
            if not qa_path.is_file() or not rubric_path.is_file():
                continue
            try:
                qa_payload = load_json(qa_path)
                rubric_payload = load_json(rubric_path)
                objective, subjective, vgif = score_pair(qa_payload, rubric_payload)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

            macro_name = english_name(
                qa_payload.get("matched_macro_domain") or rubric_payload.get("matched_macro_domain")
            )
            micro_name = english_name(
                qa_payload.get("matched_micro_domain") or rubric_payload.get("matched_micro_domain")
            )
            macro_id = macro_id_by_name.get(macro_name)
            if macro_id is None or micro_name == "UNKNOWN":
                continue

            micro_to_macro[micro_name] = macro_id
            buckets["micro"][micro_name][model].append(vgif * 100.0)
            coverage[micro_name][model] += 1

            video_file = qa_payload.get("video_file") or rubric_payload.get("video_file")
            video_name = Path(str(video_file)).name if video_file else ""
            local_video = indexes[model].get(video_name)
            if local_video is not None:
                candidates[macro_id].append(
                    {
                        "model": model,
                        "sample_id": row["sample_id"],
                        "macro": macro_name,
                        "micro": micro_name,
                        "vgif": round(vgif * 100.0, 2),
                        "objective": round(objective * 100.0, 2),
                        "subjective": round(subjective * 100.0, 2),
                        "video": str(local_video.relative_to(REPO_DIR)).replace("\\", "/"),
                    }
                )

    macro_categories = [
        {"id": domain_id, "name": name, "short": short}
        for domain_id, name, short in MACRO_DOMAINS
    ]
    macro_scores = []
    for model in MODEL_ORDER:
        values = PAPER_MACRO_SCORES[model]
        macro_scores.append(
            {
                "model": model,
                "group": "commercial" if model in COMMERCIAL_MODELS else "open",
                "values": {
                    domain_id: value
                    for (domain_id, _, _), value in zip(MACRO_DOMAINS, values, strict=True)
                },
            }
        )

    macro_position = {domain_id: position for position, (domain_id, _, _) in enumerate(MACRO_DOMAINS)}
    micro_names = sorted(
        buckets["micro"],
        key=lambda name: (macro_position[micro_to_macro[name]], name.casefold()),
    )
    micro_categories = [
        {
            "id": slugify(name),
            "name": name,
            "short": name,
            "macro": micro_to_macro[name],
        }
        for name in micro_names
    ]
    micro_ids = {item["name"]: item["id"] for item in micro_categories}
    micro_scores = []
    for model in MODEL_ORDER:
        values: dict[str, float | None] = {}
        counts: dict[str, int] = {}
        for name in micro_names:
            samples = buckets["micro"][name].get(model, [])
            values[micro_ids[name]] = round(sum(samples) / len(samples), 2) if samples else None
            counts[micro_ids[name]] = coverage[name].get(model, 0)
        micro_scores.append(
            {
                "model": model,
                "group": "commercial" if model in COMMERCIAL_MODELS else "open",
                "values": values,
                "coverage": counts,
            }
        )

    category_payload = {
        "macro": {
            "source": "Camera-ready Table 3",
            "categories": macro_categories,
            "scores": macro_scores,
        },
        "micro": {
            "source": "Canonical selected evaluation manifest; per-sample VGIF macro-average within each micro domain",
            "categories": micro_categories,
            "scores": micro_scores,
        },
    }
    candidate_payload = {
        domain_id: sorted(rows, key=lambda item: item["vgif"], reverse=True)[:12]
        for domain_id, rows in candidates.items()
    }
    return category_payload, candidate_payload


def load_manifest_rows(manifest_path: Path) -> dict[tuple[str, int], dict[str, str]]:
    selected: dict[tuple[str, int], dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if row.get("complete", "").lower() != "true":
                continue
            selected[(row["model"], int(row["sample_id"]))] = row
    return selected


def model_slug(model: str) -> str:
    return slugify(model.replace(".", "-"))


def build_case_item(
    case_spec: dict[str, Any],
    manifest_rows: dict[tuple[str, int], dict[str, str]],
) -> dict[str, Any]:
    sample_number = int(case_spec["sample_number"])
    annotation_path = (
        REPO_DIR
        / "data"
        / "autorubric"
        / f"sample_{sample_number - 1:04d}_complete.json"
    )
    annotation = load_json(annotation_path)
    nodes_by_id = {node["node_id"]: node for node in annotation["st_dag"]["nodes"]}
    qa_items = []
    for qa in annotation["original_qa_pairs"]:
        parents = re.findall(r"q\d+", qa.get("dependency", ""), flags=re.IGNORECASE)
        node = nodes_by_id[qa["node_id"]]
        qa_items.append(
            {
                "id": qa["id"],
                "node_id": qa["node_id"],
                "type": qa["type"],
                "label": node["label"],
                "question": qa["question"],
                "dependency": qa.get("dependency", "None"),
                "parents": parents,
            }
        )

    rubric_order = [
        ("cinematography", "Cinematography", "Cin"),
        ("purity", "Visual purity", "Pur"),
        ("motion_smoothness", "Motion smoothness", "Mot"),
        ("physics_adherence", "Physics adherence", "Phy"),
    ]
    rubric_dimensions = []
    source_dimensions = annotation["autorubric"]["dimensions"]
    for key, title, result_id in rubric_order:
        source_key = "visual_purity" if key == "purity" and key not in source_dimensions else key
        source = source_dimensions[source_key]
        rubric_dimensions.append(
            {
                "key": key,
                "title": title,
                "result_id": result_id,
                "goal": source["dimension_goal"],
                "criteria": source["prompt_specific_criteria"],
                "focus_points": source.get("focus_points", []),
                "anchors": source.get("score_anchors", {}),
            }
        )

    model_payloads = []
    for position, model in enumerate(case_spec["models"]):
        row = manifest_rows.get((model, sample_number))
        if row is None:
            raise ValueError(f"Missing complete manifest row for {model}, sample {sample_number}")
        qa_payload = load_json(REPO_DIR / row["qa_file"])
        rubric_payload = load_json(REPO_DIR / row["rubric_file"])
        objective, subjective, vgif = score_pair(qa_payload, rubric_payload)
        qa_results = {
            item["id"]: {
                "correct": bool(item.get("correct")),
                "answer": item.get("predicted_answer"),
                "dependency_passed": bool(item.get("dependency_passed", True)),
                "failed_dependencies": item.get("dependency_failed_ids", []),
                "reason": item.get("reason", ""),
            }
            for item in qa_payload["results"]
        }
        rubric_results = {
            item["id"]: {
                "score": item["score"],
                "reason": item.get("reason", ""),
            }
            for item in rubric_payload["results"]
            if item.get("id") in {"Cin", "Pur", "Mot", "Phy"}
        }
        model_payloads.append(
            {
                "model": model,
                "role": "stronger" if position == 0 else "contrast",
                "video": f"assets/videos/cases/{case_spec['id']}/{model_slug(model)}.mp4",
                "poster": f"assets/videos/cases/{case_spec['id']}/{model_slug(model)}.jpg",
                "objective": round(objective * 100.0, 2),
                "subjective": round(subjective * 100.0, 2),
                "vgif": round(vgif * 100.0, 2),
                "qa_correct": qa_payload["correct_count"],
                "qa_total": qa_payload["question_count"],
                "qa": qa_results,
                "rubric": rubric_results,
            }
        )

    return {
        "id": case_spec["id"],
        "sample_number": sample_number,
        "sample_id": annotation["sample_id"],
        "prompt": annotation["prompt"],
        "macro_domain": english_name(annotation["domain_info"]["macro_domain"]),
        "micro_domain": english_name(annotation["domain_info"]["micro_domain"]),
        "qa": qa_items,
        "rubric": rubric_dimensions,
        "models": model_payloads,
    }


def build_case_library_data(manifest_path: Path) -> dict[str, Any]:
    manifest_rows = load_manifest_rows(manifest_path)
    cases = [build_case_item(case_spec, manifest_rows) for case_spec in CASE_LIBRARY]
    return {
        "default_case": "surreal",
        "cases": cases,
        "model_count": len({model["model"] for case in cases for model in case["models"]}),
    }


def main() -> int:
    args = parse_args()
    category_data, candidate_data = build_category_data(args.manifest.resolve())
    if len(category_data["micro"]["categories"]) != 38:
        raise ValueError(
            f"Expected 38 micro domains, found {len(category_data['micro']['categories'])}"
        )
    dump_json(args.category_output.resolve(), category_data)
    dump_json(args.case_output.resolve(), build_case_library_data(args.manifest.resolve()))
    dump_json(args.candidates_output.resolve(), candidate_data)
    print(f"Category data: {args.category_output.resolve()}")
    print(f"Case study data: {args.case_output.resolve()}")
    print(f"Collage candidates: {args.candidates_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
