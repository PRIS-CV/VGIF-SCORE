from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from build_project_page_data import (
    COMMERCIAL_MODELS,
    MACRO_DOMAINS,
    REPO_DIR,
    english_name,
    load_json,
    score_pair,
    video_index,
)


CANVAS_SIZE = (2560, 1280)
DOMAIN_IDS = {name: domain_id for domain_id, name, _ in MACRO_DOMAINS}
FRAME_TIMES = (0.75, 1.35, 2.05, 2.75, 3.45)
TILES_PER_DOMAIN = 4
HERO_MODELS = (
    "Kling-V3",
    "Seedance-2.0",
    "Wan-2.7",
    "ViduQ3-Turbo",
    "PixVerse-V6",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the VGIF-Score hero from commercial-model video frames."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_DIR / "data" / "results_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_DIR / "docs" / "assets" / "domain_collage_v2.jpg",
    )
    parser.add_argument(
        "--sources-output",
        type=Path,
        default=REPO_DIR / "docs" / "assets" / "hero_collage_sources.json",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=REPO_DIR / "tmp" / "hero_collage_frames",
    )
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_ffmpeg(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    discovered = shutil.which("ffmpeg")
    if discovered and Path(discovered).is_file():
        try:
            subprocess.run(
                [discovered, "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return Path(discovered)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "ffmpeg is required; pass --ffmpeg with its executable path"
        ) from exc


def candidate_scores(
    qa_payload: dict[str, Any], rubric_payload: dict[str, Any]
) -> tuple[float, float, float]:
    try:
        return score_pair(qa_payload, rubric_payload)
    except ValueError:
        raw_scores = rubric_payload.get("scores")
        if not isinstance(raw_scores, dict):
            raise
        ratings = []
        for key in ("Cin", "Pur", "Mot", "Phy"):
            value = raw_scores.get(key)
            if isinstance(value, dict):
                value = value.get("score")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Missing legacy AutoRubric score: {key}")
            ratings.append(float(value))
        correct = int(qa_payload["correct_count"])
        total = int(qa_payload["question_count"])
        objective = correct / total
        subjective = sum(ratings) / 20.0
        return objective, subjective, 0.5 * objective + 0.5 * subjective


def collect_candidates(manifest_path: Path) -> dict[str, list[dict[str, Any]]]:
    indexes = {model: video_index(model) for model in COMMERCIAL_MODELS}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            model = row["model"]
            if model not in COMMERCIAL_MODELS or row.get("complete", "").lower() != "true":
                continue
            qa_path = REPO_DIR / row["qa_file"]
            rubric_path = REPO_DIR / row["rubric_file"]
            if not qa_path.is_file() or not rubric_path.is_file():
                continue
            try:
                qa_payload = load_json(qa_path)
                rubric_payload = load_json(rubric_path)
                objective, subjective, vgif = candidate_scores(qa_payload, rubric_payload)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

            macro_name = english_name(
                qa_payload.get("matched_macro_domain")
                or rubric_payload.get("matched_macro_domain")
            )
            domain_id = DOMAIN_IDS.get(macro_name)
            if domain_id is None:
                continue
            micro_name = english_name(
                qa_payload.get("matched_micro_domain")
                or rubric_payload.get("matched_micro_domain")
            )
            video_name = Path(
                str(qa_payload.get("video_file") or rubric_payload.get("video_file") or "")
            ).name
            source = indexes[model].get(video_name)
            if source is None:
                continue
            candidates[domain_id].append(
                {
                    "domain": domain_id,
                    "macro": macro_name,
                    "micro": micro_name,
                    "model": model,
                    "sample_id": int(row["sample_id"]),
                    "objective": objective * 100.0,
                    "subjective": subjective * 100.0,
                    "vgif": vgif * 100.0,
                    "video": source,
                }
            )
    return candidates


def select_candidates(
    candidates: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    model_counts: Counter[str] = Counter()

    for domain_index, (domain_id, _, _) in enumerate(MACRO_DOMAINS):
        pool = candidates.get(domain_id, [])
        if len(pool) < TILES_PER_DOMAIN:
            raise ValueError(f"Not enough commercial-model frames for {domain_id}")
        preferred_model = HERO_MODELS[domain_index % len(HERO_MODELS)]
        preferred_pool = [item for item in pool if item["model"] == preferred_model]
        if not preferred_pool:
            raise ValueError(f"No {preferred_model} candidate is available for {domain_id}")
        first_choice = max(
            preferred_pool,
            key=lambda item: item["subjective"] * 0.55 + item["vgif"] * 0.45,
        )
        domain_selected: list[dict[str, Any]] = [first_choice]
        used_models: set[str] = {first_choice["model"]}
        used_micros: set[str] = {first_choice["micro"]}
        model_counts[first_choice["model"]] += 1
        while len(domain_selected) < TILES_PER_DOMAIN:
            remaining = [item for item in pool if item not in domain_selected]

            def rank(item: dict[str, Any]) -> float:
                quality = item["subjective"] * 0.55 + item["vgif"] * 0.45
                micro_bonus = 14.0 if item["micro"] not in used_micros else 0.0
                model_bonus = 9.0 if item["model"] not in used_models else 0.0
                balance_bonus = max(0.0, 12.0 - model_counts[item["model"]] * 3.0)
                return quality + micro_bonus + model_bonus + balance_bonus

            choice = max(remaining, key=rank)
            domain_selected.append(choice)
            used_models.add(choice["model"])
            used_micros.add(choice["micro"])
            model_counts[choice["model"]] += 1
        selected.extend(domain_selected)

    missing_models = COMMERCIAL_MODELS - set(model_counts)
    if missing_models:
        raise ValueError(f"Commercial model coverage is incomplete: {sorted(missing_models)}")
    return selected


def extract_frame(
    ffmpeg: Path,
    source: Path,
    target: Path,
    timestamp: float,
    force: bool,
) -> None:
    if target.exists() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if force else "-n",
            "-ss",
            f"{timestamp:.2f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-update",
            "1",
            "-vf",
            "scale='min(1440,iw)':-2",
            "-q:v",
            "2",
            str(target),
        ],
        check=True,
    )


def row_geometry(
    rng: random.Random,
    count: int,
    y: int,
    height: int,
) -> list[tuple[int, int, int, int, float]]:
    overlap = 34
    available = CANVAS_SIZE[0] + overlap * (count - 1) + 90
    weights = [rng.uniform(0.82, 1.30) for _ in range(count)]
    widths = [round(available * weight / sum(weights)) for weight in weights]
    x = -45
    geometry = []
    for index, width in enumerate(widths):
        jitter_y = rng.randint(-28, 24)
        tile_height = height + rng.randint(-34, 40)
        angle = rng.choice((-1.8, -1.1, -0.5, 0.4, 0.9, 1.5))
        if index in {0, count - 1}:
            angle *= 0.45
        geometry.append((x, y + jitter_y, width, tile_height, angle))
        x += width - overlap
    return geometry


def paste_tile(
    canvas: Image.Image,
    source: Path,
    geometry: tuple[int, int, int, int, float],
) -> None:
    x, y, width, height, angle = geometry
    image = Image.open(source).convert("RGB")
    crop = ImageOps.fit(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.48),
    )
    card = ImageOps.expand(crop, border=5, fill="#f4f6f5").convert("RGBA")
    rotated = card.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    shadow = Image.new("RGBA", rotated.size, (0, 0, 0, 0))
    shadow_mask = rotated.getchannel("A").filter(ImageFilter.GaussianBlur(13))
    shadow.putalpha(shadow_mask.point(lambda value: round(value * 0.48)))
    canvas.alpha_composite(shadow, (x + 11, y + 15))
    canvas.alpha_composite(rotated, (x, y))


def add_graph_overlay(canvas: Image.Image) -> None:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    points = [
        (1450, 190),
        (1655, 120),
        (1790, 300),
        (1995, 190),
        (2225, 340),
        (1570, 505),
        (1850, 565),
        (2110, 655),
        (2350, 570),
        (1690, 815),
        (1970, 930),
        (2260, 875),
        (2420, 1085),
    ]
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (2, 5),
        (3, 4),
        (3, 6),
        (5, 6),
        (5, 9),
        (6, 7),
        (7, 8),
        (7, 10),
        (8, 11),
        (9, 10),
        (10, 11),
        (11, 12),
    ]
    for start, end in edges:
        draw.line(
            [points[start], points[end]],
            fill=(245, 249, 247, 154),
            width=4,
        )
    colors = (
        (67, 173, 126, 230),
        (66, 139, 202, 230),
        (220, 139, 77, 230),
        (192, 86, 102, 230),
    )
    for index, (x, y) in enumerate(points):
        radius = 19 if index not in {4, 8, 12} else 25
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(10, 18, 16, 190),
            outline=colors[index % len(colors)],
            width=6,
        )
        draw.ellipse(
            (x - 5, y - 5, x + 5, y + 5),
            fill=(246, 249, 248, 235),
        )
    canvas.alpha_composite(overlay)


def build_collage(frame_paths: list[Path], output: Path) -> None:
    if len(frame_paths) != len(MACRO_DOMAINS) * TILES_PER_DOMAIN:
        raise ValueError(f"Expected 32 frames, found {len(frame_paths)}")
    rng = random.Random(499)
    canvas = Image.new("RGBA", CANVAS_SIZE, "#0c1311")

    # Interleave the domains so every band samples the full benchmark landscape.
    ordered = [frame_paths[domain * TILES_PER_DOMAIN + variant] for variant in range(4) for domain in range(8)]
    row_specs = [
        (8, -18, 332),
        (7, 278, 330),
        (9, 570, 355),
        (8, 895, 345),
    ]
    cursor = 0
    for count, y, height in row_specs:
        geometry = row_geometry(rng, count, y, height)
        for slot in range(count):
            paste_tile(canvas, ordered[cursor], geometry[slot])
            cursor += 1

    add_graph_overlay(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output, quality=92, optimize=True, progressive=True)


def main() -> int:
    args = parse_args()
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    candidates = collect_candidates(args.manifest.resolve())
    selected = select_candidates(candidates)

    frames: list[Path] = []
    public_sources = []
    for index, item in enumerate(selected):
        timestamp = FRAME_TIMES[(index * 3 + item["sample_id"]) % len(FRAME_TIMES)]
        frame_name = (
            f"{index:02d}-{item['domain']}-{item['model'].lower().replace('.', '-')}"
            f"-{item['sample_id']:03d}.jpg"
        )
        frame_path = args.scratch.resolve() / frame_name
        extract_frame(ffmpeg, item["video"], frame_path, timestamp, args.force)
        frames.append(frame_path)
        public_sources.append(
            {
                "domain": item["macro"],
                "micro_domain": item["micro"],
                "model": item["model"],
                "sample_id": item["sample_id"],
                "timestamp_seconds": timestamp,
            }
        )

    build_collage(frames, args.output.resolve())
    args.sources_output.parent.mkdir(parents=True, exist_ok=True)
    args.sources_output.write_text(
        json.dumps(
            {
                "selection": "32 representative frames from the five commercial models",
                "models": sorted(COMMERCIAL_MODELS),
                "frames": public_sources,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Hero collage: {args.output.resolve()}")
    print(f"Source manifest: {args.sources_output.resolve()}")
    print(f"Model counts: {dict(Counter(item['model'] for item in selected))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
