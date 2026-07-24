from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from build_project_page_data import CASE_LIBRARY, REPO_DIR, load_json, model_slug, video_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export curated browser media for the project page.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_DIR / "data" / "results_manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_DIR / "docs" / "assets" / "videos" / "cases",
    )
    parser.add_argument(
        "--hero-output",
        type=Path,
        default=REPO_DIR / "docs" / "assets" / "domain_collage_v2.jpg",
    )
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def manifest_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    selected: dict[tuple[str, int], dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if row.get("complete", "").lower() == "true":
                selected[(row["model"], int(row["sample_id"]))] = row
    return selected


def resolve_ffmpeg(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    discovered = shutil.which("ffmpeg")
    if not discovered:
        raise RuntimeError("ffmpeg is required; pass --ffmpeg with its executable path")
    return Path(discovered)


def source_video(
    model: str,
    sample_number: int,
    rows: dict[tuple[str, int], dict[str, str]],
    indexes: dict[str, dict[str, Path]],
) -> Path:
    row = rows[(model, sample_number)]
    qa_payload = load_json(REPO_DIR / row["qa_file"])
    video_name = Path(str(qa_payload.get("video_file", ""))).name
    path = indexes[model].get(video_name)
    if path is None:
        raise FileNotFoundError(f"Could not match {model} sample {sample_number}: {video_name}")
    return path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def export_video(ffmpeg: Path, source: Path, target: Path, force: bool) -> None:
    if target.exists() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(ffmpeg),
            "-y" if force else "-n",
            "-i",
            str(source),
            "-vf",
            "scale='min(960,iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(target),
        ]
    )


def export_frame(ffmpeg: Path, source: Path, target: Path, timestamp: float, force: bool) -> None:
    if target.exists() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(ffmpeg),
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
            "scale='min(1280,iw)':-2",
            "-q:v",
            "3",
            str(target),
        ]
    )


def build_collage(frame_groups: list[list[Path]], output: Path) -> None:
    ordered = [group[variant] for variant in range(3) for group in frame_groups]
    width, height = 2560, 1200
    columns, rows = 6, 4
    gutter = 5
    tile_width = (width - gutter * (columns - 1)) // columns
    tile_height = (height - gutter * (rows - 1)) // rows
    canvas = Image.new("RGB", (width, height), "#07100d")
    for index, path in enumerate(ordered):
        image = Image.open(path).convert("RGB")
        tile = ImageOps.fit(image, (tile_width, tile_height), method=Image.Resampling.LANCZOS)
        x = (index % columns) * (tile_width + gutter)
        y = (index // columns) * (tile_height + gutter)
        canvas.paste(tile, (x, y))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    points = [
        (1540, 270),
        (1760, 190),
        (1900, 390),
        (2115, 250),
        (2290, 470),
        (1770, 610),
        (2020, 710),
        (2250, 830),
        (2420, 680),
    ]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 5), (3, 4), (3, 6), (5, 6), (6, 7), (4, 8), (7, 8)]
    for start, end in edges:
        draw.line([points[start], points[end]], fill=(236, 244, 240, 110), width=3)
    node_colors = [(68, 151, 108, 190), (54, 115, 170, 190), (197, 126, 75, 190)]
    for index, (x, y) in enumerate(points):
        radius = 16 if index not in {4, 8} else 22
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(8, 18, 14, 155),
            outline=node_colors[index % len(node_colors)],
            width=5,
        )
    for index in range(5):
        x = 1640 + index * 150
        y = 980
        draw.rounded_rectangle((x, y, x + 118, y + 70), radius=8, outline=(240, 246, 243, 130), width=3)
        if index < 4:
            draw.line((x + 121, y + 35, x + 143, y + 35), fill=(240, 246, 243, 130), width=3)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90, optimize=True, progressive=True)


def main() -> int:
    args = parse_args()
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    rows = manifest_rows(args.manifest.resolve())
    models = {model for case in CASE_LIBRARY for model in case["models"]}
    indexes = {model: video_index(model) for model in models}
    frame_groups: list[list[Path]] = []

    scratch = REPO_DIR / "tmp" / "project_page_hero_frames"
    for case in CASE_LIBRARY:
        case_frames: list[Path] = []
        sources: list[Path] = []
        for model in case["models"]:
            source = source_video(model, int(case["sample_number"]), rows, indexes)
            sources.append(source)
            stem = model_slug(model)
            public_video = args.output_root / case["id"] / f"{stem}.mp4"
            poster = args.output_root / case["id"] / f"{stem}.jpg"
            export_video(ffmpeg, source, public_video, args.force)
            export_frame(ffmpeg, source, poster, 1.0, args.force)
            case_frames.append(poster)
        late_frame = scratch / f"{case['id']}-late.jpg"
        export_frame(ffmpeg, sources[0], late_frame, 3.6, args.force)
        case_frames.append(late_frame)
        frame_groups.append(case_frames)

    build_collage(frame_groups, args.hero_output.resolve())
    print(f"Case media: {args.output_root.resolve()}")
    print(f"Hero collage: {args.hero_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
