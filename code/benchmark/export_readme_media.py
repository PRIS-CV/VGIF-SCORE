from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


REPO_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_DIR / "docs" / "assets" / "readme"
ASSETS = {
    "hero.jpg": REPO_DIR / "docs" / "assets" / "domain_collage_v2.jpg",
    "pipeline.png": REPO_DIR / "docs" / "assets" / "vgif_pipeline_final.png",
    "benchmark.png": REPO_DIR / "docs" / "assets" / "prompt_statistics.png",
    "diagnosis.png": REPO_DIR / "docs" / "assets" / "fig4_Dependency-aware_causal_chain_diagnosis.png",
}


def export_matted_image(source: Path, target: Path, padding: int = 42) -> None:
    image = Image.open(source).convert("RGBA")
    white = Image.new("RGBA", image.size, "white")
    white.alpha_composite(image)

    canvas = Image.new(
        "RGB",
        (white.width + padding * 2, white.height + padding * 2),
        "white",
    )
    canvas.paste(white.convert("RGB"), (padding, padding))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (padding - 1, padding - 1, padding + white.width, padding + white.height),
        outline="#d9dedb",
        width=2,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".jpg":
        canvas.save(target, quality=91, optimize=True, progressive=True)
    else:
        canvas.save(target, optimize=True)


def main() -> int:
    for filename, source in ASSETS.items():
        target = OUTPUT_DIR / filename
        export_matted_image(source, target)
        print(target.relative_to(REPO_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
