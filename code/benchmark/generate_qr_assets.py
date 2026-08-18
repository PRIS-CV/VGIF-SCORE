"""Generate print-ready QR assets for the VGIF-Score public links."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import qrcode
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "assets" / "qrcodes"

INK = "#101828"
PAPER = "#FFFFFF"
CANVAS = "#F5F7FB"
MUTED = "#667085"


@dataclass(frozen=True)
class Link:
    slug: str
    label: str
    url: str
    display_url: str
    accent: str


LINKS = (
    Link(
        slug="paper",
        label="PAPER",
        url="https://arxiv.org/abs/2607.13527",
        display_url="arxiv.org/abs/2607.13527",
        accent="#B31B1B",
    ),
    Link(
        slug="dataset",
        label="DATASET",
        url="https://huggingface.co/datasets/Notyourkev/VGIF-Bench",
        display_url="huggingface.co/datasets/Notyourkev/VGIF-Bench",
        accent="#D99A00",
    ),
    Link(
        slug="code",
        label="CODE",
        url="https://github.com/PRIS-CV/VGIF-SCORE",
        display_url="github.com/PRIS-CV/VGIF-SCORE",
        accent="#7659A8",
    ),
    Link(
        slug="project",
        label="PROJECT",
        url="https://pris-cv.github.io/VGIF-SCORE/",
        display_url="pris-cv.github.io/VGIF-SCORE",
        accent="#2F75B5",
    ),
)


def make_qr_matrix(url: str) -> list[list[bool]]:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=16,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    return qr.get_matrix()


def matrix_path(matrix: list[list[bool]], x: int = 0, y: int = 0, scale: int = 1) -> str:
    commands: list[str] = []
    for row_index, row in enumerate(matrix):
        start: int | None = None
        for column_index, active in enumerate((*row, False)):
            if active and start is None:
                start = column_index
            elif not active and start is not None:
                width = column_index - start
                commands.append(
                    f"M{x + start * scale} {y + row_index * scale}"
                    f"h{width * scale}v{scale}h{-width * scale}z"
                )
                start = None
    return "".join(commands)


def write_qr_png(matrix: list[list[bool]], destination: Path, scale: int = 16) -> None:
    side = len(matrix) * scale
    image = Image.new("RGB", (side, side), PAPER)
    draw = ImageDraw.Draw(image)
    ink = (16, 24, 40)
    for row_index, row in enumerate(matrix):
        for column_index, active in enumerate(row):
            if active:
                x0 = column_index * scale
                y0 = row_index * scale
                draw.rectangle((x0, y0, x0 + scale - 1, y0 + scale - 1), fill=ink)
    image.save(destination, format="PNG", optimize=True)


def write_qr_svg(matrix: list[list[bool]], destination: Path, title: str) -> None:
    side = len(matrix)
    path = matrix_path(matrix)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side} {side}" role="img" aria-labelledby="title">
  <title id="title">{escape(title)}</title>
  <rect width="{side}" height="{side}" fill="{PAPER}"/>
  <path d="{path}" fill="{INK}" shape-rendering="crispEdges"/>
</svg>
"""
    destination.write_text(svg, encoding="utf-8", newline="\n")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    font_path = Path("C:/Windows/Fonts") / filename
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill) -> None:
    x, y = xy
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    draw.text((x - width / 2, y), text, font=font, fill=fill)


def draw_matrix_png(
    draw: ImageDraw.ImageDraw,
    matrix: list[list[bool]],
    x: int,
    y: int,
    scale: int,
) -> None:
    side = len(matrix) * scale
    draw.rectangle((x, y, x + side - 1, y + side - 1), fill=PAPER)
    for row_index, row in enumerate(matrix):
        for column_index, active in enumerate(row):
            if active:
                x0 = x + column_index * scale
                y0 = y + row_index * scale
                draw.rectangle((x0, y0, x0 + scale - 1, y0 + scale - 1), fill=INK)


def write_share_card_png(matrices: dict[str, list[list[bool]]], destination: Path) -> None:
    width, height = 1840, 650
    image = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(image)
    title_font = load_font(38, bold=True)
    subtitle_font = load_font(19)
    label_font = load_font(23, bold=True)
    url_font = load_font(15)

    centered_text(draw, (width // 2, 28), "VGIF-Score · Quick Access", title_font, INK)
    centered_text(
        draw,
        (width // 2, 84),
        "Paper · Benchmark Dataset · Source Code · Interactive Project",
        subtitle_font,
        MUTED,
    )

    margin, gap, card_y, card_height = 50, 20, 130, 470
    card_width = (width - 2 * margin - 3 * gap) // 4

    for index, link in enumerate(LINKS):
        matrix = matrices[link.slug]
        x = margin + index * (card_width + gap)
        draw.rounded_rectangle(
            (x, card_y, x + card_width, card_y + card_height),
            radius=22,
            fill=PAPER,
            outline="#E4E7EC",
            width=2,
        )
        draw.rounded_rectangle(
            (x, card_y, x + card_width, card_y + 14),
            radius=7,
            fill=link.accent,
        )

        scale = 5
        qr_side = len(matrix) * scale
        qr_x = x + (card_width - qr_side) // 2
        qr_y = card_y + 42
        draw_matrix_png(draw, matrix, qr_x, qr_y, scale)

        centered_text(
            draw,
            (x + card_width // 2, card_y + 355),
            link.label,
            label_font,
            link.accent,
        )
        centered_text(
            draw,
            (x + card_width // 2, card_y + 402),
            link.display_url,
            url_font,
            MUTED,
        )

    centered_text(
        draw,
        (width // 2, 617),
        "VGIF-Score: Interpretable and Diagnostic Evaluation of Spatio-Temporal Instruction Following",
        url_font,
        MUTED,
    )
    image.save(destination, format="PNG", optimize=True)


def write_share_card_svg(matrices: dict[str, list[list[bool]]], destination: Path) -> None:
    width, height = 1840, 650
    margin, gap, card_y, card_height = 50, 20, 130, 470
    card_width = (width - 2 * margin - 3 * gap) // 4
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">VGIF-Score quick access QR codes</title>',
        '<desc id="desc">QR codes for the paper, benchmark dataset, source code, and interactive project page.</desc>',
        f'<rect width="{width}" height="{height}" fill="{CANVAS}"/>',
        f'<text x="{width / 2}" y="66" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="38" font-weight="700" fill="{INK}">VGIF-Score · Quick Access</text>',
        f'<text x="{width / 2}" y="105" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="19" fill="{MUTED}">Paper · Benchmark Dataset · Source Code · Interactive Project</text>',
    ]

    for index, link in enumerate(LINKS):
        matrix = matrices[link.slug]
        x = margin + index * (card_width + gap)
        scale = 5
        qr_side = len(matrix) * scale
        qr_x = x + (card_width - qr_side) // 2
        qr_y = card_y + 42
        path = matrix_path(matrix, qr_x, qr_y, scale)
        pieces.extend(
            (
                f'<rect x="{x}" y="{card_y}" width="{card_width}" height="{card_height}" rx="22" fill="{PAPER}" stroke="#E4E7EC" stroke-width="2"/>',
                f'<path d="M{x + 7} {card_y}H{x + card_width - 7}Q{x + card_width} {card_y} {x + card_width} {card_y + 7}V{card_y + 14}H{x}V{card_y + 7}Q{x} {card_y} {x + 7} {card_y}Z" fill="{link.accent}"/>',
                f'<rect x="{qr_x}" y="{qr_y}" width="{qr_side}" height="{qr_side}" fill="{PAPER}"/>',
                f'<path d="{path}" fill="{INK}" shape-rendering="crispEdges"/>',
                f'<text x="{x + card_width / 2}" y="{card_y + 380}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="23" font-weight="700" fill="{link.accent}">{link.label}</text>',
                f'<text x="{x + card_width / 2}" y="{card_y + 425}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="15" fill="{MUTED}">{escape(link.display_url)}</text>',
            )
        )

    pieces.extend(
        (
            f'<text x="{width / 2}" y="635" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="15" fill="{MUTED}">VGIF-Score: Interpretable and Diagnostic Evaluation of Spatio-Temporal Instruction Following</text>',
            "</svg>",
        )
    )
    destination.write_text("\n".join(pieces) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrices = {link.slug: make_qr_matrix(link.url) for link in LINKS}

    for link in LINKS:
        matrix = matrices[link.slug]
        write_qr_png(matrix, OUTPUT_DIR / f"{link.slug}-qr.png")
        write_qr_svg(matrix, OUTPUT_DIR / f"{link.slug}-qr.svg", f"{link.label}: {link.url}")

    write_share_card_png(matrices, OUTPUT_DIR / "vgif-score-links.png")
    write_share_card_svg(matrices, OUTPUT_DIR / "vgif-score-links.svg")

    for link in LINKS:
        print(f"{link.slug:8s} {len(matrices[link.slug]):2d}x{len(matrices[link.slug]):2d}  {link.url}")
    print(f"Wrote QR assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
