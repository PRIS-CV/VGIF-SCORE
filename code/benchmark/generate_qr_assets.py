"""Generate print-ready and branded QR assets for VGIF-Score."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import qrcode
from PIL import Image, ImageDraw
from resvg_py import svg_to_bytes


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "assets" / "qrcodes"

INK = "#101828"
PAPER = "#FFFFFF"
CANVAS = "#F5F7FB"
MUTED = "#667085"
FONT = "Segoe UI,Arial,sans-serif"

# arXiv, Hugging Face, and GitHub paths are from Simple Icons (CC0-1.0).
ARXIV_ICON = (
    "M3.8423 0a1.0037 1.0037 0 0 0-.922.6078c-.1536.3687-.0438.6275.2938 1.1113l6.9185 8.3597-1.0223 1.1058a1.0393 1.0393 0 0 0 .003 1.4229l1.2292 1.3135-5.4391 6.4444c-.2803.299-.4538.823-.2971 1.1986a1.0253 1.0253 0 0 0 .9585.635.9133.9133 0 0 0 .6891-.3405l5.783-6.126 7.4902 8.0051a.8527.8527 0 0 0 .6835.2597.9575.9575 0 0 0 .8777-.6138c.1577-.377-.017-.7502-.306-1.1407l-7.0518-8.3418 1.0632-1.13a.9626.9626 0 0 0 .0089-1.3165L4.6336.4639s-.3733-.4535-.768-.463zm0 .272h.0166c.2179.0052.4874.2715.5644.3639l.005.006.0052.0055 10.169 10.9905a.6915.6915 0 0 1-.0072.945l-1.0666 1.133-1.4982-1.7724-8.5994-10.39c-.3286-.472-.352-.6183-.2592-.841a.7307.7307 0 0 1 .6704-.4401Zm14.341 1.5701a.877.877 0 0 0-.6554.2418l-5.6962 6.1584 1.6944 1.8319 5.3089-6.5138c.3251-.4335.479-.6603.3247-1.0292a1.1205 1.1205 0 0 0-.9763-.689zm-7.6557 12.2823 1.3186 1.4135-5.7864 6.1295a.6494.6494 0 0 1-.4959.26.7516.7516 0 0 1-.706-.4669c-.1119-.2682.0359-.6864.2442-.9083l.0051-.0055.0047-.0055z"
)
HUGGING_FACE_ICON = (
    "M12.025 1.13c-5.77 0-10.449 4.647-10.449 10.378 0 1.112.178 2.181.503 3.185.064-.222.203-.444.416-.577a.96.96 0 0 1 .524-.15c.293 0 .584.124.84.284.278.173.48.408.71.694.226.282.458.611.684.951v-.014c.017-.324.106-.622.264-.874s.403-.487.762-.543c.3-.047.596.06.787.203s.31.313.4.467c.15.257.212.468.233.542.01.026.653 1.552 1.657 2.54.616.605 1.01 1.223 1.082 1.912.055.537-.096 1.059-.38 1.572.637.121 1.294.187 1.967.187.657 0 1.298-.063 1.921-.178-.287-.517-.44-1.041-.384-1.581.07-.69.465-1.307 1.081-1.913 1.004-.987 1.647-2.513 1.657-2.539.021-.074.083-.285.233-.542.09-.154.208-.323.4-.467a1.08 1.08 0 0 1 .787-.203c.359.056.604.29.762.543s.247.55.265.874v.015c.225-.34.457-.67.683-.952.23-.286.432-.52.71-.694.257-.16.547-.284.84-.285a.97.97 0 0 1 .524.151c.228.143.373.388.43.625l.006.04a10.3 10.3 0 0 0 .534-3.273c0-5.731-4.678-10.378-10.449-10.378M8.327 6.583a1.5 1.5 0 0 1 .713.174 1.487 1.487 0 0 1 .617 2.013c-.183.343-.762-.214-1.102-.094-.38.134-.532.914-.917.71a1.487 1.487 0 0 1 .69-2.803m7.486 0a1.487 1.487 0 0 1 .689 2.803c-.385.204-.536-.576-.916-.71-.34-.12-.92.437-1.103.094a1.487 1.487 0 0 1 .617-2.013 1.5 1.5 0 0 1 .713-.174m-10.68 1.55a.96.96 0 1 1 0 1.921.96.96 0 0 1 0-1.92m13.838 0a.96.96 0 1 1 0 1.92.96.96 0 0 1 0-1.92M8.489 11.458c.588.01 1.965 1.157 3.572 1.164 1.607-.007 2.984-1.155 3.572-1.164.196-.003.305.12.305.454 0 .886-.424 2.328-1.563 3.202-.22-.756-1.396-1.366-1.63-1.32q-.011.001-.02.006l-.044.026-.01.008-.03.024q-.018.017-.035.036l-.032.04a1 1 0 0 0-.058.09l-.014.025q-.049.088-.11.19a1 1 0 0 1-.083.116 1.2 1.2 0 0 1-.173.18q-.035.029-.075.058a1.3 1.3 0 0 1-.251-.243 1 1 0 0 1-.076-.107c-.124-.193-.177-.363-.337-.444-.034-.016-.104-.008-.2.022q-.094.03-.216.087-.06.028-.125.063l-.13.074q-.067.04-.136.086a3 3 0 0 0-.135.096 3 3 0 0 0-.26.219 2 2 0 0 0-.12.121 2 2 0 0 0-.106.128l-.002.002a2 2 0 0 0-.09.132l-.001.001a1.2 1.2 0 0 0-.105.212q-.013.036-.024.073c-1.139-.875-1.563-2.317-1.563-3.203 0-.334.109-.457.305-.454m.836 10.354c.824-1.19.766-2.082-.365-3.194-1.13-1.112-1.789-2.738-1.789-2.738s-.246-.945-.806-.858-.97 1.499.202 2.362c1.173.864-.233 1.45-.685.64-.45-.812-1.683-2.896-2.322-3.295s-1.089-.175-.938.647 2.822 2.813 2.562 3.244-1.176-.506-1.176-.506-2.866-2.567-3.49-1.898.473 1.23 2.037 2.16c1.564.932 1.686 1.178 1.464 1.53s-3.675-2.511-4-1.297c-.323 1.214 3.524 1.567 3.287 2.405-.238.839-2.71-1.587-3.216-.642-.506.946 3.49 2.056 3.522 2.064 1.29.33 4.568 1.028 5.713-.624m5.349 0c-.824-1.19-.766-2.082.365-3.194 1.13-1.112 1.789-2.738 1.789-2.738s.246-.945.806-.858.97 1.499-.202 2.362c-1.173.864.233 1.45.685.64.451-.812 1.683-2.896 2.322-3.295s1.089-.175.938.647-2.822 2.813-2.562 3.244 1.176-.506 1.176-.506 2.866-2.567 3.49-1.898-.473 1.23-2.037 2.16c-1.564.932-1.686 1.178-1.464 1.53s3.675-2.511 4-1.297c.323 1.214-3.524 1.567-3.287 2.405.238.839 2.71-1.587 3.216-.642.506.946-3.49 2.056-3.522 2.064-1.29.33-4.568 1.028-5.713-.624"
)
GITHUB_ICON = (
    "M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"
)
HOME_ICON = "M12 2.6 1.8 11l1.9 2.3 1.3-1.1V22h5.2v-6h3.6v6H19v-9.8l1.3 1.1 1.9-2.3z"


@dataclass(frozen=True)
class Link:
    slug: str
    label: str
    brand: str
    action: str
    url: str
    display_url: str
    accent: str
    icon_path: str


LINKS = (
    Link(
        slug="paper",
        label="PAPER",
        brand="arXiv",
        action="SCAN TO READ THE PAPER",
        url="https://arxiv.org/abs/2607.13527",
        display_url="arxiv.org/abs/2607.13527",
        accent="#B31B1B",
        icon_path=ARXIV_ICON,
    ),
    Link(
        slug="dataset",
        label="DATASET",
        brand="Hugging Face",
        action="SCAN TO OPEN THE DATASET",
        url="https://huggingface.co/datasets/Notyourkev/VGIF-Bench",
        display_url="huggingface.co/datasets/Notyourkev/VGIF-Bench",
        accent="#D99A00",
        icon_path=HUGGING_FACE_ICON,
    ),
    Link(
        slug="code",
        label="SOURCE CODE",
        brand="GitHub",
        action="SCAN TO VIEW THE SOURCE CODE",
        url="https://github.com/PRIS-CV/VGIF-SCORE",
        display_url="github.com/PRIS-CV/VGIF-SCORE",
        accent="#181717",
        icon_path=GITHUB_ICON,
    ),
    Link(
        slug="project",
        label="PROJECT PAGE",
        brand="VGIF-Score",
        action="SCAN TO VISIT THE PROJECT PAGE",
        url="https://pris-cv.github.io/VGIF-SCORE/",
        display_url="pris-cv.github.io/VGIF-SCORE",
        accent="#2F75B5",
        icon_path=HOME_ICON,
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
    for row_index, row in enumerate(matrix):
        for column_index, active in enumerate(row):
            if active:
                x0 = column_index * scale
                y0 = row_index * scale
                draw.rectangle((x0, y0, x0 + scale - 1, y0 + scale - 1), fill=INK)
    image.save(destination, format="PNG", optimize=True)


def write_qr_svg(matrix: list[list[bool]], destination: Path, title: str) -> None:
    side = len(matrix)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side} {side}" role="img" aria-labelledby="title">
  <title id="title">{escape(title)}</title>
  <rect width="{side}" height="{side}" fill="{PAPER}"/>
  <path d="{matrix_path(matrix)}" fill="{INK}" shape-rendering="crispEdges"/>
</svg>
"""
    destination.write_text(svg, encoding="utf-8", newline="\n")


def brand_icon(link: Link, center_x: float, y: int, size: int) -> str:
    x = center_x - size / 2
    circle_y = y + size / 2
    scale = size / 24
    return (
        f'<circle cx="{center_x}" cy="{circle_y}" r="{size / 2 + 10}" '
        f'fill="{link.accent}" opacity="0.10"/>'
        f'<g transform="translate({x} {y}) scale({scale})" fill="{link.accent}">'
        f'<path d="{link.icon_path}"/></g>'
    )


def qr_svg(matrix: list[list[bool]], center_x: float, y: int, target_side: int) -> tuple[str, int]:
    scale = max(1, round(target_side / len(matrix)))
    side = len(matrix) * scale
    x = round(center_x - side / 2)
    return (
        f'<rect x="{x}" y="{y}" width="{side}" height="{side}" fill="{PAPER}"/>'
        f'<path d="{matrix_path(matrix, x, y, scale)}" fill="{INK}" '
        f'shape-rendering="crispEdges"/>',
        side,
    )


def branded_card_svg(link: Link, matrix: list[list[bool]]) -> str:
    width, height = 760, 1060
    center = width / 2
    qr, qr_side = qr_svg(matrix, center, 220, 580)
    label_y = 220 + qr_side + 62
    return "\n".join(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(link.brand)} QR code for {escape(link.label.lower())}</title>',
            f'<desc id="desc">{escape(link.action.title())}: {escape(link.url)}</desc>',
            f'<rect width="{width}" height="{height}" fill="{CANVAS}"/>',
            f'<rect x="20" y="20" width="720" height="1020" rx="34" fill="{PAPER}" stroke="#E4E7EC" stroke-width="2"/>',
            f'<path d="M54 20H706Q740 20 740 54V70H20V54Q20 20 54 20Z" fill="{link.accent}"/>',
            brand_icon(link, center, 84, 62),
            f'<text x="{center}" y="185" text-anchor="middle" font-family="{FONT}" font-size="30" font-weight="700" fill="{INK}">{escape(link.brand)}</text>',
            qr,
            f'<text x="{center}" y="{label_y}" text-anchor="middle" font-family="{FONT}" font-size="24" font-weight="700" letter-spacing="1.4" fill="{link.accent}">{escape(link.action)}</text>',
            f'<text x="{center}" y="{label_y + 48}" text-anchor="middle" font-family="{FONT}" font-size="18" fill="{MUTED}">{escape(link.display_url)}</text>',
            f'<line x1="130" y1="972" x2="630" y2="972" stroke="#E4E7EC"/>',
            f'<text x="{center}" y="1015" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{MUTED}">VGIF-Score · Spatio-Temporal Instruction Following</text>',
            "</svg>",
        )
    ) + "\n"


def share_card_svg(matrices: dict[str, list[list[bool]]]) -> str:
    width, height = 1840, 720
    margin, gap, card_y, card_height = 50, 20, 130, 535
    card_width = (width - 2 * margin - 3 * gap) // 4
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">VGIF-Score branded quick access QR codes</title>',
        '<desc id="desc">Branded QR codes for arXiv, Hugging Face, GitHub, and the VGIF-Score project page.</desc>',
        f'<rect width="{width}" height="{height}" fill="{CANVAS}"/>',
        f'<text x="{width / 2}" y="64" text-anchor="middle" font-family="{FONT}" font-size="38" font-weight="700" fill="{INK}">VGIF-Score · Quick Access</text>',
        f'<text x="{width / 2}" y="103" text-anchor="middle" font-family="{FONT}" font-size="19" fill="{MUTED}">Paper · Benchmark Dataset · Source Code · Interactive Project</text>',
    ]

    for index, link in enumerate(LINKS):
        matrix = matrices[link.slug]
        x = margin + index * (card_width + gap)
        center = x + card_width / 2
        qr, _ = qr_svg(matrix, center, card_y + 132, 245)
        pieces.extend(
            (
                f'<rect x="{x}" y="{card_y}" width="{card_width}" height="{card_height}" rx="22" fill="{PAPER}" stroke="#E4E7EC" stroke-width="2"/>',
                f'<path d="M{x + 7} {card_y}H{x + card_width - 7}Q{x + card_width} {card_y} {x + card_width} {card_y + 7}V{card_y + 14}H{x}V{card_y + 7}Q{x} {card_y} {x + 7} {card_y}Z" fill="{link.accent}"/>',
                brand_icon(link, center, card_y + 36, 42),
                f'<text x="{center}" y="{card_y + 112}" text-anchor="middle" font-family="{FONT}" font-size="22" font-weight="700" fill="{INK}">{escape(link.brand)}</text>',
                qr,
                f'<text x="{center}" y="{card_y + 445}" text-anchor="middle" font-family="{FONT}" font-size="21" font-weight="700" letter-spacing="1" fill="{link.accent}">{escape(link.label)}</text>',
                f'<text x="{center}" y="{card_y + 486}" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{MUTED}">{escape(link.display_url)}</text>',
            )
        )

    pieces.extend(
        (
            f'<text x="{width / 2}" y="701" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{MUTED}">VGIF-Score: Interpretable and Diagnostic Evaluation of Spatio-Temporal Instruction Following</text>',
            "</svg>",
        )
    )
    return "\n".join(pieces) + "\n"


def write_svg_and_png(svg: str, svg_path: Path, png_path: Path, width: int, height: int) -> None:
    svg_path.write_text(svg, encoding="utf-8", newline="\n")
    png_path.write_bytes(
        svg_to_bytes(
            svg_string=svg,
            width=width,
            height=height,
            font_dirs=["C:/Windows/Fonts"],
            shape_rendering="crisp_edges",
            text_rendering="optimize_legibility",
        )
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrices = {link.slug: make_qr_matrix(link.url) for link in LINKS}

    for link in LINKS:
        matrix = matrices[link.slug]
        write_qr_png(matrix, OUTPUT_DIR / f"{link.slug}-qr.png")
        write_qr_svg(matrix, OUTPUT_DIR / f"{link.slug}-qr.svg", f"{link.label}: {link.url}")
        write_svg_and_png(
            branded_card_svg(link, matrix),
            OUTPUT_DIR / f"{link.slug}-card.svg",
            OUTPUT_DIR / f"{link.slug}-card.png",
            760,
            1060,
        )

    write_svg_and_png(
        share_card_svg(matrices),
        OUTPUT_DIR / "vgif-score-links.svg",
        OUTPUT_DIR / "vgif-score-links.png",
        1840,
        720,
    )

    for link in LINKS:
        print(
            f"{link.slug:8s} {len(matrices[link.slug]):2d}x{len(matrices[link.slug]):2d}  "
            f"{link.brand:12s} {link.url}"
        )
    print(f"Wrote QR assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
