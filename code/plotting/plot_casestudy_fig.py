"""
Publication-style case study figure for Section 5.5.

The figure follows the visual language of many AI conference diagnostics:
prompt decomposition, dashed annotation panels, compact scene sketches, and
green/red/gray status traces for a causal chain.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.06,
    }
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "scripts" / "output_figs"

GOOD_MODEL = "Kling-V3"
BAD_MODEL = "CogVideoX-1.5"
GOOD_FILE = (
    ROOT
    / "data"
    / "clean_qa"
    / GOOD_MODEL
    / "219_creative-surreal-expression_874797542247518254_1_qa_eval_dependency_rounds.json"
)
BAD_FILE = ROOT / "data" / "clean_qa" / BAD_MODEL / "CogVideoX-219_qa_eval_dependency_rounds.json"


COL = {
    "ink": "#232323",
    "muted": "#676767",
    "line": "#A7A7A0",
    "paper": "#FFFFFF",
    "panel": "#FBFAF6",
    "green": "#70B77E",
    "green_dark": "#3C8C52",
    "green_light": "#E8F5EA",
    "red": "#D95A55",
    "red_dark": "#B3302E",
    "red_light": "#FCE8E6",
    "gray": "#BDBDBD",
    "gray_light": "#F1F1EE",
    "blue": "#3B66B1",
    "blue_light": "#E9F0FF",
    "purple": "#8A59B6",
    "purple_light": "#F1E8FA",
    "orange": "#E88A2A",
    "orange_light": "#FFF0DC",
    "teal": "#1E9E9A",
    "teal_light": "#DDF3F1",
    "yellow": "#F3C766",
}


STAGES = [
    ("q1-q6", "Scene setup", "bottle, mirror,\nglove, gold key", "local"),
    ("q7-q8", "Mist ring", "spray forms\nfloating ring", "state"),
    ("q9", "Trigger action", "key rotates\nthrough ring", "trigger"),
    ("q10", "Color change", "liquid turns\n deep teal", "causal"),
    ("q11", "Reflection", "teal seeps into\nmirror image", "state"),
    ("q12", "Glove applause", "reflection causes\nglove motion", "causal"),
]


def load_case(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["by_id"] = {r["id"]: r for r in data["results"]}
    return data


def rounded(ax, x, y, w, h, fc, ec, lw=1.0, ls="-", radius=0.012, z=1):
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def pill(ax, x, y, w, h, text, fc, ec, color=None, fs=8, weight="bold", ls="-"):
    rounded(ax, x, y, w, h, fc, ec, lw=1.0, ls=ls, radius=h / 2.2)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color or COL["ink"],
        fontweight=weight,
    )


def arrow(ax, x1, y1, x2, y2, color, lw=1.7, rad=0.0, style="->", ls="-"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=6,
    )


def draw_key(ax, x, y, s, color, lw=1.8):
    ax.add_patch(patches.Circle((x, y), s * 0.13, fill=False, edgecolor=color, linewidth=lw))
    ax.plot([x + s * 0.13, x + s * 0.46], [y, y], color=color, lw=lw, solid_capstyle="round")
    ax.plot([x + s * 0.36, x + s * 0.36], [y, y - s * 0.11], color=color, lw=lw)
    ax.plot([x + s * 0.45, x + s * 0.45], [y, y - s * 0.08], color=color, lw=lw)


def draw_bottle(ax, x, y, s, liquid=None, alpha=1.0):
    glass = patches.FancyBboxPatch(
        (x - s * 0.16, y - s * 0.22),
        s * 0.32,
        s * 0.46,
        boxstyle=f"round,pad=0.002,rounding_size={s * 0.035}",
        facecolor="#F8FBFF",
        edgecolor="#384252",
        linewidth=1.2,
        alpha=alpha,
        zorder=4,
    )
    ax.add_patch(glass)
    if liquid:
        ax.add_patch(
            patches.Rectangle(
                (x - s * 0.13, y - s * 0.19),
                s * 0.26,
                s * 0.24,
                facecolor=liquid,
                edgecolor="none",
                alpha=0.78 * alpha,
                zorder=5,
            )
        )
    ax.add_patch(
        patches.Rectangle(
            (x - s * 0.055, y + s * 0.24),
            s * 0.11,
            s * 0.11,
            facecolor="#D8D8D8",
            edgecolor="#384252",
            linewidth=1.0,
            alpha=alpha,
            zorder=5,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (x - s * 0.10, y + s * 0.35),
            s * 0.20,
            s * 0.035,
            facecolor="#D8D8D8",
            edgecolor="#384252",
            linewidth=1.0,
            alpha=alpha,
            zorder=5,
        )
    )


def draw_glove(ax, x, y, s, active=False, alpha=1.0):
    edge = "#454545"
    face = "#EFE7DD" if alpha > 0.8 else "#D6D6D6"
    ax.add_patch(patches.Ellipse((x, y), s * 0.35, s * 0.22, facecolor=face, edgecolor=edge, lw=1.0, alpha=alpha))
    for i in range(4):
        ax.add_patch(
            patches.Ellipse(
                (x - s * 0.13 + i * s * 0.08, y + s * 0.12),
                s * 0.055,
                s * 0.20,
                angle=-10 + i * 6,
                facecolor=face,
                edgecolor=edge,
                lw=0.8,
                alpha=alpha,
            )
        )
    if active:
        ax.plot([x - s * 0.26, x - s * 0.36], [y + s * 0.06, y + s * 0.18], color=COL["orange"], lw=1.2)
        ax.plot([x + s * 0.26, x + s * 0.36], [y + s * 0.06, y + s * 0.18], color=COL["orange"], lw=1.2)


def draw_stage_icon(ax, x, y, w, h, kind, status):
    alpha = 0.45 if status == "blocked" else 1.0
    cx = x + w / 2
    cy = y + h * 0.62
    s = min(w, h) * 0.72
    ax.add_patch(
        patches.Ellipse(
            (cx, cy - s * 0.20),
            s * 0.92,
            s * 0.22,
            facecolor="#E7E7E0",
            edgecolor="#777777",
            lw=0.8,
            alpha=alpha,
            zorder=1,
        )
    )

    if kind == "local":
        draw_bottle(ax, cx - s * 0.12, cy, s * 0.78, liquid="#EAF4FF", alpha=alpha)
        draw_glove(ax, cx + s * 0.23, cy - s * 0.15, s * 0.72, active=False, alpha=alpha)
        draw_key(ax, cx + s * 0.04, cy + s * 0.16, s * 0.74, COL["yellow"], lw=1.7)
    elif kind == "state":
        draw_bottle(ax, cx - s * 0.12, cy - s * 0.03, s * 0.78, liquid="#EAF4FF", alpha=alpha)
        ax.add_patch(patches.Ellipse((cx + s * 0.18, cy + s * 0.19), s * 0.42, s * 0.25, fill=False, edgecolor=COL["teal"], lw=2.0, alpha=alpha))
        ax.plot([cx - s * 0.04, cx + s * 0.09], [cy + s * 0.31, cy + s * 0.25], color=COL["teal"], lw=1.4, alpha=alpha)
    elif kind == "trigger":
        ax.add_patch(patches.Ellipse((cx + s * 0.10, cy + s * 0.10), s * 0.48, s * 0.27, fill=False, edgecolor=COL["teal"], lw=1.8, alpha=alpha))
        if status == "miss":
            draw_key(ax, cx - s * 0.30, cy - s * 0.08, s * 0.85, COL["red_dark"], lw=1.9)
            ax.plot([cx + s * 0.02, cx + s * 0.19], [cy + s * 0.02, cy + s * 0.18], color=COL["red_dark"], lw=2.0)
            ax.plot([cx + s * 0.19, cx + s * 0.02], [cy + s * 0.02, cy + s * 0.18], color=COL["red_dark"], lw=2.0)
        else:
            draw_key(ax, cx - s * 0.08, cy + s * 0.08, s * 0.82, COL["yellow"], lw=1.9)
            arrow(ax, cx - s * 0.22, cy + s * 0.00, cx + s * 0.23, cy + s * 0.12, COL["purple"], lw=1.2, rad=-0.18)
def draw_stage_specific(ax, x, y, w, h, stage_name, role, status):
    if stage_name == "Scene setup":
        draw_stage_icon(ax, x, y, w, h, "local", status)
    elif stage_name == "Mist ring":
        draw_stage_icon(ax, x, y, w, h, "state", status)
    elif stage_name == "Trigger action":
        draw_stage_icon(ax, x, y, w, h, "trigger", status)
    elif stage_name == "Color change":
        alpha = 0.45 if status == "blocked" else 1.0
        cx = x + w / 2
        cy = y + h * 0.62
        s = min(w, h) * 0.72
        draw_bottle(ax, cx, cy, s * 0.92, liquid=COL["teal"] if status == "ok" else "#C8C8C8", alpha=alpha)
        if status == "blocked":
            ax.text(cx, cy - s * 0.35, "no color\nshift", ha="center", va="center", fontsize=5.2, color="white", fontweight="bold")
    elif stage_name == "Reflection":
        alpha = 0.45 if status == "blocked" else 1.0
        cx = x + w / 2
        cy = y + h * 0.62
        s = min(w, h) * 0.72
        ax.add_patch(patches.Ellipse((cx, cy - s * 0.05), s * 0.85, s * 0.48, facecolor="#E8E8E8", edgecolor="#545454", lw=1.0, alpha=alpha))
        if status == "ok":
            ax.plot([cx - s * 0.30, cx - s * 0.07, cx + s * 0.23], [cy - s * 0.04, cy + s * 0.06, cy - s * 0.02], color=COL["teal"], lw=2.2, alpha=alpha)
        else:
            ax.plot([cx - s * 0.30, cx + s * 0.28], [cy - s * 0.05, cy - s * 0.05], color="#999999", lw=1.8, alpha=alpha)
    elif stage_name == "Glove applause":
        cx = x + w / 2
        cy = y + h * 0.62
        s = min(w, h) * 0.72
        draw_glove(ax, cx, cy, s * 1.05, active=status == "ok", alpha=0.45 if status == "blocked" else 1.0)


def stage_status(case: dict, ids: list[str]) -> str:
    rows = [case["by_id"][qid] for qid in ids if qid in case["by_id"]]
    if rows and all(bool(r["correct"]) for r in rows):
        return "ok"
    if rows and not all(bool(r.get("dependency_passed", True)) for r in rows):
        return "blocked"
    return "miss"


def draw_stage(ax, x, y, w, h, stage, status):
    qid, title, detail, role = stage
    face = {"ok": COL["green_light"], "miss": COL["red_light"], "blocked": COL["gray_light"]}[status]
    edge = {"ok": COL["green_dark"], "miss": COL["red_dark"], "blocked": "#777777"}[status]
    if role == "causal":
        edge = COL["blue"]
    elif role == "trigger":
        edge = COL["purple"]

    rounded(ax, x, y, w, h, face, edge, lw=1.8 if role in {"causal", "trigger"} else 1.2, radius=0.012)
    draw_stage_specific(ax, x, y, w, h, title, role, status)
    status_text = {"ok": "OK", "miss": "MISS", "blocked": "BLOCKED"}[status]
    color = {"ok": COL["green_dark"], "miss": COL["red_dark"], "blocked": COL["muted"]}[status]
    ax.text(x + 0.010, y + h - 0.018, qid, ha="left", va="top", fontsize=7.6, fontweight="bold", color=edge)
    ax.text(x + w / 2, y + 0.040, title, ha="center", va="center", fontsize=7.1, fontweight="bold", color=COL["ink"])
    ax.text(x + w / 2, y + 0.020, detail, ha="center", va="center", fontsize=5.8, color=COL["muted"], linespacing=1.0)
    ax.text(x + w - 0.010, y + h - 0.018, status_text, ha="right", va="top", fontsize=6.2, fontweight="bold", color=color)


def draw_prompt_panel(ax, prompt: str):
    rounded(ax, 0.035, 0.815, 0.93, 0.110, COL["paper"], COL["line"], lw=1.15, ls=(0, (4, 3)), radius=0.018)
    ax.text(0.055, 0.900, "Case study prompt", fontsize=10.2, fontweight="bold", color=COL["ink"], va="center")
    ax.text(0.190, 0.900, "sample 219 | Creative & Surreal Expression", fontsize=8.2, color=COL["muted"], va="center")
    short = (
        "A clear perfume bottle on a mirrored plinth sprays mist into a floating ring; "
        "when a gold key rotates through the ring, the liquid turns deep teal, the reflection changes, "
        "and a silk glove animates to applaud the bottle."
    )
    ax.text(0.055, 0.858, textwrap.fill(short, 150), fontsize=8.0, color=COL["ink"], va="center")
    pill(ax, 0.795, 0.881, 0.073, 0.024, "Objects", COL["orange_light"], COL["orange"], fs=7.0)
    pill(ax, 0.874, 0.881, 0.078, 0.024, "Causal QA", COL["blue_light"], COL["blue"], fs=7.0)


def draw_decomposition(ax):
    rounded(ax, 0.035, 0.605, 0.93, 0.175, COL["panel"], COL["line"], lw=1.15, ls=(0, (4, 3)), radius=0.018)
    ax.text(0.055, 0.754, "Prompt decomposition into ST-DAG constraints", fontsize=10.0, fontweight="bold", color=COL["ink"], va="center")
    cols = [
        ("Objects", COL["orange"], COL["orange_light"], ["perfume bottle", "mirrored plinth", "silk glove", "gold key"]),
        ("Attributes", COL["blue"], COL["blue_light"], ["clear bottle", "gold key", "deep teal liquid"]),
        ("Actions / states", COL["purple"], COL["purple_light"], ["spray mist", "floating ring", "key rotates", "mirror affected"]),
        ("Causal links", COL["red_dark"], COL["red_light"], ["q9 -> q10: key triggers teal liquid", "q11 -> q12: reflection triggers applause"]),
    ]
    x0, w, gap = 0.055, 0.205, 0.022
    for i, (name, color, light, items) in enumerate(cols):
        x = x0 + i * (w + gap)
        rounded(ax, x, 0.630, w, 0.098, COL["paper"], "#D5D5CF", lw=0.9, radius=0.012)
        pill(ax, x + 0.010, 0.708, min(0.100, w - 0.020), 0.024, name, light, color, color=color, fs=7.0)
        for j, item in enumerate(items):
            ax.text(x + 0.016, 0.692 - j * 0.017, item, ha="left", va="center", fontsize=6.6, color=COL["ink"])
    for x in [0.278, 0.505, 0.732]:
        ax.plot([x, x], [0.625, 0.730], color="#D9D9D3", lw=0.8, ls=(0, (3, 3)))


def draw_score_card(ax, x, y, w, h, case, model, status):
    edge = COL["green_dark"] if status == "good" else COL["red_dark"]
    face = COL["green_light"] if status == "good" else COL["red_light"]
    rounded(ax, x, y, w, h, face, edge, lw=1.3, radius=0.014)
    causal = [r for r in case["results"] if r["type"] == "causal"]
    causal_ok = sum(bool(r["correct"]) for r in causal)
    local_ids = [f"q{i}" for i in range(1, 9)]
    local_ok = sum(bool(case["by_id"][qid]["correct"]) for qid in local_ids)
    ax.text(x + w / 2, y + h * 0.80, model, ha="center", va="center", fontsize=8.6, fontweight="bold", color=COL["ink"])
    rows = [
        ("QA", f"{case['correct_count']}/{case['question_count']}"),
        ("Local", f"{local_ok}/8"),
        ("Causal", f"{causal_ok}/{len(causal)}"),
    ]
    row_y = [0.53, 0.31, 0.11]
    for i, (k, v) in enumerate(rows):
        ax.text(x + 0.014, y + h * row_y[i], k, ha="left", va="center", fontsize=6.8, color=COL["muted"])
        ax.text(x + w - 0.014, y + h * row_y[i], v, ha="right", va="center", fontsize=7.6, color=COL["ink"], fontweight="bold")


def draw_model_row(ax, case, model, y, status_kind, override_statuses=None):
    row_h = 0.175
    face = "#FFFFFF" if status_kind == "good" else "#FFFCFC"
    rounded(ax, 0.035, y, 0.93, row_h, face, COL["line"], lw=1.0, ls=(0, (4, 3)), radius=0.018)
    ax.text(0.058, y + row_h - 0.038, model, ha="left", va="center", fontsize=11.8, fontweight="bold", color=COL["ink"])
    subtitle = "successful causal execution" if status_kind == "good" else "local success but causal break"
    ax.text(0.058, y + row_h - 0.066, subtitle, ha="left", va="center", fontsize=7.5, color=COL["muted"])

    x0, sw, gap = 0.220, 0.092, 0.014
    statuses = []
    for i, stage in enumerate(STAGES):
        qid, title, detail, role = stage
        if qid == "q1-q6":
            ids = [f"q{k}" for k in range(1, 7)]
        elif qid == "q7-q8":
            ids = ["q7", "q8"]
        else:
            ids = [qid]
        st = override_statuses[i] if override_statuses else stage_status(case, ids)
        statuses.append(st)
        draw_stage(ax, x0 + i * (sw + gap), y + 0.025, sw, 0.116, stage, st)

    for i in range(len(STAGES) - 1):
        x1 = x0 + i * (sw + gap) + sw + 0.004
        x2 = x0 + (i + 1) * (sw + gap) - 0.004
        yy = y + 0.083
        if statuses[i] == "ok" and statuses[i + 1] == "ok":
            color = COL["green_dark"]
        elif statuses[i + 1] == "miss":
            color = COL["red_dark"]
        else:
            color = "#777777"
        arrow(ax, x1, yy, x2, yy, color, lw=1.55)

    draw_score_card(ax, 0.872, y + 0.028, 0.076, 0.112, case, model, status_kind)


def draw_bottom_note(ax):
    rounded(ax, 0.035, 0.045, 0.93, 0.080, COL["panel"], COL["line"], lw=1.0, ls=(0, (4, 3)), radius=0.018)
    ax.text(0.055, 0.102, "Failure propagation diagnosis", fontsize=10.0, fontweight="bold", color=COL["ink"], va="center")
    steps = [
        ("q9 key-through-ring absent", COL["red_light"], COL["red_dark"]),
        ("q10 teal liquid blocked", COL["gray_light"], COL["muted"]),
        ("q11 reflection blocked", COL["gray_light"], COL["muted"]),
        ("q12 glove applause blocked", COL["gray_light"], COL["blue"]),
    ]
    x = 0.260
    for i, (txt, fc, ec) in enumerate(steps):
        pill(ax, x, 0.077, 0.145, 0.030, txt, fc, ec, color=COL["ink"], fs=6.8, weight="bold")
        if i < len(steps) - 1:
            arrow(ax, x + 0.150, 0.092, x + 0.172, 0.092, COL["muted"], lw=1.2)
        x += 0.175
    ax.text(
        0.055,
        0.058,
        "High object/action coverage does not imply correct causal state-transition following.",
        fontsize=7.2,
        color=COL["muted"],
        va="center",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    good = load_case(GOOD_FILE)
    bad = load_case(BAD_FILE)

    fig = plt.figure(figsize=(15.5, 9.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.5,
        0.988,
        "Dependency-aware causal chain diagnosis",
        ha="center",
        va="top",
        fontsize=14.8,
        fontweight="bold",
        color=COL["ink"],
    )
    ax.text(
        0.5,
        0.958,
        "A good model completes the trigger-state-effect chain; a weaker model preserves local objects but breaks the causal transition.",
        ha="center",
        va="top",
        fontsize=8.2,
        color=COL["muted"],
    )

    draw_prompt_panel(ax, good["prompt"])
    draw_decomposition(ax)

    ax.text(0.055, 0.565, "Model-level evaluation trace", fontsize=10.2, fontweight="bold", color=COL["ink"], va="center")
    pill(ax, 0.795, 0.552, 0.060, 0.024, "green: correct", COL["green_light"], COL["green_dark"], fs=6.4)
    pill(ax, 0.862, 0.552, 0.052, 0.024, "red: miss", COL["red_light"], COL["red_dark"], fs=6.4)
    pill(ax, 0.920, 0.552, 0.052, 0.024, "gray: blocked", COL["gray_light"], "#777777", fs=6.4)

    draw_model_row(ax, good, GOOD_MODEL, 0.350, "good")
    draw_model_row(ax, bad, BAD_MODEL, 0.145, "bad")
    draw_bottom_note(ax)

    fig.savefig(OUTPUT_DIR / "fig_casestudy_dag.pdf", format="pdf", bbox_inches=None)
    fig.savefig(OUTPUT_DIR / "fig_casestudy_dag.png", format="png", bbox_inches=None)
    plt.close(fig)
    print("Saved publication-style case study figure.")


if __name__ == "__main__":
    main()
