"""
§5.5 完整案例图 — 视频帧 + QA链 + AutoRubric + 因果分析
Kling-V3 vs CogVideoX-1.5 on Sample 219
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':9,'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
})

CASE_DIR = 'case1'
FRAME_DIR_K = f'{CASE_DIR}/Kling-V3/frame'
FRAME_DIR_C = f'{CASE_DIR}/CogVideoX/frame'

# ── 读取关键帧 ──
def load_frame(path):
    img = plt.imread(path)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img

k_frames = [load_frame(f'{FRAME_DIR_K}/{i}.jpg') for i in [1, 3, 6]]
c_frames = [load_frame(f'{FRAME_DIR_C}/{i}.jpg') for i in [1, 3, 5]]

# ── QA节点定义 ──
# (label, type, correct_for_cog)
QA_NODES = [
    ("q1  Bottle shown",          "entity",   True),
    ("q2  Clear liquid",          "attr",     True),
    ("q3  Mirrored plinth",       "entity",   True),
    ("q4  Silk glove visible",    "entity",   True),
    ("q5  Key visible",           "entity",   True),
    ("q6  Key is gold",           "attr",     True),
    ("q7  Bottle sprays mist",    "action",   True),
    ("q8  Mist forms ring",       "state",    True),
    ("q9  Key rotates thru ring", "action",   False),   # ← breakpoint
    ("q10 Liquid turns teal",     "causal",   False),
    ("q11 Teal affects mirror",   "action",   False),
    ("q12 Glove applauds bottle", "causal",   False),
]

TYPE_COLORS = {"entity":"#8DD3C7","attr":"#FFFFB3","action":"#BEBADA","state":"#FB8072","causal":"#E41A1C"}

# ── Prompt ──
PROMPT = (
    "Prompt (Sample 219 — Perfume Transformation): In a minimalist white gallery, place a clear perfume bottle on a mirrored plinth, "
    "with a silk glove draped beside it and a gold key leaning against the bottle. The bottle emits a fine mist that swirls into a "
    "floating ring; the gold key rotates through the mist ring. As it passes through, the perfume inside shifts from clear to deep teal. "
    "The teal essence seeps into the mirror reflection, and as the reflection ripples, the silk glove animates into an applauding motion toward the bottle."
)

# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14.5))

# ── ① 提示词 (顶部) ──
fig.text(0.5, 0.975, PROMPT, ha='center', va='top', fontsize=8.5, fontstyle='italic', color='#444',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F8F8', edgecolor='#CCCCCC', linewidth=0.8),
         linespacing=1.5, wrap=True)

# ── Section labels ──
fig.text(0.25, 0.92, 'Kling-V3', ha='center', fontsize=14, fontweight='bold', color='#2166AC')
fig.text(0.75, 0.92, 'CogVideoX-1.5', ha='center', fontsize=14, fontweight='bold', color='#B2182B')
fig.text(0.25, 0.90, 'QA: 12/12 (100%)   AutoRubric: Cin=4  Pur=5  Mot=2  Phy=2', ha='center', fontsize=9, color='#555')
fig.text(0.75, 0.90, 'QA:  8/12  (67%)   AutoRubric: Cin=2  Pur=5  Mot=2  Phy=1', ha='center', fontsize=9, color='#555')

# ── ② 视频帧 (3列 × 2行) ──
frame_labels = ['Scene Setup', 'Mist Ring', 'Causal Outcome']
frame_y = 0.84
frame_h = 0.08
for col, (label, k_img, c_img) in enumerate(zip(frame_labels, k_frames, c_frames)):
    x_l = 0.05 + col * 0.18
    x_r = 0.55 + col * 0.18
    # Kling frame
    ax_k = fig.add_axes([x_l, frame_y - frame_h, 0.12, frame_h])
    ax_k.imshow(k_img); ax_k.axis('off')
    ax_k.set_title(label, fontsize=8, pad=2)
    # CogVideoX frame
    ax_c = fig.add_axes([x_r, frame_y - frame_h, 0.12, frame_h])
    ax_c.imshow(c_img); ax_c.axis('off')
    ax_c.set_title(label, fontsize=8, pad=2)
    # Green/red border for CogVideoX frames
    border_color = '#4DAF4A' if col < 2 else '#E41A1C'
    for spine in ax_c.spines.values():
        spine.set_visible(True); spine.set_color(border_color); spine.set_linewidth(3)

# ── ③ QA链 (左右各一列) ──
chain_top = 0.70
chain_bottom = 0.08
node_spacing = (chain_top - chain_bottom) / (len(QA_NODES) - 1)

for side, x_center in [('K', 0.40), ('C', 0.90)]:
    for i, (label, qtype, cog_ok) in enumerate(QA_NODES):
        y = chain_top - i * node_spacing
        correct = True if side == 'K' else cog_ok
        fc = TYPE_COLORS.get(qtype, '#CCC') if correct else '#FFD0D0'
        ec = '#4DAF4A' if correct else '#E41A1C'
        lw = 1.8 if correct else 2.5

        rect = FancyBboxPatch((x_center - 0.08, y - 0.012), 0.16, 0.024,
                boxstyle="round,pad=0.004", facecolor=fc, edgecolor=ec,
                linewidth=lw, alpha=0.9, transform=fig.transFigure, zorder=3)
        fig.patches.append(rect)
        fig.text(x_center, y, label, ha='center', va='center', fontsize=5.5,
                fontweight='bold', color='#222', zorder=4)

        # Arrow between nodes
        if i < len(QA_NODES) - 1:
            y_next = chain_top - (i+1) * node_spacing
            fig.add_artist(FancyArrowPatch(
                (x_center, y - 0.012), (x_center, y_next + 0.012),
                arrowstyle='->', color='#999999' if correct else '#E41A1C',
                lw=1.0 if correct else 1.5, transform=fig.transFigure, zorder=2))

    # Type legend on right column only
    if side == 'C':
        for t, (tname, tcol) in enumerate(TYPE_COLORS.items()):
            y_leg = 0.10 - t * 0.015
            fig.patches.append(FancyBboxPatch((0.92, y_leg), 0.03, 0.012,
                    boxstyle="round,pad=0.002", facecolor=tcol, edgecolor='#888',
                    linewidth=0.5, transform=fig.transFigure, zorder=5))
            fig.text(0.955, y_leg + 0.006, tname.capitalize(), ha='left', va='center',
                    fontsize=5.5, transform=fig.transFigure)

# ── ④ Breakpoint annotation (CogVideoX) ──
break_y = chain_top - 8 * node_spacing  # q9 position
fig.text(0.94, break_y, '← Causal chain\n   breaks here', ha='left', va='center',
         fontsize=7, color='#E41A1C', fontweight='bold', fontstyle='italic')

# ── ⑤ 底部总结 ──
fig.text(0.5, 0.025,
    'Key Insight: Both models correctly establish the scene and early events (q1–q8). '
    'However, CogVideoX-1.5 fails at the bridge action q9 (key rotating through the mist ring), '
    'which is the single causal trigger for the entire downstream chain (q10–q12). '
    'The AutoRubric independently confirms this: physics adherence drops from 2→1 and cinematography from 4→2, '
    'verifying that the objective QA failure corresponds to degraded subjective perception.',
    ha='center', va='center', fontsize=8, fontstyle='italic', color='#666',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8F0', edgecolor='#E0D0C0', linewidth=0.8))

fig.savefig('scripts/output_figs/fig5_casestudy_final.pdf', format='pdf')
fig.savefig('scripts/output_figs/fig5_casestudy_final.png', format='png')
plt.close(fig)
print('Saved: fig5_casestudy_final.pdf')
