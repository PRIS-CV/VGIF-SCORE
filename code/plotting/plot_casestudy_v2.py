"""
§5.5 Case Study Figure — Kling-V3 vs CogVideoX-1.5 on Sample 219
简洁设计：左侧 Kling QA链 + 右侧 CogVideoX QA链 + 底部 AutoRubric对比
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':11,'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
})

TYPE_COLORS = {"entity":"#8DD3C7","attr":"#FFFFB3","action":"#BEBADA","state":"#FB8072","causal":"#E41A1C"}

# ── QA chain: (label, type, y, x_off, deps, cog_ok) ──
QA = [
    ("Bottle shown",         "entity",  9.5, 0, [],      True),
    ("Clear liquid",         "attr",    9.5, 1, [0],     True),
    ("Mirrored plinth",      "entity",  8.5, 0, [0],     True),
    ("Silk glove visible",   "entity",  7.5, 0, [],      True),
    ("Key visible",          "entity",  6.5, 0, [],      True),
    ("Key is gold",          "attr",    6.5, 1, [4],     True),
    ("Bottle sprays mist",   "action",  5.5, 0, [0],     True),
    ("Mist forms ring",      "state",   4.5, 0, [6],     True),
    ("Key rotates thru ring","action",  3.5, 0, [7,4],   False),   # ← break
    ("Liquid turns teal",    "causal",  2.5, 0, [8,0],   False),   # ← cascade
    ("Teal affects mirror",  "action",  1.5, 0, [9,1],   False),   # ← cascade
    ("Glove applauds bottle","causal",  0.5, 0, [10,2,0],False),   # ← cascade
]
# Build pos dict
pos = {}
for i, (label, qtype, y, xo, deps, cog_ok) in enumerate(QA):
    pos[i] = (xo * 1.8 + 0.5, y)

RUBRIC_DIMS = ["Cin.", "Pur.", "Mot.", "Phy."]
KLING_AR = [4, 5, 2, 2]
COG_AR   = [2, 5, 2, 1]

fig = plt.figure(figsize=(16, 7.2))

# ===== Left: Kling-V3 QA chain =====
ax1 = fig.add_axes([0.02, 0.18, 0.42, 0.78])
ax1.set_xlim(0, 6); ax1.set_ylim(-1, 11)
ax1.axis('off')
ax1.set_title('Kling-V3  —  QA: 12/12 (100%)  |  AR: Cin=4 Pur=5 Mot=2 Phy=2',
              fontweight='bold', fontsize=12, pad=6)

for i, (label, qtype, y, xo, deps, cog_ok) in enumerate(QA):
    x, yp = pos[i]
    for dep in deps:
        dx, dy = pos[dep]
        ax1.annotate("", xy=(x, yp-0.25), xytext=(dx, dy+0.25),
                    arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.8))
    c = TYPE_COLORS.get(qtype, "#CCC")
    rect = mpatches.FancyBboxPatch((x-0.9, yp-0.22), 1.8, 0.44,
            boxstyle="round,pad=0.03", facecolor=c, edgecolor='#4DAF4A',
            linewidth=2.5, alpha=0.9)
    ax1.add_patch(rect)
    ax1.text(x, yp, f"q{i+1}", ha='center', va='center', fontsize=7.5, fontweight='bold')
    ax1.text(x+1.0, yp, label, ha='left', va='center', fontsize=7, color='#444')

leg1 = [mpatches.Patch(facecolor=TYPE_COLORS[t], edgecolor='#555', label=t.capitalize()) for t in ["entity","attr","action","state","causal"]]
ax1.legend(handles=leg1, loc='lower right', fontsize=7, ncol=5, frameon=True, edgecolor='#ccc')

# ===== Right: CogVideoX QA chain =====
ax2 = fig.add_axes([0.44, 0.18, 0.42, 0.78])
ax2.set_xlim(0, 6); ax2.set_ylim(-1, 11)
ax2.axis('off')
ax2.set_title('CogVideoX-1.5  —  QA: 8/12 (67%)  |  AR: Cin=2 Pur=5 Mot=2 Phy=1',
              fontweight='bold', fontsize=12, pad=6)

for i, (label, qtype, y, xo, deps, cog_ok) in enumerate(QA):
    x, yp = pos[i]
    for dep in deps:
        dx, dy = pos[dep]
        ax2.annotate("", xy=(x, yp-0.25), xytext=(dx, dy+0.25),
                    arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.8))
    ok = cog_ok
    ec = '#4DAF4A' if ok else '#E41A1C'
    lw = 2.5 if ok else 3.0
    fc = TYPE_COLORS.get(qtype, "#CCC") if ok else '#FFD0D0'
    rect = mpatches.FancyBboxPatch((x-0.9, yp-0.22), 1.8, 0.44,
            boxstyle="round,pad=0.03", facecolor=fc, edgecolor=ec,
            linewidth=lw, alpha=0.9)
    ax2.add_patch(rect)
    ax2.text(x, yp, f"q{i+1}", ha='center', va='center', fontsize=7.5, fontweight='bold')
    ax2.text(x+1.0, yp, label, ha='left', va='center', fontsize=7, color='#444')

leg2 = [mpatches.Patch(facecolor='#4DAF4A', alpha=0.7, label='Correct'),
        mpatches.Patch(facecolor='#E41A1C', alpha=0.7, label='Incorrect')]
ax2.legend(handles=leg2, loc='lower right', fontsize=7, frameon=True, edgecolor='#ccc')

# ===== Bottom: Prompt text =====
prompt = (
    "Prompt 219: In a minimalist white gallery, place a clear perfume bottle on a mirrored plinth, "
    "with a silk glove draped beside it and a gold key leaning against the bottle. The bottle emits a fine "
    "mist that swirls into a floating ring; the gold key rotates through the mist ring. As it passes through, "
    "the perfume inside shifts from clear to deep teal. The teal essence seeps into the mirror reflection, "
    "and as the reflection ripples, the silk glove animates into an applauding motion toward the bottle."
)
fig.text(0.5, 0.09, prompt, ha='center', va='center', fontsize=7.8,
         fontstyle='italic', color='#555',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAFAFA', edgecolor='#DDDDDD'))

# Key insight annotation
fig.text(0.5, 0.015, "CogVideoX-1.5 correctly generates the scene and early mist formation (q1–q8), "
         "but misses the bridge action q9 (key rotating through ring). "
         "This single failure cascades to nullify all downstream causal/state nodes (q10–q12).",
         ha='center', fontsize=9, fontstyle='italic', color='#888')

fig.savefig('scripts/output_figs/fig5_casestudy_diagnosis.pdf', format='pdf')
fig.savefig('scripts/output_figs/fig5_casestudy_diagnosis.png', format='png')
plt.close(fig)
print('Saved: fig5_casestudy_diagnosis.pdf / .png')
