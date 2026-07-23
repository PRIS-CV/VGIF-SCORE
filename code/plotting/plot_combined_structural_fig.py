"""
Consolidated structural figure: (a) Position line chart + (b) Depth heatmap
Reads from data/final_qa/ — 1 decimal precision throughout.
"""
import json, os, re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BASE = Path(__file__).resolve().parent.parent.parent
QA_DIR = BASE / 'data' / 'final_qa'
OUT_DIR = BASE / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Data loading from data/final_qa/
# ═══════════════════════════════════════════════════════════════

Q_PREFIXES = ['is there a ','is there an ','is the ','is a ','does the ','does a ',
    'are the ','are there ','do the ','was the ','were the ',
    'can the ','will the ','should the ','has the ','have the ']
Q_SUFFIXES = [' shown in the scene',' visible in the scene',' present in the scene',
    ' displayed',' visible',' shown',' present in the video',' depicted',' observed',
    ' in the scene',' in the video',' appearing in the scene',' appear in the scene',
    ' seen',' clearly visible',' clearly shown',' clearly depicted']

def find_position(question, prompt_lower, n_words):
    q_lower = question.lower().rstrip('?')
    for p in Q_PREFIXES:
        if q_lower.startswith(p): q_lower = q_lower[len(p):]; break
    for s in Q_SUFFIXES:
        if q_lower.endswith(s): q_lower = q_lower[:-len(s)]; break
    idx = prompt_lower.find(q_lower)
    if idx >= 0: return len(prompt_lower[:idx].split()) / max(n_words, 1)
    q_words = q_lower.split()
    best_pos, best_len = None, 0
    for i in range(len(q_words)):
        for j in range(i+2, len(q_words)+1):
            phrase = ' '.join(q_words[i:j])
            pos = prompt_lower.find(phrase)
            if pos >= 0 and len(phrase) > best_len:
                best_len = len(phrase); best_pos = len(prompt_lower[:pos].split())
    return best_pos / max(n_words, 1) if best_pos is not None else None

def compute_depths(results):
    deps = {}
    for r in results: deps[r['id']] = r.get('dependency_ids', [])
    memo = {}
    def cd(qid, visited=None):
        if qid in memo: return memo[qid]
        if visited is None: visited = set()
        if qid in visited: return 0
        visited.add(qid)
        memo[qid] = 0 if not deps.get(qid, []) else 1 + max(cd(d, visited.copy()) for d in deps[qid])
        return memo[qid]
    return {qid: cd(qid) for qid in deps}

print('Loading data from data/final_qa/...')
depth_data = defaultdict(lambda: defaultdict(list))  # model -> depth -> [correct]
position_data = defaultdict(list)                     # model -> [(rel_pos, correct)]
overall_d = defaultdict(list)                         # depth -> [correct]

for mdir in sorted(QA_DIR.iterdir()):
    if not mdir.is_dir(): continue
    m = mdir.name
    for fp in sorted(mdir.glob('*.json')):
        try: d = json.load(open(fp, encoding='utf-8'))
        except: continue
        if not d.get('success') or 'results' not in d: continue
        prompt = d.get('prompt',''); prompt_l = prompt.lower(); nw = len(prompt.split())
        dps = compute_depths(d['results'])
        for r in d['results']:
            dp = dps.get(r['id'], 0); corr = r.get('correct', False)
            depth_data[m][dp].append(corr)
            overall_d[dp].append(corr)
            rel = find_position(r['question'], prompt_l, nw)
            if rel is not None:
                position_data[m].append((rel, corr))

# Aggregate
def aggregate_depth(depth_data):
    rows = []
    for m, dp_dict in depth_data.items():
        for dp, vals in dp_dict.items():
            n = len(vals); acc = sum(vals)/n*100
            rows.append({'model': m, 'depth': dp, 'total': n, 'accuracy': acc})
    return rows

def aggregate_position(position_data, n_bins=10):
    rows = []
    for m, pairs in position_data.items():
        bins = np.linspace(0, 1, n_bins+1)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i+1]
            vals = [c for r, c in pairs if lo <= r < hi]
            if vals:
                rows.append({'model': m, 'bin_mid': (lo+hi)/2, 'total': len(vals),
                             'accuracy': sum(vals)/len(vals)*100,
                             'bin': '{:.1f}-{:.1f}'.format(lo, hi)})
    return rows

agg_d = aggregate_depth(depth_data)
agg_p = aggregate_position(position_data)

# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':11,'axes.titlesize':14,'axes.labelsize':13,
    'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':8,
    'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.linewidth':0.8,'axes.spines.top':False,'axes.spines.right':False,
})

COLORS = {
    'PixVerse-V6':'#E64B35','Wan-2.7':'#00A087','ViduQ3-Turbo':'#DC0000',
    'Seedance-2.0':'#4DBBD5','Kling-V3':'#F39B7F','Wan2.2-A14B':'#7E6148',
    'InfinityStar':'#33A02C','LongCat-Video':'#91D1C2','HyVideo-1.5':'#BEBADA',
    'LTX-2.0':'#3C5488','Mochi-1':'#B09C85','CogVideoX-1.5':'#FF7F00',
    'URSA':'#B15928','MAGI-1':'#6A3D9A',
}
MARKERS = ['o','D','^','s','v','>','d','h','p','<','H','*','P','X']

# Sort models by overall accuracy
model_acc = {}
for m in depth_data:
    all_v = [c for vals in depth_data[m].values() for c in vals]
    model_acc[m] = sum(all_v)/len(all_v)*100 if all_v else 0
model_order = sorted(model_acc, key=model_acc.get, reverse=True)

FIG_W, FIG_H = 18, 7.0
fig = plt.figure(figsize=(FIG_W, FIG_H))

# ---- Left: Position line chart ----
plot_w = (1.0 - 0.04 - 0.03 - 0.08) / 2  # 0.08 gap
ax1 = fig.add_axes([0.04, 0.13, plot_w, 0.82])
for i, model in enumerate(model_order):
    pts = [(r['bin_mid'], r['accuracy']) for r in agg_p if r['model'] == model]
    if not pts: continue
    pts.sort()
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax1.plot(xs, ys, marker=MARKERS[i], color=COLORS.get(model,'#888'),
             linewidth=1.3, markersize=5.5, markeredgecolor='white', markeredgewidth=0.3, label=model)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-5, 90)
ax1.yaxis.set_major_locator(mticker.FixedLocator([0, 20, 40, 60, 80]))
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
ax1.set_xlabel('Relative Position in Prompt', fontsize=13)
ax1.set_ylabel('QA Accuracy (%)', fontsize=13)
ax1.set_title('(a) Accuracy vs. Prompt Position', fontweight='bold', fontsize=14, pad=10)
ax1.grid(axis='y', alpha=0.3, linewidth=0.5)
ax1.legend(ncol=2, frameon=True, edgecolor='#cccccc', loc='upper right', fontsize=8)

# ---- Right: Depth heatmap (1 decimal) ----
ax2 = fig.add_axes([0.04 + plot_w + 0.08, 0.13, plot_w, 0.82])
# Merge depths 8-12
agg_d2 = []
for r in agg_d:
    d = r['depth']
    label = '8-12' if d >= 8 else str(int(d))
    agg_d2.append({'model': r['model'], 'depth_label': label, 'total': r['total'], 'accuracy': r['accuracy']})

from collections import Counter
agg_m = defaultdict(lambda: defaultdict(lambda: {'sum': 0, 'n': 0}))
for r in agg_d2:
    m = r['model']; dl = r['depth_label']
    agg_m[m][dl]['sum'] += r['accuracy'] * r['total']
    agg_m[m][dl]['n'] += r['total']

pivot_data = {}
pivot_n = {}
for m in agg_m:
    pivot_data[m] = {}
    pivot_n[m] = {}
    for dl in agg_m[m]:
        pivot_data[m][dl] = agg_m[m][dl]['sum'] / agg_m[m][dl]['n']
        pivot_n[m][dl] = agg_m[m][dl]['n']

col_order = [str(d) for d in range(8)] + ['8-12']
row_order = sorted(pivot_data.keys(), key=lambda m: model_acc.get(m, 0), reverse=True)

pivot_arr = np.array([[pivot_data[m].get(c, np.nan) for c in col_order] for m in row_order])

cmap = LinearSegmentedColormap.from_list('bwy', ['#A8D8EA','#F8F8F8','#FFE066'])
im = ax2.imshow(pivot_arr, aspect='auto', cmap=cmap, vmin=0, vmax=100)
for i in range(len(row_order)):
    for j in range(len(col_order)):
        val = pivot_arr[i][j]
        if not np.isnan(val):
            ax2.text(j, i, '{:.1f}'.format(val), ha='center', va='center',
                     fontsize=9, color='#333333', fontweight='medium')
ax2.set_xticks(range(len(col_order)))
ax2.set_xticklabels(['D{}'.format(c) for c in col_order], rotation=0, fontsize=10)
ax2.set_yticks(range(len(row_order)))
ax2.set_yticklabels(row_order, fontsize=10)
ax2.set_xlabel('Dependency Depth', fontsize=13, labelpad=8)
ax2.set_title('(b) Accuracy Heatmap: Model x Dependency Depth', fontweight='bold', fontsize=14, pad=10)
cbar = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
cbar.ax.tick_params(labelsize=10)
cbar.set_label('QA Accuracy (%)', fontsize=13)

fig.savefig(str(OUT_DIR / 'fig_structural_combined.pdf'), format='pdf')
fig.savefig(str(OUT_DIR / 'fig_structural_combined.png'), format='png')
plt.close(fig)
print('Saved: fig_structural_combined.pdf / .png')

# Print key numbers for paper verification
print()
print('=== KEY NUMBERS (verify against paper text) ===')
print('Depth 0: {:.1f}%'.format(sum(overall_d[0])/len(overall_d[0])*100))
print('Depth 1: {:.1f}%'.format(sum(overall_d[1])/len(overall_d[1])*100))
print('Depth 2: {:.1f}%'.format(sum(overall_d[2])/len(overall_d[2])*100))
print('Depth 3: {:.1f}%'.format(sum(overall_d[3])/len(overall_d[3])*100))
print('Depth 4: {:.1f}%'.format(sum(overall_d[4])/len(overall_d[4])*100))
print()
for m in row_order[:3]:
    print('{} D2: {:.1f}%'.format(m, pivot_data[m].get('2', 0)))
for m in row_order[-3:]:
    print('{} D2: {:.1f}%'.format(m, pivot_data[m].get('2', 0)))
# Position
pos_bins = defaultdict(list)
for m, pairs in position_data.items():
    for rel, corr in pairs:
        lo = int(rel*10)/10
        pos_bins['{:.1f}'.format(lo)].append(corr)
for k in sorted(pos_bins.keys())[:5]:
    vals = pos_bins[k]
    print('Pos {}: {:.1f}% (n={})'.format(k, sum(vals)/len(vals)*100, len(vals)))
