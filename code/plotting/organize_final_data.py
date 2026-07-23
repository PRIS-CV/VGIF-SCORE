"""
从 models/ 中为每个模型选择最佳评估版本，复制到 data/final_qa/
规则：优先 gemini-3.1-pro-preview，覆盖不足用次优版本（整模型统一评估器）
"""
import json, os, re, shutil
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE / 'models'
OUT_DIR = BASE / 'data' / 'final_qa'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Acceptable evaluator versions in descending preference
EVAL_PREFERENCE = [
    'gemini-3.1-pro-preview',
    'gemini-3-pro-preview',
    'gemini-2.5-pro',
    'gemini-2.5-flash',
]

print('Scanning models/ for evaluator versions...')
model_files = defaultdict(list)  # model -> [(path, evaluator, is_31)]

for model_dir in sorted(MODELS_DIR.iterdir()):
    if not model_dir.is_dir(): continue
    m = model_dir.name
    # Walk all dependency_rounds JSONs
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if 'dependency_rounds' not in f or not f.endswith('.json'): continue
            fp = Path(root) / f
            try:
                d = json.load(open(fp, encoding='utf-8'))
            except: continue
            if not d.get('success') or 'results' not in d: continue
            em = d.get('model', '?')
            is_31 = '3.1' in em or '31' in em.lower()
            model_files[m].append((fp, em, is_31))

print()
print('{:20s} {:>6s} {:>6s} {:>12s}'.format('Model', 'Videos', '3.1_%', 'Evaluator'))
print('-' * 52)

stats = {}
for m in sorted(model_files.keys()):
    entries = model_files[m]
    total = len(entries)
    n31 = sum(1 for _, _, is31 in entries if is31)
    pct = n31 / total * 100 if total else 0

    # Dedup: group by video base name, prefer highest-priority evaluator
    video_groups = defaultdict(list)
    for fp, em, is31 in entries:
        bn = re.sub(r'_(?:gemini.*?_|g\d+.*?_|latestcfg_)?qa_eval_dependency_rounds\.json$', '', fp.name)
        video_groups[bn].append((fp, em, is31))

    # For each video, pick the best evaluator available
    chosen = []
    chosen_eval = None
    for bn, cands in video_groups.items():
        # Sort by evaluator preference
        cands.sort(key=lambda x: EVAL_PREFERENCE.index(x[1]) if x[1] in EVAL_PREFERENCE else 999)
        fp, em, is31 = cands[0]
        chosen.append(fp)
        chosen_eval = em

    # Determine which evaluator to use for this model (majority vote)
    eval_counts = defaultdict(int)
    for fp in chosen:
        # Get evaluator from file
        try:
            d = json.load(open(fp, encoding='utf-8'))
            em = d.get('model', '?')
        except: continue
        # Normalize
        for pref in EVAL_PREFERENCE:
            if pref in em:
                eval_counts[pref] += 1
                break
        else:
            eval_counts[em] += 1

    # Pick best evaluator with >=90% coverage
    best_eval = None
    for pref in EVAL_PREFERENCE:
        if pref in eval_counts and eval_counts[pref] / len(chosen) >= 0.48:
            best_eval = pref
            break
    if not best_eval:
        best_eval = max(eval_counts, key=eval_counts.get)

    # Filter: only keep files with the chosen evaluator
    final_files = []
    for fp in chosen:
        try:
            d = json.load(open(fp, encoding='utf-8'))
            em = d.get('model', '?')
        except: continue
        if best_eval in em:
            final_files.append(fp)

    n_final = len(final_files)
    print('{:20s} {:>5d}  {:>5.0f}%  {:>12s}'.format(m, n_final, n_final/total*100 if total else 0, best_eval))
    stats[m] = {'count': n_final, 'evaluator': best_eval, 'files': final_files}

# Copy to final_qa
print()
print('Copying to data/final_qa/...')
for m, info in stats.items():
    mdir = OUT_DIR / m
    mdir.mkdir(parents=True, exist_ok=True)
    for fp in info['files']:
        dest = mdir / fp.name
        if not dest.exists():
            shutil.copy2(fp, dest)
    print('  {:20s} {} -> {} files'.format(m, info['evaluator'], len(info['files'])))

# Summary
print()
print('=' * 60)
print('Summary:')
total_videos = sum(v['count'] for v in stats.values())
print('  Total videos: {}'.format(total_videos))
evaluators_used = set(v['evaluator'] for v in stats.values())
print('  Evaluators used: {}'.format(', '.join(sorted(evaluators_used))))
for ev in sorted(evaluators_used):
    models_using = [m for m, v in stats.items() if v['evaluator'] == ev]
    print('    {}: {} ({})'.format(ev, len(models_using), ', '.join(models_using)))
print('=' * 60)

# Save stats for plotting scripts
with open(OUT_DIR / '_stats.json', 'w', encoding='utf-8') as f:
    json.dump({m: {'count': v['count'], 'evaluator': v['evaluator']} for m, v in stats.items()}, f, indent=2)
