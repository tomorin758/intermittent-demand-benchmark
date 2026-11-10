"""Window stability: does the leading set survive a shift of the evaluation window?

The paper reports one held-out window per dataset (Section 3.2). This script re-scores the same
models on earlier, non-overlapping windows of the same held-out region and summarises how much the
ranking moves, which backs the "Window stability" paragraph of Section 4.1.

The per-model prediction arrays for those earlier windows are not redistributed (see the README,
"Scope of this bundle"), so the computed table ships as results/window_stability.csv and this script
documents exactly how it was produced.

For each dataset the windows are spaced by one horizon H and indexed from the reported window
(index 0) backwards; window w ends at row L - H - w*H of the history table. Stored prediction arrays
are in model column order for three families, so the per-model transform recorded in
code/pipeline/build_final.py is applied to every window before scoring. The always-zero reference is
scored from the same ground truth.

Usage:  python analysis/window_stability.py [dataset ...]        # default: parts fresh_retail m5
        PRED_DIR points at <dataset>/<model>/{pred,true}.npy, default data/predictions.
Writes results/window_stability.csv and results/window_stability.txt (override with RESULTS_DIR).
"""
import csv, os, sys
import numpy as np

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.environ.get('PRED_DIR', os.path.join(BUNDLE, 'data', 'predictions'))
RESULTS = os.environ.get('RESULTS_DIR', os.path.join(BUNDLE, 'results'))
DS = [a for a in sys.argv[1:] if a] or ['parts', 'fresh_retail', 'm5']
C = 0.5
HORIZON = {'parts': 3, 'fresh_retail': 7, 'm5': 28}

# Per-model column transform for the raw stored arrays, copied from code/pipeline/build_final.py:
# k rotates the column order ("remove column k, append at the end"), unscramble reshapes the block.
TRANSFORM = {
    'parts': {'Autoformer': dict(k=0), 'DLinear': dict(k=0), 'HurdleDLinear': dict(k=0),
              'PatchTCN': dict(k=0), 'PatchTST': dict(k=0), 'PatchMixer': dict(k=0),
              'Moirai': dict(), 'PatchTSMixer': dict(), 'TTM': dict(), 'TimesFM': dict(),
              'MOMENT': dict(unscramble='global')},
    'fresh_retail': {'Autoformer': dict(k=0), 'DLinear': dict(k=179), 'HurdleDLinear': dict(k=179),
                     'PatchTCN': dict(k=179), 'PatchTST': dict(k=179), 'Moirai': dict(),
                     'PatchMixer': dict(), 'PatchTSMixer': dict(), 'TTM': dict(),
                     'TimesFM': dict(), 'MOMENT': dict(unscramble=500)},
    'm5': {'Autoformer': dict(k=0), 'Moirai': dict(), 'PatchMixer': dict(), 'PatchTSMixer': dict(),
           'TTM': dict(), 'TimesFM': dict(), 'MOMENT': dict(unscramble=500),
           'Timer': dict(unscramble=500)},
}

_missing = [d for d in DS if not os.path.isdir(os.path.join(PRED, d))]
if _missing:
    raise SystemExit(
        'missing prediction arrays in %s: %s\n'
        'The per-model prediction arrays are not redistributed, so the window check cannot be '
        'recomputed from the bundle alone; its values ship as results/window_stability.csv (see the '
        'README, "Scope of this bundle"). Point PRED_DIR at a directory of '
        '<dataset>/<model>/{pred,true}.npy to recompute it.'
        % (PRED, ', '.join(_missing)))


def unscramble(block, chunk=None):
    h, n = block.shape
    if chunk is None or n <= chunk:
        return block.ravel(order='C').reshape(n, h).T
    parts, start = [], 0
    while start < n:
        length = min(chunk, n - start)
        parts.append(block[:, start:start + length].ravel(order='C').reshape(length, h).T)
        start += length
    return np.concatenate(parts, axis=1)


def to_report_order(block, k):
    if k is None:
        return block
    n = block.shape[1]
    out = np.empty_like(block)
    out[:, list(range(k)) + list(range(k + 1, n)) + [k]] = block
    return out


def apply_recipe(block, recipe):
    if recipe.get('unscramble') == 'global':
        block = unscramble(block)
    elif recipe.get('unscramble') == 500:
        block = unscramble(block, 500)
    return to_report_order(block, recipe.get('k'))


def f1_zero(pred, true):
    pz, tz = pred < C, true < C
    tp = int((pz & tz).sum()); fp = int((pz & ~tz).sum()); fn = int((~pz & tz).sum())
    prec = tp / (tp + fp) if tp + fp else float('nan')
    rec = tp / (tp + fn) if tp + fn else float('nan')
    return 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec > 0 else float('nan')


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    da = np.sqrt(sum((x - ma) ** 2 for x in ra)); db = np.sqrt(sum((y - mb) ** 2 for y in rb))
    return float(sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (da * db))


rows, report = [], []
for ds in DS:
    h = HORIZON[ds]
    preds, trues = {}, {}
    for model, recipe in TRANSFORM[ds].items():
        base = os.path.join(PRED, ds, model)
        preds[model] = np.load(os.path.join(base, 'pred.npy'))
        trues[model] = np.load(os.path.join(base, 'true.npy'))
    windows = (min(a.shape[0] for a in preds.values()) - 1) // h + 1
    scores = []
    for w in range(windows):
        window = {}
        for model, recipe in TRANSFORM[ds].items():
            t = preds[model].shape[0] - 1 - w * h
            p = apply_recipe(preds[model][t], recipe)
            y = apply_recipe(trues[model][t], recipe)
            ql = 0.5 * float(np.mean(np.abs(p - y)))
            window[model] = (ql, f1_zero(p, y))
            rows.append([ds, w, model, '%.6f' % ql, '%.6f' % f1_zero(p, y)])
        t = trues['TimesFM'].shape[0] - 1 - w * h
        y = apply_recipe(trues['TimesFM'][t], TRANSFORM[ds]['TimesFM'])
        z = np.zeros_like(y)
        ql0 = 0.5 * float(np.mean(np.abs(z - y)))
        window['always-zero'] = (ql0, f1_zero(z, y))
        rows.append([ds, w, 'always-zero', '%.6f' % ql0, '%.6f' % f1_zero(z, y)])
        scores.append(window)

    models = sorted(scores[0])
    winners = [min(scores[w], key=lambda m: scores[w][m][0]) for w in range(windows)]
    base = [scores[0][m][0] for m in models]
    rhos = [spearman(base, [scores[w][m][0] for m in models]) for w in range(1, windows)]
    report.append('')
    report.append('%s: %d models, %d non-overlapping windows of H=%d'
                  % (ds, len(models), windows, h))
    report.append('  winner per window: ' + ', '.join(winners))
    if rhos:
        report.append('  rank correlation against the reported window: %.3f to %.3f'
                      % (min(rhos), max(rhos)))
    top3 = {tuple(sorted(models, key=lambda m: scores[w][m][0])[:3]) for w in range(windows)}
    report.append('  distinct top-three sets: %d %s' % (len(top3), sorted(top3)[0] if len(top3) == 1 else ''))

os.makedirs(RESULTS, exist_ok=True)
out = os.path.join(RESULTS, 'window_stability.csv')
with open(out, 'w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['dataset', 'window', 'model', 'ql_raw', 'f1_trunc'])
    writer.writerows(rows)
txt = os.path.join(RESULTS, 'window_stability.txt')
with open(txt, 'w') as fh:
    fh.write('\n'.join(report) + '\n')
print('\n'.join(report))
print('[saved] %s (%d rows) and %s'
      % (os.path.relpath(out, BUNDLE), len(rows), os.path.relpath(txt, BUNDLE)))
