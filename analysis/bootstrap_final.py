"""Paired bootstrap over SERIES on the unified final-window per-series tables.

All models in a dataset now score the same H periods of the same series, so the per-series
metric vectors are directly pairable by series name. For every model pair we resample series
with replacement (B=4000, Poisson bootstrap weights, mean 1) and report the 95% percentile CI
of the difference in the pooled metric.

Vectorised: one weight matrix per chunk is reused for all pairs, and all pairs are evaluated
as a single W @ D product.

Usage: python analysis/bootstrap_final.py <tag> <metric>
   metric in {ql_raw (default), ql_trunc, f1_trunc, mae_nz_raw}
Reads metrics/perseries/<dataset>/*.csv and writes results/bootstrap_<metric>.{txt,json}.
"""
import csv, os, sys, itertools, json
import numpy as np

# This bundle is self-contained: per-series tables come from metrics/perseries/, outputs are
# written to results/. Override with PERSERIES_DIR / RESULTS_DIR if you moved things around.
BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSERIES = os.environ.get('PERSERIES_DIR', os.path.join(BUNDLE, 'metrics', 'perseries'))
RESULTS = os.environ.get('RESULTS_DIR', os.path.join(BUNDLE, 'results'))
os.makedirs(RESULTS, exist_ok=True)
DS = ['parts', 'fresh_retail', 'm5']
tag = sys.argv[1] if len(sys.argv) > 1 else 'unified'
METRIC = sys.argv[2] if len(sys.argv) > 2 else 'ql_raw'
COL = {'ql_raw': 'ql_raw', 'ql_trunc': 'ql_trunc', 'f1_trunc': 'f1_trunc',
       'mae_nz_raw': 'mae_nz_raw'}[METRIC]
HIGHER_BETTER = {'f1_trunc'}          # metrics where a larger value ranks first
B, CHUNK = 4000, 500
rng = np.random.default_rng(20260924)
out = []


def p(s=''):
    print(s, flush=True); out.append(s)


COUNTS = ('tp0_trunc', 'fp0_trunc', 'fn0_trunc')


def load(ds, counts=False):
    """counts=False -> {model: {series: metric value}}; counts=True -> {model: {series: (tp,fp,fn)}}"""
    d = os.path.join(PERSERIES, ds)
    tab = {}
    if not os.path.isdir(d):
        return tab
    for f in sorted(os.listdir(d)):
        if f.endswith('.csv'):
            with open(os.path.join(d, f)) as fh:
                rows = list(csv.DictReader(fh))
            if counts:
                tab[f[:-4]] = {r['series']: tuple(float(r[c]) for c in COUNTS) for r in rows}
            else:
                tab[f[:-4]] = {r['series']: float(r[COL]) for r in rows if r[COL] not in ('', None)}
    return tab


def f1(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)


summary = {}
p(f'metric = {COL}   B = {B}   paired over series (identical resample for both models)')
for ds in DS:
    USE_COUNTS = METRIC == 'f1_trunc'
    tab = load(ds, counts=USE_COUNTS)
    p(); p('=' * 96); p(f'{ds}: {len(tab)} models'); p('=' * 96)
    if len(tab) < 2:
        p('  (not enough models)'); continue
    common = None
    for m in tab:
        common = set(tab[m]) if common is None else (common & set(tab[m]))
    common = sorted(common)
    if USE_COUNTS:
        M = {m: np.array([tab[m][s] for s in common], dtype=np.float64) for m in tab}
        means = {m: float(f1(M[m][:, 0].sum(), M[m][:, 1].sum(), M[m][:, 2].sum())) for m in tab}
    else:
        means = {m: float(np.mean([tab[m][s] for s in common])) for m in tab}
    order = sorted(tab, key=lambda m: -means[m] if METRIC in HIGHER_BETTER else means[m])
    n = len(common)
    p(f'  common series: {n}')
    p('  ranking: ' + ' < '.join(f'{m}({means[m]:.4f})' for m in order))
    pairs = list(itertools.combinations(order, 2))
    boot = np.empty((B, len(pairs)), dtype=np.float64)
    if USE_COUNTS:
        for s0 in range(0, B, CHUNK):
            k = min(CHUNK, B - s0)
            W = rng.poisson(1.0, size=(k, n)).astype(np.float64)
            F = {m: f1(W @ M[m][:, 0], W @ M[m][:, 1], W @ M[m][:, 2]) for m in order}
            for i, (a, b) in enumerate(pairs):
                boot[s0:s0 + k, i] = F[a] - F[b]
        mean = np.array([f1(M[a][:, 0].sum(), M[a][:, 1].sum(), M[a][:, 2].sum())
                         - f1(M[b][:, 0].sum(), M[b][:, 1].sum(), M[b][:, 2].sum()) for a, b in pairs])
    else:
        D = np.empty((n, len(pairs)), dtype=np.float32)
        for i, (a, b) in enumerate(pairs):
            D[:, i] = (np.array([tab[a][s] for s in common], dtype=np.float32)
                       - np.array([tab[b][s] for s in common], dtype=np.float32))
        for s0 in range(0, B, CHUNK):
            k = min(CHUNK, B - s0)
            W = rng.poisson(1.0, size=(k, n)).astype(np.float32)
            boot[s0:s0 + k] = (W @ D) / n
        mean = D.mean(axis=0)
    lo = np.percentile(boot, 2.5, axis=0); hi = np.percentile(boot, 97.5, axis=0)
    sig = int(np.sum((lo > 0) | (hi < 0)))
    p(f'  pairs: {len(pairs)}   CI excludes 0: {sig}')
    idx = {pr: i for i, pr in enumerate(pairs)}
    p('  adjacent pairs:')
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        j = idx[(a, b)]
        p(f'    {a:16s} vs {b:16s} d={mean[j]:+.4f}  CI=[{lo[j]:+.4f},{hi[j]:+.4f}]  '
          f'{"separated" if (lo[j] > 0 or hi[j] < 0) else "overlapping"}')
    best = order[0]
    grp = [best]
    p(f'  leading group (CI overlaps {best}):')
    for m in order[1:]:
        j = idx[(best, m)] if (best, m) in idx else idx[(m, best)]
        inb = not (lo[j] > 0 or hi[j] < 0)
        d = mean[j] if (best, m) in idx else -mean[j]
        if inb:
            grp.append(m)
        p(f'    {m:16s} d={d:+.4f} CI=[{lo[j]:+.4f},{hi[j]:+.4f}] {"in" if inb else "out"}')
    p(f'  => leading group: {grp}')
    summary[ds] = {'ranking': order, 'means': means, 'leading_group': grp,
                   'pairs_separated': sig, 'pairs_total': len(pairs)}
    del boot

with open(os.path.join(RESULTS, f'bootstrap_{METRIC}.txt'), 'w') as fh:
    fh.write('\n'.join(out) + '\n')
json.dump(summary, open(os.path.join(RESULTS, f'bootstrap_{METRIC}.json'), 'w'), indent=1)
print(f'[saved] results/bootstrap_{METRIC}.txt', flush=True)
