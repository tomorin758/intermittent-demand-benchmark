"""Within-family rankings: does the leading model change between families or inside them?

The paper's Section 4.1 ranks models within the five families defined in Section 3.1 -- classical
intermittent-demand methods, zero-inflated and count-likelihood models, per-dataset-trained deep
regressors, zero-shot foundation models and head-only fine-tuned foundation models -- and reports how
many within-family pairs the paired bootstrap actually separates. That is how the paper separates a
change of leading family from a reshuffle inside a family, and how it tests whether fine-tuning
budget orders the foundation models.

The analysis needs only the per-series metric tables that ship in metrics/perseries/, so it runs from
the bundle alone. The resampling is the one used everywhere else in the paper: identical resample for
both models of a pair, B = 4000 Poisson weights, 95% percentile interval, seed 20260924, and every
pair of a dataset is drawn so that the random stream stays the same as in
analysis/bootstrap_final.py.

Usage:  python analysis/group_analysis.py [dataset ...]      # default: parts fresh_retail m5
Writes results/group_analysis.csv (one row per model and dataset) and results/group_analysis.txt
(the within-family summary quoted in Section 4.1).
"""
import csv, itertools, os, sys
import numpy as np

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSERIES = os.environ.get('PERSERIES_DIR', os.path.join(BUNDLE, 'metrics', 'perseries'))
RESULTS = os.environ.get('RESULTS_DIR', os.path.join(BUNDLE, 'results'))
DS = [a for a in sys.argv[1:] if a] or ['parts', 'fresh_retail', 'm5']
B, CHUNK, ALPHA = 4000, 500, 0.05
SEED = 20260924

# Section 3.1 of the paper; Timer appears on M5 only, so the last group has two members elsewhere.
GROUPS = [
    ('classical intermittent-demand', ['Croston', 'SBA', 'TSB']),
    ('zero-inflated and count-likelihood', ['DeepAR', 'DeepAR-negbin', 'TweedieGP', 'HurdleDLinear']),
    ('per-dataset-trained deep regressor',
     ['PatchTST', 'PatchMixer', 'PatchTSMixer', 'PatchTCN', 'DLinear', 'LSTM', 'TCN',
      'WaveNet-like', 'Autoformer', 'FEDformer']),
    ('foundation model, zero-shot', ['TimesFM', 'Moirai', 'Chronos']),
    ('foundation model, head-only fine-tuned', ['TTM', 'MOMENT', 'Timer']),
]


def load(ds):
    d = os.path.join(PERSERIES, ds)
    tab = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith('.csv'):
            continue
        with open(os.path.join(d, f)) as fh:
            rows = list(csv.DictReader(fh))
        tab[f[:-4]] = {r['series']: float(r['ql_raw']) for r in rows if r['ql_raw'] not in ('', None)}
    return tab


rng = np.random.default_rng(SEED)
rows = []
report = []
for ds in DS:
    if not os.path.isdir(os.path.join(PERSERIES, ds)):
        report.append('')
        report.append('%s: skipped -- metrics/perseries/%s is not part of this copy' % (ds, ds))
        continue
    tab = load(ds)
    common = None
    for m in tab:
        common = set(tab[m]) if common is None else (common & set(tab[m]))
    common = sorted(common)
    n = len(common)
    means = {m: float(np.mean([tab[m][s] for s in common])) for m in tab}
    order = sorted(tab, key=lambda m: means[m])
    pairs = list(itertools.combinations(order, 2))
    D = np.empty((n, len(pairs)), dtype=np.float32)
    for i, (a, b) in enumerate(pairs):
        D[:, i] = (np.array([tab[a][s] for s in common], dtype=np.float32)
                   - np.array([tab[b][s] for s in common], dtype=np.float32))
    boot = np.empty((B, len(pairs)), dtype=np.float32)
    for s0 in range(0, B, CHUNK):
        k = min(CHUNK, B - s0)
        W = rng.poisson(1.0, size=(k, n)).astype(np.float32)
        boot[s0:s0 + k] = (W @ D) / n
    lo = np.percentile(boot, 100 * ALPHA / 2, axis=0)
    hi = np.percentile(boot, 100 * (1 - ALPHA / 2), axis=0)
    sep = (lo > 0) | (hi < 0)
    idx = {p: i for i, p in enumerate(pairs)}

    def separated(a, b):
        i = idx[(a, b)] if (a, b) in idx else idx[(b, a)]
        return bool(sep[i])

    report.append('')
    report.append('%s: %d models, %d series' % (ds, len(order), n))
    for gname, members in GROUPS:
        ms = [m for m in members if m in means]
        if not ms:
            continue
        gs = sorted(ms, key=lambda m: means[m])
        npairs = len(gs) * (len(gs) - 1) // 2
        nsep = sum(1 for a, b in itertools.combinations(gs, 2) if separated(a, b))
        report.append('  %-42s leader %-14s %.4f   within-family separated %d/%d'
                      % (gname, gs[0], means[gs[0]], nsep, npairs))
        report.append('    ' + ', '.join('%s %.3f' % (m, means[m]) for m in gs))
        for rank, m in enumerate(gs, 1):
            rows.append([ds, gname, m, '%.6f' % means[m], n, rank,
                         1 if rank == 1 else 0,
                         '' if rank == 1 else int(separated(gs[0], m))])

os.makedirs(RESULTS, exist_ok=True)
out = os.path.join(RESULTS, 'group_analysis.csv')
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['dataset', 'family', 'model', 'ql_raw', 'n_series', 'rank_in_family',
                'is_family_leader', 'separated_from_family_leader'])
    w.writerows(rows)
txt = os.path.join(RESULTS, 'group_analysis.txt')
with open(txt, 'w') as fh:
    fh.write('\n'.join(report) + '\n')
print('\n'.join(report))
print('[saved] %s (%d rows) and %s'
      % (os.path.relpath(out, BUNDLE), len(rows), os.path.relpath(txt, BUNDLE)))
