"""Trivial reference predictors for the intermittent-demand benchmark.

An evaluation of intermittent demand has to answer a prior question: what does a predictor that
never attempts to forecast demand score?  This script computes five such references on exactly the
same evaluation window, ground truth and metric conventions as the paper:

    zero            always predict 0 (the median-optimal forecast when most points are zero)
    naive           last observed value of the history table
    seasonal naive  the value one seasonal cycle earlier (12 months on Parts, 7 days elsewhere)
    nzmean          constant = mean of the non-zero history values
    mean            constant = mean of the whole history

Metrics match the paper: QL(0.5) = 0.5 * MAE over all evaluation points (untruncated), and, after
truncating both sides at c = 0.5, the zero-detection precision/recall/F1 plus the truncated
QL(0.5).  The zero class is the positive class, so "always predict 0" has recall 1 by construction
and its precision equals the zero fraction of the window -- which is why zero-detection F1 is not
comparable across datasets with different zero rates.

The references are *not* part of the subject pool: they are reported next to it, never ranked
inside it.  `rank_*_among_subjects` gives the position the reference would take if it were inserted
into the ranking, which is what the paper quotes.

Inputs (override with HISTORY_DIR / POOLED_DIR / RESULTS_DIR):
    data/<dataset>.csv                  history table (ships for parts, fresh_retail)
    metrics/pooled/<dataset>/*.json     subject metrics, used only for the rank positions
M5's history table is not redistributed; set M5_HISTORY to its path to recompute the M5 row,
otherwise the shipped results/trivial_baselines.csv keeps the published values.

Usage: python analysis/trivial_baselines.py
"""
import csv, glob, json, os, sys
import numpy as np

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY = os.environ.get('HISTORY_DIR', os.path.join(BUNDLE, 'data'))
POOLED = os.environ.get('POOLED_DIR', os.path.join(BUNDLE, 'metrics', 'pooled'))
RESULTS = os.environ.get('RESULTS_DIR', os.path.join(BUNDLE, 'results'))
os.makedirs(RESULTS, exist_ok=True)

DS = ['parts', 'fresh_retail', 'm5']
H = {'parts': 3, 'fresh_retail': 7, 'm5': 28}
LAG = {'parts': 12, 'fresh_retail': 7, 'm5': 7}
C = 0.5


def read_history(path):
    rows = list(csv.reader(open(path, encoding='utf-8-sig')))
    names = rows[0][1:]
    V = np.array([[float(x) if x not in ('', 'nan') else 0.0 for x in r[1:]] for r in rows[1:]],
                 dtype=np.float64)
    return names, V


def metrics(P, T):
    """QL(0.5) untruncated, then the truncated pair and the zero-detection scores."""
    ql = 0.5 * float(np.mean(np.abs(T - P)))
    Pc, Tc = (P >= C), (T >= C)                      # True = non-zero after truncation
    TP = int((~Tc & ~Pc).sum()); FP = int((Tc & ~Pc).sum()); FN = int((~Tc & Pc).sum())
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    ql_t = 0.5 * float(np.mean(np.abs(np.where(Tc, T, 0.0) - np.where(Pc, P, 0.0))))
    return ql, ql_t, prec, rec, f1


def subject_metrics(ds):
    d = os.path.join(POOLED, ds)
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        with open(f) as fh:
            out[os.path.basename(f)[:-5]] = json.load(fh)
    return out


def rank_among_subjects(value, subj, key, higher_is_better):
    vals = [v[key] for v in subj.values() if isinstance(v.get(key), float)]
    if higher_is_better:
        return 1 + sum(1 for v in vals if v > value)
    return 1 + sum(1 for v in vals if v < value)


HEADER = ['dataset', 'predictor', 'ql_raw', 'ql_trunc', 'precision0', 'recall0', 'f1',
          'n_subjects', 'rank_ql_among_subjects', 'rank_f1_among_subjects']
OUT = os.path.join(RESULTS, 'trivial_baselines.csv')

available = []
for ds in DS:
    hist = os.path.join(HISTORY, '%s.csv' % ds)
    if ds == 'm5' and os.environ.get('M5_HISTORY'):
        hist = os.environ['M5_HISTORY']
    if os.path.exists(hist):
        available.append(ds)
if os.path.exists(OUT):
    with open(OUT) as fh:
        kept = [r for r in csv.reader(fh) if r and r[0] not in available and r[0] != 'dataset']
rows = [HEADER] + kept
print('recomputing: %s; keeping published rows for: %s'
      % (', '.join(available), ', '.join(sorted({r[0] for r in kept})) or 'none'), flush=True)

for ds in DS:
    hist = os.path.join(HISTORY, '%s.csv' % ds)
    if ds == 'm5' and os.environ.get('M5_HISTORY'):
        hist = os.environ['M5_HISTORY']
    if not os.path.exists(hist):
        print('[skip] %s: history table not available (set M5_HISTORY)' % ds, flush=True)
        continue
    names, V = read_history(hist)
    h = H[ds]
    T, train = V[-h:], V[:-h]
    P_ = {
        'zero': np.zeros_like(T),
        'naive': np.tile(train[-1], (h, 1)),
        'seasonal naive': V[-h - LAG[ds]:-LAG[ds]],
        'nzmean': np.tile(np.array([s[s > 0].mean() if (s > 0).any() else 0.0 for s in train.T]),
                          (h, 1)),
        'mean': np.tile(train.mean(axis=0), (h, 1)),
    }
    subj = subject_metrics(ds)
    print('%s: %d series, window %d steps, %d subjects, %d zeros in window (%.1f%%)'
          % (ds, V.shape[1], h, len(subj), int((T == 0).sum()), 100 * float((T == 0).mean())))
    for name, P in P_.items():
        ql, ql_t, prec, rec, f1 = metrics(P.ravel(), T.ravel())
        rq = rank_among_subjects(ql, subj, 'QL(0.5)', False)
        rf = rank_among_subjects(f1, subj, '0F1(c=0.5)', True)
        print('  %-15s QL=%.3f (rank %d/%d)  F1=%.3f (rank %d/%d)'
              % (name, ql, rq, len(subj) + 1, f1, rf, len(subj) + 1))
        rows.append([ds, name, '%.6f' % ql, '%.6f' % ql_t, '%.6f' % prec, '%.6f' % rec,
                     '%.6f' % f1, len(subj), rq, rf])

with open(OUT, 'w', newline='') as fh:
    csv.writer(fh).writerows(rows)
print('[saved] %s (%d rows)' % (os.path.relpath(OUT, BUNDLE), len(rows) - 1))
