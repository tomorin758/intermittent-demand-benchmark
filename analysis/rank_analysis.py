"""Rank / metric-independence analysis on a set of Sheet-format CSVs.

Usage:  python rank_analysis.py <parts.csv> <fresh.csv> <m5.csv> <tag>
Writes .scratch/unify/rank_analysis_<tag>.txt and prints the same report.

Rows used are the TRUNCATED (tau=0.5) rows, i.e. the c=0.5 variant, matching the
published analysis; the reference axis is the non-zero MAE column.
"""
import csv, sys, math, os
from itertools import combinations

PAPER = '${PAPER_ROOT}/Projects/paper'
AXES = ['QL(0.5)', 'Precision', 'Recall', 'F1 score', 'MAE', 'RMSE', '[0, q10] RMSE',
        '(q10, q25] RMSE', '(q25, q50] RMSE', '(q50, q75] RMSE', '(q75, q90] RMSE',
        '(q90, ∞) RMSE', 'intermittent RMSE', 'lumpy RMSE', 'smooth RMSE', 'erratic RMSE']


def num(s):
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return float('nan')


ZERO_COLS = {'Precision', 'Recall', 'F1 score'}


def load(path):
    """Per column, take the row variant the paper uses: the untruncated row for the
    point-forecast columns and the truncated row for the zero-detection columns."""
    rows = list(csv.reader(open(path, encoding='utf-8-sig')))
    hdr = [c.strip() for c in rows[1]]
    idx = {c: hdr.index(c) for c in AXES}
    D = {}
    for r in rows[2:]:
        if len(r) < 2 or not r[1].strip():
            continue
        trunc = '(τ=0.5)' in r[1]
        name = r[1].strip().replace('(τ=0.5)', '')
        d = D.setdefault(name, {})
        for c in AXES:
            if trunc == (c in ZERO_COLS):
                d[c] = num(r[idx[c]])
    return {k: v for k, v in D.items() if len(v) == len(AXES)}


def rank(vals, keys):
    """average ranks (1 = smallest value)"""
    order = sorted(keys, key=lambda k: vals[k])
    out = {}; i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return out


def pear(a, b):
    p = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    n = len(p)
    if n < 3:
        return float('nan')
    ma = sum(x for x, _ in p) / n; mb = sum(y for _, y in p) / n
    da = math.sqrt(sum((x - ma) ** 2 for x, _ in p)); db = math.sqrt(sum((y - mb) ** 2 for _, y in p))
    return 0.0 if da * db == 0 else sum((x - ma) * (y - mb) for x, y in p) / (da * db)


def spearman(D, ax1, ax2):
    keys = [k for k in D if not (math.isnan(D[k][ax1]) or math.isnan(D[k][ax2]))]
    r1 = rank({k: D[k][ax1] for k in keys}, keys)
    r2 = rank({k: D[k][ax2] for k in keys}, keys)
    return pear([r1[k] for k in keys], [r2[k] for k in keys]), keys, r1, r2


def main():
    files = sys.argv[1:4]
    tag = sys.argv[4] if len(sys.argv) > 4 else 'out'
    out = []
    def p(s=''):
        print(s); out.append(s)
    sets = {}
    for f in files:
        D = load(f)
        sets[os.path.basename(f).replace(' - Sheet1.csv', '').replace('_unified.csv', '')] = D
    p('=' * 92)
    p('A. active columns (a column counts as active if at least one model has a value)')
    p('=' * 92)
    for name, D in sets.items():
        act = [a for a in AXES if any(not math.isnan(D[m][a]) for m in D)]
        dead = [a for a in AXES if a not in act]
        p(f'  {name:20s} models={len(D):3d}  active={len(act)}/{len(AXES)}  always empty: {dead}')

    p(); p('=' * 92); p('B. redundancy: |rho| >= 0.95 between metric pairs'); p('=' * 92)
    for name, D in sets.items():
        red = []
        for a, b in combinations(AXES, 2):
            r, keys, _, _ = spearman(D, a, b)
            if not math.isnan(r) and abs(r) >= 0.95:
                red.append((a, b, r))
        p(f'  {name}: {len(red)} redundant pairs')
        for a, b, r in sorted(red, key=lambda t: -abs(t[2])):
            p(f'     {a:20s} ~ {b:20s} {r:+.3f}')

    p(); p('=' * 92); p('C. rank movement relative to the non-zero MAE axis'); p('=' * 92)
    for name, D in sets.items():
        base = 'MAE'
        keys = [k for k in D if not math.isnan(D[k][base])]
        rb = rank({k: D[k][base] for k in keys}, keys)
        p(f'\n  {name}  (base = non-zero MAE, {len(keys)} models)')
        for ax in AXES:
            if ax == base:
                continue
            r, kk, _, r1 = spearman(D, base, ax)
            if math.isnan(r):
                continue
            diffs = [abs(rb[k] - r1[k]) for k in kk]
            big = sum(1 for d in diffs if d >= 5)
            p(f'    vs {ax:20s} rho={r:+.3f}   |rank diff|>=5: {big:2d}/{len(kk)}  max={max(diffs):.0f}')

    p(); p('=' * 92); p('D. best model per axis'); p('=' * 92)
    for name, D in sets.items():
        p(f'\n  {name}')
        for ax in AXES:
            vals = {m: D[m][ax] for m in D if not math.isnan(D[m][ax])}
            if not vals:
                continue
            best = min(vals, key=lambda m: vals[m])
            p(f'    {ax:20s} {best:18s} {vals[best]:.3f}')

    with open(os.path.join(PAPER, f'.scratch/unify/rank_analysis_{tag}.txt'), 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    p(f'\n[saved] .scratch/unify/rank_analysis_{tag}.txt')


main()
