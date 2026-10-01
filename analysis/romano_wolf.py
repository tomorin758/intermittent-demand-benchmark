"""Romano-Wolf stepdown correction for the pairwise comparisons of the benchmark.

The paper reports a paired bootstrap over series for every model pair (B=4000, Poisson weights,
95% percentile CI).  A dataset with 22-23 models yields 231-253 comparisons, so per-pair CIs
overstate how many differences survive a family-wise criterion.  This script reuses exactly the
same resampling scheme (same seed, same B, same weights, same metric definitions as
analysis/bootstrap_final.py) and adds:

  * the naive count  "CI excludes 0"                       (the number quoted in the paper)
  * the Romano-Wolf stepdown count over the same family    (FWER-controlled, exploits dependence)
  * a stepdown-controlled comparison of every model against the leader (family of m-1 tests),
    which turns the heuristic "CI overlaps the best" leading group into a controlled one.

Stepdown (two-sided, percentile-t): with the family of pairs K, observed t_k = |d_k| / se_k and
bootstrap null statistics z*_{b,k} = (d*_{b,k} - d_k) / se_k, order pairs by decreasing t_k and
walk down: at each step the critical value is the 95% quantile of max_{k in active} |z*_{b,k}|;
reject if t_k exceeds it, drop the pair from the active set, recompute, continue; stop at the
first pair that does not exceed.  Under subset pivotality this controls the family-wise error
rate at 5% without assuming independence between comparisons.

Usage: python analysis/romano_wolf.py [metric ...]      # default: ql_raw f1_trunc
Reads metrics/perseries/<dataset>/*.csv, writes results/romano_wolf_<metric>.{txt,json}.
Overrides: PERSERIES_DIR, RESULTS_DIR.
"""
import csv, os, sys, itertools, json
import numpy as np

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSERIES = os.environ.get('PERSERIES_DIR', os.path.join(BUNDLE, 'metrics', 'perseries'))
RESULTS = os.environ.get('RESULTS_DIR', os.path.join(BUNDLE, 'results'))
os.makedirs(RESULTS, exist_ok=True)
DS = ['parts', 'fresh_retail', 'm5']
METRICS = sys.argv[1:] or ['ql_raw', 'f1_trunc']
COL = {'ql_raw': 'ql_raw', 'ql_trunc': 'ql_trunc', 'f1_trunc': 'f1_trunc',
       'mae_nz_raw': 'mae_nz_raw'}
HIGHER_BETTER = {'f1_trunc'}
B, CHUNK, ALPHA = 4000, 500, 0.05
COUNTS = ('tp0_trunc', 'fp0_trunc', 'fn0_trunc')


def f1(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)


def load(ds, counts=False, col='ql_raw'):
    d = os.path.join(PERSERIES, ds)
    tab = {}
    if not os.path.isdir(d):
        return tab
    for f in sorted(os.listdir(d)):
        if not f.endswith('.csv'):
            continue
        with open(os.path.join(d, f)) as fh:
            rows = list(csv.DictReader(fh))
        if counts:
            tab[f[:-4]] = {r['series']: tuple(float(r[c]) for c in COUNTS) for r in rows}
        else:
            tab[f[:-4]] = {r['series']: float(r[col]) for r in rows if r[col] not in ('', None)}
    return tab


def bootstrap_pairs(ds, metric):
    """Return (order, means, pairs, boot[B, n_pairs], mean_diff[n_pairs]) - identical resampling
    to analysis/bootstrap_final.py for the same metric."""
    counts = metric == 'f1_trunc'
    tab = load(ds, counts=counts, col=COL[metric])
    if len(tab) < 2:
        return None
    common = None
    for m in tab:
        common = set(tab[m]) if common is None else (common & set(tab[m]))
    common = sorted(common)
    n = len(common)
    if counts:
        M = {m: np.array([tab[m][s] for s in common], dtype=np.float64) for m in tab}
        means = {m: float(f1(M[m][:, 0].sum(), M[m][:, 1].sum(), M[m][:, 2].sum())) for m in tab}
    else:
        means = {m: float(np.mean([tab[m][s] for s in common])) for m in tab}
    order = sorted(tab, key=lambda m: -means[m] if metric in HIGHER_BETTER else means[m])
    pairs = list(itertools.combinations(order, 2))
    boot = np.empty((B, len(pairs)), dtype=np.float64)
    if counts:
        for s0 in range(0, B, CHUNK):
            k = min(CHUNK, B - s0)
            W = rng.poisson(1.0, size=(k, n)).astype(np.float64)
            F = {m: f1(W @ M[m][:, 0], W @ M[m][:, 1], W @ M[m][:, 2]) for m in order}
            for i, (a, b) in enumerate(pairs):
                boot[s0:s0 + k, i] = F[a] - F[b]
        mean = np.array([f1(M[a][:, 0].sum(), M[a][:, 1].sum(), M[a][:, 2].sum())
                         - f1(M[b][:, 0].sum(), M[b][:, 1].sum(), M[b][:, 2].sum())
                         for a, b in pairs])
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
    return order, means, pairs, boot, mean, n


def stepdown(mean, boot, active=None):
    """Romano-Wolf stepdown on the columns of `boot` (already differences).  Returns bool mask."""
    se = boot.std(axis=0, ddof=1)
    se = np.maximum(se, 1e-12)
    t = np.abs(mean) / se
    z = (boot - mean) / se                       # bootstrap null statistics
    n_pairs = boot.shape[1]
    act = np.ones(n_pairs, dtype=bool) if active is None else active.copy()
    rej = np.zeros(n_pairs, dtype=bool)
    order = np.argsort(-t)
    for i in order:
        if not act[i]:
            continue
        crit = np.quantile(np.abs(z[:, act]).max(axis=1), 1 - ALPHA)
        if t[i] > crit:
            rej[i] = True
            act[i] = False
        else:
            break
    return rej, t, se


out = []


def p(s=''):
    print(s, flush=True)
    out.append(s)


for metric in METRICS:
    out.clear()                                  # one result file per metric
    rng = np.random.default_rng(20260924)        # same stream as bootstrap_final.py
    p('=' * 100)
    p(f'Romano-Wolf stepdown   metric = {COL[metric]}   B = {B}   alpha = {ALPHA}   '
      f'(same resampling as results/bootstrap_{metric}.txt)')
    p('=' * 100)
    summary = {}
    skipped = []
    for ds in DS:
        if not os.path.isdir(os.path.join(PERSERIES, ds)):
            p(f'\n{ds}: skipped -- metrics/perseries/{ds} is not part of this copy '
              f'(see SUPPLEMENTARY_NOTE.md); the full bundle is at the anonymised mirror')
            skipped.append(ds); continue
        res = bootstrap_pairs(ds, metric)
        if res is None:
            p(f'\n{ds}: not enough models'); skipped.append(ds); continue
        order, means, pairs, boot, mean, n = res
        lo = np.percentile(boot, 100 * ALPHA / 2, axis=0)
        hi = np.percentile(boot, 100 * (1 - ALPHA / 2), axis=0)
        naive = (lo > 0) | (hi < 0)
        rej, t, se = stepdown(mean, boot)
        p(''); p(f'{ds}: {len(order)} models, {n} series, {len(pairs)} pairs')
        p(f'  CI excludes 0 (naive, as quoted in the paper): {int(naive.sum())} / {len(pairs)}')
        p(f'  Romano-Wolf stepdown (FWER 5%):                {int(rej.sum())} / {len(pairs)}')
        # leader family: every model vs the best
        best = order[0]
        idx = {pr: i for i, pr in enumerate(pairs)}
        lead = [idx[(best, m)] for m in order[1:]]
        rej_lead, t_lead, _ = stepdown(mean, boot, active=np.isin(np.arange(len(pairs)), lead))
        naive_lead = [j for j in lead if naive[j]]
        kept = [best] + [m for m in order[1:] if not rej_lead[idx[(best, m)]]]
        sep = [m for m in order[1:] if rej_lead[idx[(best, m)]]]
        p(f'  vs the leader ({best}): naive separates {len(naive_lead)} / {len(order) - 1}; '
          f'stepdown separates {len(sep)} / {len(order) - 1}')
        p(f'  stepdown-controlled leading group: {kept}')
        p('  pairs separated by the per-pair CI but not by stepdown (largest t first):')
        for i in np.argsort(-t):
            if naive[i] and not rej[i]:
                a, b = pairs[i]
                p(f'    {a:16s} vs {b:16s} d={mean[i]:+.4f} CI=[{lo[i]:+.4f},{hi[i]:+.4f}] '
                  f't={t[i]:5.2f}')
        summary[ds] = {'metric': COL[metric], 'models': order, 'series': n,
                       'pairs_total': len(pairs), 'naive_separated': int(naive.sum()),
                       'stepdown_separated': int(rej.sum()),
                       'leader': best, 'leader_family': len(order) - 1,
                       'leader_naive_separated': len(naive_lead),
                       'leader_stepdown_separated': len(sep),
                       'stepdown_leading_group': kept}
        del boot
    # a partial run (some datasets absent) must not overwrite the complete shipped report
    tag = '_partial' if skipped else ''
    with open(os.path.join(RESULTS, f'romano_wolf_{metric}{tag}.txt'), 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    json.dump(summary, open(os.path.join(RESULTS, f'romano_wolf_{metric}{tag}.json'), 'w'), indent=1)
    print(f'[saved] results/romano_wolf_{metric}{tag}.txt', flush=True)
