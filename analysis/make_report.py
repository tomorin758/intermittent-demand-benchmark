"""Generate .scratch/unify/UNIFY_REPORT.md: what the window unification changed.

Reads:
  data/metrics_final/<ds>/<model>.json     (unified, pooled)
  <sheet>.csv                              (published numbers)
Writes a markdown report with per-dataset comparison tables plus rank statistics.
"""
import csv, json, os
import numpy as np

PAPER = '${PAPER_ROOT}/Projects/paper'
DS = {'parts': ('Parts - Sheet1.csv', 'Parts'),
      'fresh_retail': ('Fresh Retail - Sheet1.csv', 'Fresh retail'),
      'm5': ('M5 - Sheet1.csv', 'M5')}
AXES = ['MAE', 'RMSE', '(q90, ∞) RMSE', 'intermittent RMSE', 'lumpy RMSE', 'smooth RMSE', 'erratic RMSE']
RENAME = {'DeepAR-negbin': 'DeepAR(negbin)', 'DeepAR': 'DeepAR', 'WaveNet-like': 'WaveNet-like'}


def num(s):
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return float('nan')


def sheet(path):
    rows = list(csv.reader(open(os.path.join(PAPER, path), encoding='utf-8-sig')))
    hdr = [c.strip() for c in rows[1]]
    out = {}
    for r in rows[2:]:
        if len(r) < 2 or not r[1].strip():
            continue
        name = r[1].strip(); trunc = '(τ=0.5)' in name
        key = name.replace('(τ=0.5)', '')
        d = out.setdefault(key, {})
        d['trunc' if trunc else 'raw'] = {
            'QL': num(r[hdr.index('QL(0.5)')]), 'F1': num(r[hdr.index('F1 score')]),
            'MAE': num(r[hdr.index('MAE')]), 'P': num(r[hdr.index('Precision')]),
            'R': num(r[hdr.index('Recall')]),
            **{a: num(r[hdr.index(a)]) for a in AXES}}
    return out


def ranks(vals):
    """vals: {model: value} -> {model: rank} (1 = smallest)"""
    ms = [m for m in vals if not np.isnan(vals[m])]
    order = sorted(ms, key=lambda m: vals[m])
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
    p = [(x, y) for x, y in zip(a, b) if not (np.isnan(x) or np.isnan(y))]
    if len(p) < 3:
        return float('nan')
    a_ = np.array([x for x, _ in p]); b_ = np.array([y for _, y in p])
    if a_.std() == 0 or b_.std() == 0:
        return float('nan')
    return float(np.corrcoef(a_, b_)[0, 1])


L = []
def w(s=''):
    L.append(s)


w('# Window unification — what changed')
w()
w('Unified protocol: every model forecasts the **final H periods** of the history table,')
w('with the ground truth taken from that table. Numeric changes are computed from')
w('`data/final/` (released arrays) against the published `*- Sheet1.csv`.')
w()
for ds, (shf, label) in DS.items():
    mdir = os.path.join(PAPER, 'data/metrics_final', ds)
    if not os.path.isdir(mdir):
        w(f'## {label}\n\n_(not regenerated yet)_\n'); continue
    sh = sheet(shf)
    met = {f[:-5]: json.load(open(os.path.join(mdir, f))) for f in os.listdir(mdir) if f.endswith('.json')}
    sh = {**sh, **{k: sh[v] for k, v in RENAME.items() if v in sh}}
    w(f'## {label}')
    w()
    w('| model | QL(0.5) pub | QL(0.5) unified | Δ | rank pub | rank new | F1 pub | F1 new |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|')
    ql_pub = {m: sh[m].get('raw', {}).get('QL', sh[m].get('trunc', {}).get('QL', float('nan')))
              for m in met if m in sh}
    ql_new = {m: met[m]['QL(0.5)'] for m in met}
    r_pub = ranks(ql_pub); r_new = ranks(ql_new)
    for m in sorted(met, key=lambda k: ql_new[k]):
        pub = ql_pub.get(m, float('nan')); new = ql_new[m]
        f1p = sh.get(m, {}).get('trunc', {}).get('F1', float('nan'))
        f1n = met[m]['0F1(c=0.5)']
        w(f'| {m} | {pub:.3f} | {new:.3f} | {new-pub:+.3f} | {r_pub.get(m, float("nan")):.0f} | '
          f'{r_new.get(m, float("nan")):.0f} | {f1p:.3f} | {f1n:.3f} |')
    w()
    # rank statistics on the truncated variant
    keys = sorted(met)
    r_mae = ranks({m: met[m]['MAE_nonzero(c=0.5)'] for m in keys})
    r_ql = ranks({m: met[m]['QL(0.5,c=0.5)'] for m in keys})
    w(f'* models: **{len(keys)}**   Spearman rho(rank by non-zero MAE, rank by QL(0.5)) = '
      f'**{pear([r_mae[m] for m in keys], [r_ql[m] for m in keys]):+.3f}**')
    best = {}
    def seg6(prefix):
        ks = [k for k in met[keys[0]] if k.startswith(prefix + '_Seg')]
        ks.sort(key=lambda k: int(k.split('Seg')[1].split('_')[0]))
        return ks[-1] if ks else None

    for ax, key, high in [('QL(0.5) raw', 'QL(0.5)', False), ('QL(0.5) trunc', 'QL(0.5,c=0.5)', False),
                          ('MAE>0', 'MAE_nonzero', False), ('MAE>0 trunc', 'MAE_nonzero(c=0.5)', False),
                          ('RMSE>0', 'RMSE_nonzero', False), ('0F1', '0F1(c=0.5)', True),
                          ('tail (q90,inf) raw', 'NT_' + (seg6('NT') or ''), False),
                          ('tail (q90,inf) trunc', 'T_' + (seg6('T') or ''), False),
                          ('lumpy RMSE raw', 'NT_Lumpy RMSE', False),
                          ('lumpy RMSE trunc', 'T_Lumpy RMSE', False)]:
        if key not in met[keys[0]]:
            continue
        vals = {m: met[m][key] for m in keys if met[m].get(key) is not None}
        b = (max if high else min)(vals, key=lambda k: vals[k])
        best[ax] = (b, vals[b])
    w('* best model per axis: ' + '; '.join(f'{ax} → **{b}** ({v:.3f})' for ax, (b, v) in best.items()))
    w()
open(os.path.join(PAPER, '.scratch/unify/UNIFY_REPORT.md'), 'w').write('\n'.join(L) + '\n')
print('[saved] .scratch/unify/UNIFY_REPORT.md')
