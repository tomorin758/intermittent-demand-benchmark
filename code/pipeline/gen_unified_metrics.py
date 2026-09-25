"""Step 3: recompute every metric for every model from the UNIFIED final-window arrays.

Outputs
  data/metrics_final/<ds>/<model>.json   -- full metric_comprehensive() dict (pooled)
  data/perseries_final/<ds>/<model>.csv  -- per-series metrics + ADI-CV2 class
  data/table_rows_final.csv              -- one row per (dataset, model, variant) in the layout
                                            of the current "*- Sheet1.csv" files, so that
                                            bench2026-paper/gen_tables.py can be pointed at it.

The pooled metrics use the repo's own utils.metrics.metric_comprehensive (same function that
produced the published numbers), with historical_data = first 70% of the history table in the
SAME column order as the predictions (the original permuted runs mismatched these).
"""
import os, sys, csv, json
import numpy as np

PAPER = '${PAPER_ROOT}/Projects/paper'
sys.path.insert(0, '${PAPER_ROOT}/Projects/PatchTST/PatchTST_supervised')
from utils.metrics import metric_comprehensive                      # noqa: E402

DS_ALL = {'parts': ('parts.csv', 3), 'fresh_retail': ('fresh_retail.csv', 7), 'm5': ('m5.csv', 28)}
_only = [a for a in sys.argv[1:] if a in DS_ALL]
DS = {k: v for k, v in DS_ALL.items() if not _only or k in _only}
C = 0.5
# models that are reported untruncated only (already emit exact zeros) /
# truncated only (never emit an exact zero) -- mirrors the paper's current tables
UNTRUNC_ONLY = {'DeepAR-negbin', 'DeepAR', 'TweedieGP', 'HurdleDLinear'}
TRUNC_ONLY = {'Croston', 'SBA', 'TSB'}


def load_hist(path):
    with open(path) as fh:
        rd = csv.reader(fh); hdr = next(rd); rows = list(rd)
    # hdr[0] is the date column; the series names are the remaining non-empty cells
    return [c for c in hdr if c][1:], np.array(
        [[float(x) if x not in ('', 'nan') else 0.0 for x in r[1:]] for r in rows], dtype=np.float64)


def perseries(P, T):
    """P,T: (H, N) -> dict of per-series arrays."""
    H, N = P.shape
    e = P - T
    out = {}
    out['ql_raw'] = 0.5 * np.abs(e).mean(axis=0)
    Pt = np.where(P < C, 0.0, P); Tt = np.where(T < C, 0.0, T)
    out['ql_trunc'] = 0.5 * np.abs(Pt - Tt).mean(axis=0)
    nz = T > 0
    cnt = nz.sum(axis=0)
    out['nz_count'] = cnt
    with np.errstate(invalid='ignore', divide='ignore'):
        out['mae_nz'] = np.where(cnt > 0, (np.where(nz, np.abs(e), 0)).sum(axis=0) / np.maximum(cnt, 1), np.nan)
        out['rmse_nz'] = np.where(cnt > 0, np.sqrt((np.where(nz, e * e, 0)).sum(axis=0) / np.maximum(cnt, 1)), np.nan)
    for tag, zt, zp in [('raw', T == 0, P == 0), ('trunc', Tt == 0, Pt == 0)]:
        tp = (zt & zp).sum(axis=0).astype(float)
        fp = (~zt & zp).sum(axis=0).astype(float)
        fn = (zt & ~zp).sum(axis=0).astype(float)
        pr = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        rc = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        out[f'P0_{tag}'] = pr; out[f'R0_{tag}'] = rc
        out[f'F1_{tag}'] = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(pr), where=(pr + rc) > 0)
        # keep the raw counts so that both the pooled and the per-series-mean convention
        # can be recomputed from the released per-series tables alone
        out[f'tp_{tag}'] = tp; out[f'fp_{tag}'] = fp; out[f'fn_{tag}'] = fn
    return out


def classes(hist):
    """ADI-CV^2 class per series, computed on the history window only."""
    cls = []
    for j in range(hist.shape[1]):
        s = hist[:, j]; pos = s[s > 0]
        if len(pos) < 2:
            cls.append('na'); continue
        adi = len(s) / len(pos)
        cv2 = (pos.std(ddof=1) / pos.mean()) ** 2
        cls.append(('smooth' if cv2 < 0.49 else 'erratic') if adi < 1.32
                   else ('intermittent' if cv2 < 0.49 else 'lumpy'))
    return cls


PCOLS = ['series', 'class', 'n_test', 'ql_raw', 'mae_raw', 'rmse_raw', 'mae_nz_raw', 'rmse_nz_raw',
         'prec0_raw', 'rec0_raw', 'f1_raw', 'tp0_raw', 'fp0_raw', 'fn0_raw',
         'ql_trunc', 'mae_trunc', 'rmse_trunc', 'mae_nz_trunc',
         'rmse_nz_trunc', 'prec0_trunc', 'rec0_trunc', 'f1_trunc', 'tp0_trunc', 'fp0_trunc', 'fn0_trunc']

rows_out = []
summary = {}
for ds, (hf, H) in DS.items():
    names, V = load_hist(os.path.join(PAPER, 'data/history', hf))
    L, N = V.shape
    hist = V[:int(L * 0.7)]
    cls = classes(hist)
    d = os.path.join(PAPER, 'data/final', ds)
    models = sorted(m for m in os.listdir(d) if os.path.isdir(os.path.join(d, m)))
    print('=' * 100); print(f'{ds}: {len(models)} models, history rows 0..{int(L*0.7)-1}, '
                           f'final window rows {L-H}..{L-1}')
    summary[ds] = {}
    for m in models:
        P = np.load(os.path.join(d, m, 'pred.npy')).astype(np.float64)     # (1,H,N)
        T = np.load(os.path.join(d, m, 'true.npy')).astype(np.float64)
        assert P.shape == T.shape and P.shape[1] == H and P.shape[2] == N, (m, P.shape, T.shape)
        met = metric_comprehensive(P, T, hist, c=C)
        # per-series
        ps = perseries(P[0], T[0])
        od = os.path.join(PAPER, 'data/perseries_final', ds); os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, f'{m}.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(PCOLS)
            for j in range(N):
                def f(x):
                    return '' if (isinstance(x, float) and np.isnan(x)) else f'{x:.6g}'
                w.writerow([names[j], cls[j], H,
                            f(ps['ql_raw'][j]), f(2 * ps['ql_raw'][j]), '', f(ps['mae_nz'][j]), f(ps['rmse_nz'][j]),
                            f(ps['P0_raw'][j]), f(ps['R0_raw'][j]), f(ps['F1_raw'][j]),
                            int(ps['tp_raw'][j]), int(ps['fp_raw'][j]), int(ps['fn_raw'][j]),
                            f(ps['ql_trunc'][j]), f(2 * ps['ql_trunc'][j]), '', f(ps['mae_nz'][j]), f(ps['rmse_nz'][j]),
                            f(ps['P0_trunc'][j]), f(ps['R0_trunc'][j]), f(ps['F1_trunc'][j]),
                            int(ps['tp_trunc'][j]), int(ps['fp_trunc'][j]), int(ps['fn_trunc'][j])])
        om = os.path.join(PAPER, 'data/metrics_final', ds); os.makedirs(om, exist_ok=True)
        with open(os.path.join(om, f'{m}.json'), 'w') as fh:
            json.dump({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in met.items()},
                      fh, indent=1, ensure_ascii=False)
        zero_frac = float((P[0] == 0).mean())
        summary[ds][m] = {'QL_raw': met['QL(0.5)'], 'QL_trunc': met[f'QL(0.5,c={C})'],
                          'F1_trunc': met[f'0F1(c={C})'], 'P0_trunc': met[f'0Precision(c={C})'],
                          'R0_trunc': met[f'0Recall(c={C})'], 'MAE_nz': met['MAE_nonzero'],
                          'RMSE_nz': met['RMSE_nonzero'], 'zero_frac_pred': zero_frac}
        # rows for the Sheet-style table
        def mk(variant, tag):
            return {'Dataset': ds, 'Model': m if variant == 'raw' else f'{m}(τ=0.5)',
                    'QL(0.5)': met['QL(0.5)'] if variant == 'raw' else met[f'QL(0.5,c={C})'],
                    'Precision': met[f'0Precision(c={C})'] if variant == 'trunc' else float('nan'),
                    'Recall': met[f'0Recall(c={C})'] if variant == 'trunc' else float('nan'),
                    'F1 score': met[f'0F1(c={C})'] if variant == 'trunc' else float('nan'),
                    'MAE': met['MAE_nonzero'] if variant == 'raw' else met[f'MAE_nonzero(c={C})'],
                    'RMSE': met['RMSE_nonzero'] if variant == 'raw' else met[f'RMSE_nonzero(c={C})'],
                    'NT': met if variant == 'raw' else None, 'T': met if variant == 'trunc' else None}
        if m not in UNTRUNC_ONLY:
            rows_out.append(mk('raw', 'NT'))
        if m not in TRUNC_ONLY:
            rows_out.append(mk('trunc', 'T'))
        print(f'  {m:16s} QL_raw={met["QL(0.5)"]:.4f} QL_tr={met[f"QL(0.5,c={C})"]:.4f} '
              f'F1_tr={met[f"0F1(c={C})"]:.4f}  exact-zero pred {zero_frac*100:5.1f}%')

json.dump(summary, open(os.path.join(PAPER, '.scratch/unify/step3_summary.json'), 'w'), indent=1)
with open(os.path.join(PAPER, 'data/table_rows_final.json'), 'w') as fh:
    json.dump(rows_out, fh, indent=1, default=str)
print('\n[done] metrics_final/, perseries_final/, data/table_rows_final.json')
