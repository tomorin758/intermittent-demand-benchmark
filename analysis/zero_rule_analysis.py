"""Zero-rule diagnostics: exact-zero abstention and the error decomposition.

Two quantities reported in the paper need the per-model prediction arrays, which are not
redistributed (see the README, "Scope of this bundle"), so their values are shipped as
results/zero_rule_analysis.csv by this script:

  * what a model scores when ``predicted zero'' is its exact zero rather than the protocol's
    truncation device, for the models that can abstain at all (Section 4.1 of the paper);
  * where a model gains and loses against the always-zero reference predictor -- the error on
    zero-demand points against the error on positive-demand points (Section 5.1).

Both are pooled over the same held-out window and the same ground truth as every other number in
the paper. The always-zero reference predicts 0 everywhere, so its error on a zero-demand point is 0
and its error on a positive-demand point is that point's demand.

Usage:  python analysis/zero_rule_analysis.py [dataset ...]      # default: parts fresh_retail m5
        PRED_DIR points at <dataset>/<model>/{pred,true}.npy, default data/final.
Reads the prediction arrays, writes results/zero_rule_analysis.csv (override with RESULTS_DIR).
"""
import csv, os, sys
import numpy as np

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.environ.get('PRED_DIR', os.path.join(BUNDLE, 'data', 'final'))
RESULTS = os.environ.get('RESULTS_DIR', os.path.join(BUNDLE, 'results'))
DS = [a for a in sys.argv[1:] if a] or ['parts', 'fresh_retail', 'm5']
C = 0.5                      # the protocol's truncation threshold

_missing = [d for d in DS if not os.path.isdir(os.path.join(PRED, d))]
if _missing:
    raise SystemExit(
        'missing prediction arrays in %s: %s\n'
        'The per-model prediction arrays are not redistributed, so these diagnostics cannot be '
        'recomputed from the bundle alone; their values ship as results/zero_rule_analysis.csv '
        '(see the README, "Scope of this bundle"). Point PRED_DIR at a directory of '
        '<dataset>/<model>/{pred,true}.npy to recompute them.'
        % (PRED, ', '.join(_missing)))

HEADER = ['dataset', 'model', 'n_points', 'n_zero', 'n_nonzero', 'exact_zero_rate',
          'precision_exact', 'recall_exact', 'f1_exact',
          'precision_protocol', 'recall_protocol', 'f1_protocol',
          'mae_all', 'mae_all_reference', 'err_zero_points', 'err_positive_points',
          'mae_positive_reference', 'n_called_nonzero_correctly',
          'mae_called_nonzero_correctly', 'mae_reference_on_those_points']


def prf(pred_zero, actual_zero):
    tp = int(np.sum(actual_zero & pred_zero))
    fp = int(np.sum(~actual_zero & pred_zero))
    fn = int(np.sum(actual_zero & ~pred_zero))
    prec = tp / (tp + fp) if tp + fp else float('nan')
    rec = tp / (tp + fn) if tp + fn else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec > 0 else float('nan')
    return prec, rec, f1


rows = []
for ds in DS:
    d = os.path.join(PRED, ds)
    models = sorted(m for m in os.listdir(d) if os.path.isdir(os.path.join(d, m)))
    # ground truth is identical for every model on a dataset
    y = np.load(os.path.join(d, models[0], 'true.npy')).ravel().astype(np.float64)
    zero = y == 0
    pos = ~zero
    n = y.size
    n0, n1 = int(zero.sum()), int(pos.sum())
    ybar = float(y[pos].mean())
    ref_all = n1 * ybar / n
    print('%s: %d models, %d points (%d zero, %d positive), always-zero pooled MAE %.4f'
          % (ds, len(models), n, n0, n1, ref_all), flush=True)
    for m in models:
        p = np.load(os.path.join(d, m, 'pred.npy')).ravel().astype(np.float64)
        e0 = float(np.abs(p[zero]).mean())
        e1 = float(np.abs(p[pos] - y[pos]).mean())
        mae_all = (n0 * e0 + n1 * e1) / n
        pz_exact = p == 0
        pz_prot = p < C
        pe, re_, fe = prf(pz_exact, zero)
        pp, rp, fp_ = prf(pz_prot, zero)
        called = pos & (p >= C)
        cond_n = int(called.sum())
        cond_mae = float(np.abs(p[called] - y[called]).mean()) if cond_n else float('nan')
        cond_ref = float(y[called].mean()) if cond_n else float('nan')
        rows.append([ds, m, n, n0, n1, float(pz_exact.mean()),
                     pe, re_, fe, pp, rp, fp_,
                     mae_all, ref_all, e0, e1, ybar, cond_n, cond_mae, cond_ref])
    print('  [saved] per-model rows computed', flush=True)

os.makedirs(RESULTS, exist_ok=True)
out = os.path.join(RESULTS, 'zero_rule_analysis.csv')
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(HEADER)
    for r in rows:
        w.writerow(r[:5] + ['%.6f' % v if isinstance(v, float) else v for v in r[5:]])
print('[saved] %s (%d rows)' % (os.path.relpath(out, BUNDLE), len(rows)), flush=True)
