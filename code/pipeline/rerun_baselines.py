"""Step 2a: re-run the deterministic intermittent-demand baselines (Croston / SBA / TSB)
on the FINAL window of each history table.

For each (dataset, method):
  * sanity check: reproduce the ORIGINAL window (origin = int(0.8*L)) and compare with the
    stored pred.npy -- proves the re-run uses the same code path / parameters;
  * then predict the final H periods (origin = L-H) from the full preceding history and
    write data/final/<ds>/<model>/{pred,true}.npy.
"""
import os, sys, csv, json
import numpy as np

REPO = '${PAPER_ROOT}/Projects/PatchTST'
sys.path.insert(0, REPO)
os.chdir(REPO)  # intermittent_baseline imports 'PatchTST_supervised/utils/metrics' relatively

from intermittent_baseline import make_forecast   # noqa: E402
from croston_baseline import croston_forecast     # noqa: E402

HF = {'parts': 'parts.csv', 'fresh_retail': 'fresh_retail.csv', 'm5': 'm5.csv'}
PAPER = '${PAPER_ROOT}/Projects/paper'
OUT = os.path.join(PAPER, 'data/final')
ALPHA = 0.1


def load_hist(path):
    with open(path) as fh:
        rd = csv.reader(fh); hdr = next(rd); rows = list(rd)
    # hdr[0] is the date column (the caller currently uses only V, but keep the names aligned)
    return [c for c in hdr if c][1:], np.array(
        [[float(x) if x not in ('', 'nan') else 0.0 for x in r[1:]] for r in rows], dtype=np.float64)


def predict(method, insample_matrix, H):
    """insample_matrix: (T, C) -> (H, C)"""
    T, C = insample_matrix.shape
    out = np.zeros((H, C), dtype=np.float64)
    for j in range(C):
        y = insample_matrix[:, j]
        if method == 'croston':
            out[:, j] = croston_forecast(y, forecast_horizon=H, alpha=ALPHA)
        else:
            out[:, j] = make_forecast(method, y, H, ALPHA, ALPHA, ALPHA)
    return out


report = {}
for ds, f in HF.items():
    names, V = load_hist(os.path.join(PAPER, 'data/history', f))
    L, N = V.shape
    H = {'parts': 3, 'fresh_retail': 7, 'm5': 28}[ds]
    report[ds] = {}
    print('=' * 90); print(f'{ds}: L={L} C={N} H={H}')
    for method, model in [('croston', 'Croston'), ('sba', 'SBA'), ('tsb', 'TSB')]:
        # --- (a) reproduce the original window
        origin0 = int(L * 0.8)
        p0 = predict(method, V[:origin0], H)
        stored = os.path.join(PAPER, 'data/predictions', ds, model, 'pred.npy')
        if os.path.exists(stored):
            S = np.load(stored)[0].astype(np.float64)          # (H, N)
            d = float(np.abs(p0 - S).max())
        else:
            d = float('nan')
        # --- (b) final window
        origin = L - H
        p = predict(method, V[:origin], H)
        od = os.path.join(OUT, ds, model); os.makedirs(od, exist_ok=True)
        np.save(os.path.join(od, 'pred.npy'), p[None].astype(np.float32))
        np.save(os.path.join(od, 'true.npy'), V[origin:origin + H][None].astype(np.float32))
        nz = float((p > 0).mean())
        report[ds][model] = {'reproduce_maxdiff': d, 'pred_nz_frac': nz,
                             'pred_max': float(p.max()), 'origin_final': origin}
        print(f'  {model:8s} reproduce-vs-stored max|diff|={d:.3g}   final-window pred: '
              f'nonzero {nz*100:.1f}%  max {p.max():.2f}')

json.dump(report, open(os.path.join(PAPER, '.scratch/unify/step2a_report.json'), 'w'), indent=1)
print('\n[done]')
