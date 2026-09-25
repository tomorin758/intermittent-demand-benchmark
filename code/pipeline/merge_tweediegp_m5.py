"""Merge the re-run TweedieGP M5 fit into the unified store.

The run used dataset/m5.csv with h=28 (train on days 0..1884, test 1885..1912) and kept all
30,490 series (no all-zero training series), so the forecast rows map 1:1 onto the history
columns; this is verified against test_actual.npy before anything is written.
"""

# ---------------------------------------------------------------------------
# PROTOCOL RECORD, NOT RUNNABLE FROM THIS BUNDLE.
# These scripts re-run models or re-extract their predictions, so they need the third-party
# repositories (PatchTST harness, TweedieGP, granite-tsfm, MOMENT/Timer/TimesFM) and the trained
# checkpoints, none of which are redistributed here. Set WORKSPACE to the directory that holds
# both the paper workspace and those repositories; the defaults below only document the layout
# used in the paper. What *is* reproducible from this bundle is described in README.md
# ("What is reproducible from this bundle").
# ---------------------------------------------------------------------------
import os as _os
WORKSPACE = _os.environ.get('WORKSPACE', _os.path.expanduser('~'))
PAPER = _os.environ.get('PAPER_DIR', _os.path.join(WORKSPACE, 'Projects', 'paper'))
PATCHTST = _os.environ.get('PATCHTST_DIR', _os.path.join(WORKSPACE, 'Projects', 'PatchTST'))
TWEEDIEGP = _os.environ.get('TWEEDIEGP_DIR', _os.path.join(WORKSPACE, 'Projects', 'TweedieGP'))
import numpy as np, pandas as pd, glob, os, json, sys

PAPER = PAPER
TW = TWEEDIEGP
runs = sorted(glob.glob(os.path.join(TW, 'trained_models', 'm5_unified__*')))
if not runs:
    sys.exit('no m5_unified__* run found under trained_models/')
run = runs[-1]
print('using', run)

q = np.load(os.path.join(run, 'forecasts', 'q0.5.npy'))          # (N, h)
ta = np.load(os.path.join(run, 'forecasts', 'test_actual.npy'))   # (N, h)
times = np.load(os.path.join(run, 'forecasts', 'times.npy'))
print(f'forecast {q.shape}  mean fit {np.nanmean(times):.2f} s/series  total {np.nansum(times)/3600:.1f} h')

df = pd.read_csv(os.path.join(TW, 'dataset', 'm5.csv'))
V = df.iloc[:, 1:].to_numpy(dtype=np.float32)
L = V.shape[0]; H = 28
assert ta.shape[1] == H and ta.shape[0] == V.shape[1], (ta.shape, V.shape)
ok = np.array_equal(ta, V[L - H:L].T)
print('test_actual == history rows %d..%d (transposed): %s' % (L - H, L - 1, ok))
if not ok:
    sys.exit('period mismatch -- refusing to write')

od = os.path.join(PAPER, 'data/final/m5/TweedieGP'); os.makedirs(od, exist_ok=True)
np.save(os.path.join(od, 'pred.npy'), q.T[None].astype(np.float32))
np.save(os.path.join(od, 'true.npy'), V[L - H:L][None].astype(np.float32))
ql = 0.5 * float(np.abs(q - ta).mean())
print(f'wrote data/final/m5/TweedieGP  pooled QL(0.5) = {ql:.4f}  '
      f'exact-zero frac = {(q == 0).mean()*100:.1f}%')
json.dump({'run': run, 'ql_raw': ql, 'exact_zero_frac': float((q == 0).mean()),
           'fit_hours': float(np.nansum(times) / 3600)},
          open(os.path.join(PAPER, '.scratch/unify/tweediegp_m5_merge.json'), 'w'), indent=1)
