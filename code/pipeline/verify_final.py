"""Verify every array in data/final/: shape, ground truth identity, sane predictions.

Run after every writer has stopped:  python verify_final.py
Exit code 1 if anything is off.
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
import numpy as np, csv, os, sys

PAPER = PAPER
DS = {'parts': ('parts.csv', 3), 'fresh_retail': ('fresh_retail.csv', 7), 'm5': ('m5.csv', 28)}
bad = 0
for ds, (hf, H) in DS.items():
    with open(os.path.join(PAPER, 'data/history', hf)) as fh:
        rd = csv.reader(fh); next(rd); rows = list(rd)
    V = np.array([[float(x) if x not in ('', 'nan') else 0.0 for x in r[1:]] for r in rows], dtype=np.float64)
    L, N = V.shape
    d = os.path.join(PAPER, 'data/final', ds)
    models = sorted(os.listdir(d)) if os.path.isdir(d) else []
    print(f'== {ds}: {len(models)} models, expected final window rows {L-H}..{L-1} of {N} series')
    for m in models:
        p = os.path.join(d, m, 'pred.npy'); t = os.path.join(d, m, 'true.npy')
        if not (os.path.exists(p) and os.path.exists(t)):
            print(f'   [MISS] {m}: missing pred/true'); bad += 1; continue
        P = np.load(p); T = np.load(t)
        msgs = []
        if P.shape != (1, H, N):
            msgs.append(f'pred shape {P.shape}')
        if T.shape != (1, H, N):
            msgs.append(f'true shape {T.shape}')
        if T.shape == (1, H, N) and not np.array_equal(T[0], V[L - H:L]):
            msgs.append('true != history final rows')
        if not np.isfinite(P).all():
            msgs.append(f'non-finite pred ({(~np.isfinite(P)).sum()})')
        # the unconstrained baselines (TCN/LSTM/WaveNet-like) legitimately emit small
        # negative point forecasts; only flag absurd excursions
        if P.min() < -20:
            msgs.append(f'pred min {P.min():.3f} (implausible)')
        if msgs:
            print(f'   [FAIL] {m}: ' + '; '.join(msgs)); bad += 1
        else:
            neg = ' (has negatives)' if P.min() < -0.5 else ''
            print(f'   ok    {m:16s} pred [{P.min():6.2f},{P.max():7.2f}] '
                  f'exactzeros {100*(P==0).mean():5.1f}%{neg}')
print(f'\n{"ALL OK" if bad == 0 else str(bad) + " PROBLEM(S)"}')
sys.exit(1 if bad else 0)
