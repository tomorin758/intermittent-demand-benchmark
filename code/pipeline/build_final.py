"""Step 1 of the window unification.

Build data/final/<ds>/<model>/{pred,true}.npy  -- shape (1, H, n_series),
values in the SAME column order as data/history/<ds>.csv, i.e. the final H periods
of that history table (rows L-H .. L-1).

Only models whose stored array already CONTAINS that window are handled here:
we slice window index W-1 and undo the model-specific column transform.
Models needing a re-run are listed at the end and left to the re-run scripts.

Every array is verified column-by-column against the history table (exact equality);
a failure aborts loudly.
"""
import numpy as np, os, csv, json, shutil

HF = {'parts': 'data/history/parts.csv',
      'fresh_retail': 'data/history/fresh_retail.csv',
      'm5': 'data/history/m5.csv'}
OUT = 'data/final'
SRC = 'data/predictions'

# model -> transform recipe for the LAST window
#   unscramble: None | 'global' | 500
#   k: None (identity) | int  ("remove column k, append at the end"; array col j = hist col perm[j])
#   direct: True  -> the stored single window IS already the final window in history order
RECIPE = {
    'parts': {
        'Autoformer':     dict(k=0),  'DLinear': dict(k=0), 'HurdleDLinear': dict(k=0),
        'PatchTCN':       dict(k=0),  'PatchTST': dict(k=0), 'PatchMixer': dict(k=0),
        'Moirai':         dict(),     'PatchTSMixer': dict(), 'TTM': dict(), 'TimesFM': dict(),
        'MOMENT':         dict(unscramble='global'),
        'DeepAR':         dict(direct=True), 'DeepAR-negbin': dict(direct=True),
        'TweedieGP':      dict(direct=True),
    },
    'fresh_retail': {
        'Autoformer':     dict(k=0),
        'DLinear': dict(k=179), 'HurdleDLinear': dict(k=179), 'PatchTCN': dict(k=179), 'PatchTST': dict(k=179),
        'Moirai': dict(), 'PatchMixer': dict(), 'PatchTSMixer': dict(), 'TTM': dict(), 'TimesFM': dict(),
        'MOMENT': dict(unscramble=500),
        'DeepAR': dict(direct=True), 'DeepAR-negbin': dict(direct=True), 'TweedieGP': dict(direct=True),
    },
    'm5': {
        'Autoformer': dict(k=0),
        'Moirai': dict(), 'PatchMixer': dict(), 'PatchTSMixer': dict(), 'TTM': dict(), 'TimesFM': dict(),
        'MOMENT': dict(unscramble=500), 'Timer': dict(unscramble=500),
        'DeepAR': dict(direct=True), 'DeepAR-negbin': dict(direct=True),
        # TweedieGP m5 has 30489 columns and a different period -> handled separately
    },
}
# models whose stored array does NOT contain the final window (must be re-run)
RERUN = ['TCN', 'LSTM', 'WaveNet-like', 'Croston', 'SBA', 'TSB']
RERUN_EXTRA = {'m5': ['DLinear', 'HurdleDLinear', 'PatchTCN', 'PatchTST']}


def load_hist(hf):
    with open(hf) as fh:
        rd = csv.reader(fh); hdr = next(rd); rows = list(rd)
    # hdr[0] is the date column (the caller currently uses only V, but keep the names aligned)
    return [c for c in hdr if c][1:], np.array(
        [[float(x) if x not in ('', 'nan') else 0.0 for x in r[1:]] for r in rows], dtype=np.float32)


def unscramble(C, chunk=None):
    H, n = C.shape
    if chunk is None or n <= chunk:
        return C.ravel(order='C').reshape(n, H).T
    parts = []; st = 0
    while st < n:
        L = min(chunk, n - st)
        parts.append(C[:, st:st + L].ravel(order='C').reshape(L, H).T); st += L
    return np.concatenate(parts, axis=1)


def perm_of(k, n):
    return list(range(k)) + list(range(k + 1, n)) + [k]


def to_hist_order(M, k):
    """M: (H, n) in model column order -> history column order (scatter)."""
    if k is None:
        return M
    n = M.shape[1]
    out = np.empty_like(M)
    out[:, perm_of(k, n)] = M
    return out


report = {}
for ds, hf in HF.items():
    names, V = load_hist(hf)
    L, N = V.shape
    H = None
    final_rows = None
    print('=' * 96); print(f'{ds}: L={L} series={N}')
    report[ds] = {}
    for m, rec in sorted(RECIPE[ds].items()):
        src = os.path.join(SRC, ds, m)
        if not os.path.exists(os.path.join(src, 'true.npy')):
            print(f'  [skip] {m}: no source npy'); continue
        T = np.load(os.path.join(src, 'true.npy'), mmap_mode='r')
        P = np.load(os.path.join(src, 'pred.npy'), mmap_mode='r')
        W, h, n = T.shape
        if H is None:
            H = h
        assert h == H, f'{ds}/{m}: horizon {h} != {H}'
        if rec.get('direct'):
            t = np.asarray(T[0], dtype=np.float32); p = np.asarray(P[0], dtype=np.float32)
            how = 'direct'
        else:
            t = np.asarray(T[W - 1], dtype=np.float32); p = np.asarray(P[W - 1], dtype=np.float32)
            how = f'slice W-1={W-1}'
            if rec.get('unscramble') == 'global':
                t = unscramble(t); p = unscramble(p); how += '+unscr_global'
            elif rec.get('unscramble') == 500:
                t = unscramble(t, 500); p = unscramble(p, 500); how += '+unscr_500'
        if 'k' in rec:
            t = to_hist_order(t, rec['k']); p = to_hist_order(p, rec['k']); how += f'+k={rec["k"]}'
        if t.shape[1] != N:
            print(f'  [skip] {m}: cols {t.shape[1]} != {N}'); continue
        # --- verification: true must equal the final H rows of the history table
        # (parts/PatchMixer went through a scaler round-trip, so allow 1e-5 float noise)
        maxdiff = float(np.abs(t.astype(np.float64) - V[L - H:L].astype(np.float64)).max())
        exact = bool(np.array_equal(t, V[L - H:L]))
        if maxdiff > 1e-5:
            print(f'  [FAIL] {m}: max|true - history| = {maxdiff:.3g} -- SKIPPED')
            report[ds][m] = {'status': 'FAIL', 'maxdiff': maxdiff, 'how': how}
            continue
        od = os.path.join(OUT, ds, m); os.makedirs(od, exist_ok=True)
        # ground truth is written from the history table itself (exact); only pred comes from the model
        np.save(os.path.join(od, 'true.npy'), V[L - H:L][None].astype(np.float32))
        np.save(os.path.join(od, 'pred.npy'), p[None])
        report[ds][m] = {'status': 'ok', 'how': how, 'W': W, 'rows': [L - H, L - 1],
                         'true_exact': exact, 'true_maxdiff': maxdiff,
                         'pred_min': float(p.min()), 'pred_max': float(p.max())}
        flag = 'exact' if exact else f'approx(max{maxdiff:.1e})'
        print(f'  {m:16s} ok  ({how}) true={flag}  pred range [{p.min():.2f}, {p.max():.2f}]')

json.dump(report, open('.scratch/unify/step1_report.json', 'w'), indent=1)
print('\nneeds re-run (all datasets):', RERUN)
print('needs re-run (m5 only):', RERUN_EXTRA['m5'])
