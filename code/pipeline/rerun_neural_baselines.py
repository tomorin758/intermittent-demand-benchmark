"""Step 2b: re-run TCN / LSTM / WaveNet-like on the FINAL window of each history table.

The checkpoints live in PatchTST/{tcn,lstm,wavenet}_checkpoints/lightning_logs/version_*/checkpoints/.
For each dataset the candidate checkpoints are those with the matching
(context_len, pred_len) = (6,3) parts / (14,7) fresh / (56,28) m5; when several
candidates exist the one that reproduces the stored original-window predictions
(origin = int(0.8*L)) is selected.

Writes data/final/<ds>/<Model>/{pred,true}.npy
"""
import os, sys, csv, json, glob
import numpy as np

PAPER = '${PAPER_ROOT}/Projects/paper'
REPO = '${PAPER_ROOT}/Projects/PatchTST'
sys.path.insert(0, REPO)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch                                            # noqa: E402
from tcn_train import TCNForecaster                     # noqa: E402
from lstm_train import LSTMForecaster                   # noqa: E402
from wavenet_train import WaveNetForecaster             # noqa: E402

FAMS = {'TCN': ('tcn_checkpoints', TCNForecaster),
        'LSTM': ('lstm_checkpoints', LSTMForecaster),
        'WaveNet-like': ('wavenet_checkpoints', WaveNetForecaster)}
HF = {'parts': ('parts.csv', 3, 6), 'fresh_retail': ('fresh_retail.csv', 7, 14), 'm5': ('m5.csv', 28, 56)}
OUT = os.path.join(PAPER, 'data/final')
# NOTE: this model family is ~60x FASTER on CPU than on the GPUs of this box
# (kernel-1 conv with 1 input channel + huge batch picks pathological CUDA kernels).
DEV = os.environ.get('DEV', 'cpu')
torch.set_num_threads(int(os.environ.get('NTHREADS', '16')))
BATCH = 4096


def load_hist(path):
    with open(path) as fh:
        rd = csv.reader(fh); next(rd); rows = list(rd)
    return np.array([[float(x) if x not in ('', 'nan') else 0.0 for x in r[1:]] for r in rows],
                    dtype=np.float32)


def predict(model, V, num_train, ctx, origin, H):
    """V: (L, C) raw -> (H, C) raw. Same per-series normalisation as the training scripts."""
    L, C = V.shape
    out = np.zeros((H, C), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, C, BATCH):
            blk = V[:, s:s + BATCH].astype(np.float64)
            mean = blk[:num_train].mean(axis=0)
            std = blk[:num_train].std(axis=0)
            std[std < 1e-6] = 1.0
            x = blk[origin - ctx:origin, :]
            xn = (x - mean) / std
            t = torch.tensor(xn.T[:, :, None].astype(np.float32), dtype=torch.float32, device=DEV)
            p = model(t).cpu().numpy().reshape(t.shape[0], -1)
            out[:, s:s + BATCH] = (p * std[:, None] + mean[:, None]).T
    return out


def candidates(fam_dir, ctx, H):
    out = []
    for f in sorted(glob.glob(os.path.join(REPO, fam_dir, 'lightning_logs', '**', '*.ckpt'), recursive=True)):
        ck = torch.load(f, map_location='cpu', weights_only=False)
        hp = ck.get('hyper_parameters') or {}
        if hp.get('context_len') == ctx and hp.get('pred_len') == H:
            out.append(f)
    return out


report = {}
for ds, (fname, H, ctx) in HF.items():
    V = load_hist(os.path.join(PAPER, 'data/history', fname))
    L, C = V.shape
    num_train = int(L * 0.7)
    origin0 = int(L * 0.8)
    origin = L - H
    report[ds] = {}
    print('=' * 92); print(f'{ds}: L={L} C={C} H={H} ctx={ctx} num_train={num_train} '
                          f'orig0={origin0} origin_final={origin}')
    for model_name, (fam_dir, cls) in FAMS.items():
        done = os.path.join(OUT, ds, model_name, 'pred.npy')
        if os.path.exists(done) and os.environ.get('SKIP_EXISTING', '1') == '1':
            print(f'  [skip] {ds}/{model_name}: already written')
            report[ds][model_name] = {'status': 'skipped-existing'}
            continue
        cands = candidates(fam_dir, ctx, H)
        stored_path = os.path.join(PAPER, 'data/predictions', ds, model_name, 'pred.npy')
        stored = np.load(stored_path)[0].astype(np.float64) if os.path.exists(stored_path) else None
        best = None
        for f in cands:
            try:
                m = cls.load_from_checkpoint(f, map_location='cpu').to(DEV).eval()
            except Exception as e:                                   # noqa: BLE001
                print(f'  [load-fail] {model_name} {os.path.basename(f)}: {type(e).__name__}: {e}')
                continue
            p0 = predict(m, V, num_train, ctx, origin0, H)
            d = float(np.abs(p0 - stored).max()) if stored is not None else float('nan')
            print(f'  {model_name:12s} cand {f.split("/")[-3]:10s} {os.path.basename(f):28s} '
                  f'reproduce max|diff|={d:.3g}')
            if best is None or d < best[0]:
                best = (d, f, m)
        if best is None or not (best[0] < 1e-3):
            print(f'  [FAIL] {model_name}: no checkpoint reproduced the stored predictions '
                  f'(best={None if best is None else best[0]})')
            report[ds][model_name] = {'status': 'FAIL'}
            continue
        d, f, m = best
        p = predict(m, V, num_train, ctx, origin, H)
        od = os.path.join(OUT, ds, model_name); os.makedirs(od, exist_ok=True)
        np.save(os.path.join(od, 'pred.npy'), p[None].astype(np.float32))
        np.save(os.path.join(od, 'true.npy'), V[origin:origin + H][None].astype(np.float32))
        report[ds][model_name] = {'status': 'ok', 'ckpt': f, 'reproduce_maxdiff': d,
                                  'pred_max': float(p.max()), 'pred_nz_frac': float((p > 0).mean())}
        print(f'  -> {model_name}: using {os.path.basename(f)} (max|diff|={d:.2e}); '
              f'final window pred: nonzero {(p > 0).mean()*100:.1f}%  max {p.max():.2f}')

json.dump(report, open(os.path.join(PAPER, '.scratch/unify/step2b_report.json'), 'w'), indent=1)
print('\n[done]')
