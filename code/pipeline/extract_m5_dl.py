"""Step 2c-extract: pull the FINAL window out of the re-run m5 test passes
(DLinear / HurdleDLinear / PatchTCN / PatchTST) and store it canonically.

All four ran with --target FOODS_3_090__CA_3 (history column index 8412), so the array
column order is "remove column 8412, append at the end"; the true array must reproduce
the final 28 rows of data/history/m5.csv exactly after undoing that move.
"""
import numpy as np, csv, os, json

PAPER = '${PAPER_ROOT}/Projects/paper'
RES = '${PAPER_ROOT}/Projects/PatchTST/PatchTST_supervised/results'
SET = {
    'DLinear': 'm5_112_28_DLinear_custom_ftM_sl112_ll28_pl28_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0',
    'HurdleDLinear': 'm5_112_28_hurdle_HurdleDLinear_custom_ftM_sl112_ll28_pl28_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0',
    'PatchTCN': 'm5_112_28_PatchTCN_custom_ftM_sl112_ll28_pl28_dm32_nh4_el3_dl1_df64_fc1_ebtimeF_dtTrue_Exp_0',
    'PatchTST': 'm5_112_28_PatchTST_custom_ftM_sl112_ll28_pl28_dm32_nh4_el3_dl1_df64_fc1_ebtimeF_dtTrue_Exp_0',
}
K = 8412
N = 30490

with open(os.path.join(PAPER, 'data/history/m5.csv')) as fh:
    rd = csv.reader(fh); next(rd); rows = list(rd)
V = np.array([[float(x) if x not in ('', 'nan') else 0.0 for x in r[1:]] for r in rows], dtype=np.float32)
L = V.shape[0]
H = 28
perm = list(range(K)) + list(range(K + 1, N)) + [K]


def to_hist(M):
    out = np.empty_like(M); out[:, perm] = M; return out


rep = {}
for m, s in SET.items():
    p = os.path.join(RES, s, 'pred.npy'); t = os.path.join(RES, s, 'true.npy')
    P = np.load(p, mmap_mode='r'); T = np.load(t, mmap_mode='r')
    print(f'{m:15s} shape={P.shape}')
    pw = to_hist(np.asarray(P[P.shape[0] - 1], dtype=np.float32))
    tw = to_hist(np.asarray(T[T.shape[0] - 1], dtype=np.float32))
    ok = np.array_equal(tw, V[L - H:L])
    print(f'   last window rows {L-H}..{L-1}: true matches history exactly: {ok}')
    if not ok:
        ncol = int(np.sum([np.array_equal(tw[:, j], V[L - H:L, j]) for j in range(N)]))
        print(f'   [FAIL] only {ncol}/{N} columns match -> not storing'); rep[m] = {'status': 'FAIL'}
        continue
    od = os.path.join(PAPER, 'data/final/m5', m); os.makedirs(od, exist_ok=True)
    np.save(os.path.join(od, 'pred.npy'), pw[None])
    np.save(os.path.join(od, 'true.npy'), V[L - H:L][None].astype(np.float32))
    rep[m] = {'status': 'ok', 'from': s, 'windows': int(P.shape[0]),
              'pred_max': float(pw.max()), 'pred_nz': float((pw > 0).mean())}
    print(f'   stored -> data/final/m5/{m}/  pred nonzero {(pw>0).mean()*100:.1f}% max {pw.max():.2f}')

json.dump(rep, open(os.path.join(PAPER, '.scratch/unify/step2c_report.json'), 'w'), indent=1)
print('\n[done]')
