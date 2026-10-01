"""Companion figure for the dataset-profile table (Table 1).

Panels
  (a) ADI-CV^2 of every series on the training split, with the four Syntetos-Boylan class
      boundaries (ADI = 1.32, CV^2 = 0.49) used in Section 3.3.
  (b) ECDF over series of the non-zero ratio of the full history, with the pooled non-zero
      ratio of the evaluation window marked.
  (c) Survival P(demand >= k) of non-zero demand sizes over the full history (log-log).
  (d) Class composition of the training split, including series with no defined class.

Conventions match gen_unified_metrics.classes(): classes are computed on the first 70 % of the
history table, ADI = len(s)/len(pos), CV^2 = (std(ddof=1)/mean)^2, and a series with fewer than
two positive observations is 'na'.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Inside the released bundle the figures belong in results/; in the working tree they go to
# .scratch/unify/bundle_results. FIG_OUT overrides either.
OUT = os.environ.get('FIG_OUT') or (
    os.path.join(BUNDLE, 'results') if os.path.isdir(os.path.join(BUNDLE, 'results'))
    else os.path.join(BUNDLE, 'unify', 'bundle_results'))
os.makedirs(OUT, exist_ok=True)
# The history tables live in data/history in the working tree and are shipped as data/<ds>.csv in
# the bundle; HIST_DIR overrides both.
HIST = os.environ.get('HIST_DIR') or next(
    (d for d in (os.path.join(BUNDLE, 'data', 'history'), os.path.join(BUNDLE, 'data'),
                 os.path.join(os.path.dirname(BUNDLE), 'data', 'history')) if os.path.isdir(d)),
    os.path.join(BUNDLE, 'data', 'history'))

DS = [('Parts', 'parts.csv', 3), ('Fresh retail', 'fresh_retail.csv', 7), ('M5', 'm5.csv', 28)]

_missing = [f for _, f, _ in DS if not os.path.exists(os.path.join(HIST, f))]
if _missing:
    raise SystemExit(
        'missing history table(s) in %s: %s\n'
        'The M5 history table is not redistributed (see DATA_LICENSE.md), so Figure 1 cannot be '
        'rebuilt from the bundle alone; the printed figure ships as results/fig_profiles.pdf.'
        % (HIST, ', '.join(_missing)))
STYLE = {  # Okabe-Ito palette; line style and marker duplicate the identity so the figure
           # stays readable when printed in black and white (venue requirement)
    'Parts':        dict(ls='-',  color='#0072B2', marker='o'),
    'Fresh retail': dict(ls='--', color='#D55E00', marker='s'),
    'M5':           dict(ls=':',  color='#009E73', marker='^'),
}
CLASSES = ['smooth', 'erratic', 'intermittent', 'lumpy']
HATCH = {'smooth': '///', 'erratic': '...', 'intermittent': '', 'lumpy': '\\\\\\', 'na': 'xx'}
FILL = {'smooth': '#56B4E9', 'erratic': '#E69F00', 'intermittent': '#009E73',
        'lumpy': '#D55E00', 'na': '#CCCCCC'}


def load(fname):
    import csv
    with open(os.path.join(HIST, fname), newline='', encoding='utf-8-sig') as fh:
        r = csv.reader(fh)
        header = next(r)
        cols = [[] for _ in header[1:]]
        for row in r:
            for j, v in enumerate(row[1:]):
                cols[j].append(float(v))
    return header[1:], np.array(cols, dtype=np.float64).T  # (T, N)


def adi_cv2(hist):
    """Vectorised copy of gen_unified_metrics.classes()/ADI-CV^2."""
    T, N = hist.shape
    pos = hist > 0
    cnt = pos.sum(axis=0)
    tot = np.where(cnt > 0, hist.sum(axis=0), 0.0)
    mean = np.divide(tot, cnt, out=np.full(N, np.nan), where=cnt > 0)
    sq = np.where(pos, hist ** 2, 0.0).sum(axis=0)
    var = np.divide(sq, cnt - 1, out=np.full(N, np.nan), where=cnt > 1)
    var = np.maximum(var - np.divide(tot ** 2, cnt * np.maximum(cnt - 1, 1),
                                     out=np.full(N, np.nan), where=cnt > 1), 0.0)
    adi = np.divide(T, cnt, out=np.full(N, np.nan), where=cnt > 0)
    cv2 = np.divide(var, mean ** 2, out=np.full(N, np.nan), where=(cnt > 1) & (mean > 0))
    cls = np.array(['na'] * N, dtype=object)
    ok = cnt >= 2
    cls[ok & (adi < 1.32) & (cv2 < 0.49)] = 'smooth'
    cls[ok & (adi < 1.32) & (cv2 >= 0.49)] = 'erratic'
    cls[ok & (adi >= 1.32) & (cv2 < 0.49)] = 'intermittent'
    cls[ok & (adi >= 1.32) & (cv2 >= 0.49)] = 'lumpy'
    return adi, cv2, cls


data = {}
for name, fname, H in DS:
    series, V = load(fname)
    T, N = V.shape
    train = V[:int(T * 0.7)]
    adi, cv2, cls = adi_cv2(train)
    hist_nz_ratio = (V > 0).mean(axis=0)
    eval_ratio = (V[-H:] > 0).mean()
    sizes = V[V > 0]
    data[name] = dict(T=T, N=N, adi=adi, cv2=cv2, cls=cls,
                      nz_ratio=hist_nz_ratio, eval_ratio=eval_ratio, sizes=sizes)
    # cross-check against the values printed in Table 1
    defined = (cls != 'na').sum()
    shares = {c: 100.0 * (cls == c).sum() / defined for c in CLASSES}
    print(f"{name:<13} T={T:<5} N={N:<6} eval non-zero {100*eval_ratio:5.1f}%  "
          f"classes(from train) " + "  ".join(f"{c[:4]} {shares[c]:4.0f}%" for c in CLASSES)
          + f"  na {100*(cls=='na').mean():4.1f}%")
    fin = np.isfinite(adi) & np.isfinite(cv2)
    print(f"{'':<13} median ADI {np.median(adi[fin]):.2f}  median CV^2 {np.median(cv2[fin]):.3f}")

fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05))
plt.subplots_adjust(wspace=0.42, left=0.055, right=0.995, top=0.97, bottom=0.26)

# ---------------------------------------------------------------- (a) ADI-CV^2
ax = axes[0]
adi_edges = np.logspace(np.log10(1.0), np.log10(40), 60)
cv2_edges = np.logspace(np.log10(0.005), np.log10(20), 60)
all_adi = np.concatenate([data[n]['adi'] for n, _, _ in DS])
all_cv2 = np.concatenate([data[n]['cv2'] for n, _, _ in DS])
m = np.isfinite(all_adi) & np.isfinite(all_cv2) & (all_cv2 > 0)
H2, xe, ye = np.histogram2d(all_adi[m], all_cv2[m], bins=[adi_edges, cv2_edges])
Z = np.log10(H2.T + 1.0)
ax.pcolormesh(xe, ye, Z, cmap='Blues', shading='auto', rasterized=True)
for name, _, _ in DS:
    d = data[name]
    mm = np.isfinite(d['adi']) & np.isfinite(d['cv2']) & (d['cv2'] > 0)
    mx, my = np.median(d['adi'][mm]), np.median(d['cv2'][mm])
    ax.plot(mx, my, marker=STYLE[name]['marker'], ms=4.4, color=STYLE[name]['color'],
            markerfacecolor=STYLE[name]['color'], markeredgecolor='black', markeredgewidth=0.5,
            ls='none', zorder=5)
    ax.annotate(name, (mx, my), textcoords='offset points', xytext=(4, -1.5), fontsize=5.2,
                color='black', zorder=6,
                bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))
ax.axvline(1.32, color='black', lw=0.7)
ax.axhline(0.49, color='black', lw=0.7)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(1, 20); ax.set_ylim(0.01, 8)
ax.set_xlabel('ADI (periods between demands)', fontsize=6.5)
ax.set_ylabel('CV$^2$ of demand sizes', fontsize=6.5)
ax.text(0.5, -0.22, '(a)', transform=ax.transAxes, ha='center', va='top', fontsize=7)
ax.tick_params(labelsize=5.5, length=2, pad=1)
for (x, y, lab, ha, va) in [(1.03, 6.5, 'erratic', 'left', 'top'),
                            (19.0, 0.012, 'intermittent', 'right', 'bottom')]:
    ax.text(x, y, lab, fontsize=5.5, color='0.15', ha=ha, va=va, zorder=6,
            bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85))

# ------------------------------------------------- (b) ECDF of per-series zero share
ax = axes[1]
for name, _, _ in DS:
    d = data[name]
    x = np.sort(d['nz_ratio'])
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where='post', color=STYLE[name]['color'], ls=STYLE[name]['ls'], lw=1.0)
    ax.plot([d['eval_ratio']], [0.5], marker=STYLE[name]['marker'], ms=3.0,
            color=STYLE[name]['color'], ls='none')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel('per-series non-zero share (history)', fontsize=6.5)
ax.set_ylabel('fraction of series', fontsize=6.5)
ax.text(0.5, -0.22, '(b)', transform=ax.transAxes, ha='center', va='top', fontsize=7)
ax.tick_params(labelsize=5.5, length=2, pad=1)

# ------------------------------------------------- (c) demand-size survival
ax = axes[2]
ks = np.arange(1, 201)
for name, _, _ in DS:
    s = data[name]['sizes']
    surv = np.array([(s >= k).mean() for k in ks])
    ax.plot(ks, surv, color=STYLE[name]['color'], ls=STYLE[name]['ls'], lw=1.0)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(1, 200); ax.set_ylim(1e-6, 1.05)
ax.set_xlabel('demand size $k$', fontsize=6.5)
ax.set_ylabel('P(demand $\\geq k$)', fontsize=6.5)
ax.text(0.5, -0.22, '(c)', transform=ax.transAxes, ha='center', va='top', fontsize=7)
ax.tick_params(labelsize=5.5, length=2, pad=1)

# ------------------------------------------------- (d) class composition
ax = axes[3]
ys = np.arange(len(DS))[::-1]
ABBR = {'smooth': 'smooth', 'erratic': 'errat.', 'intermittent': 'interm.',
        'lumpy': 'lumpy', 'na': 'no class'}


def largest_remainder(shares, total=100):
    """Integer percentages that sum to `total` (largest-remainder / Hamilton method).
    Table 1 in the paper uses exactly these values, so the panel and the table agree."""
    floors = {k: int(np.floor(v)) for k, v in shares.items()}
    seats = total - sum(floors.values())
    order = sorted(shares, key=lambda k: shares[k] - floors[k], reverse=True)
    for k in order[:seats]:
        floors[k] += 1
    return floors


defined = np.array([(data[name]['cls'] != 'na').sum() for name, _, _ in DS], dtype=float)
exact, rounded = [], []
for i, (name, _, _) in enumerate(DS):
    sh = {c: 100.0 * (data[name]['cls'] == c).sum() / defined[i] for c in CLASSES}
    exact.append(sh)
    rounded.append(largest_remainder(sh))

left = np.zeros(len(DS))
for c in CLASSES:
    vals = np.array([rounded[i][c] for i in range(len(DS))], dtype=float)
    if vals.sum() == 0:
        continue
    ax.barh(ys, vals, left=left, height=0.5, color=FILL[c], edgecolor='black', linewidth=0.5,
            hatch=HATCH[c])
    for y, v, l in zip(ys, vals, left):
        if v >= 9:
            edge = l + v >= 99.5
            ax.text(l + v / 2 if not edge else l + v - 1.5, y, f'{ABBR[c]} {int(v)}',
                    ha='center' if not edge else 'right', va='center', fontsize=5.0,
                    bbox=dict(fc='white', ec='none', pad=0.4, alpha=0.85))
    left += vals
for i, (name, _, _) in enumerate(DS):
    print(f"  bar check {name:<13} 精确 " + "  ".join(f"{c[:4]} {exact[i][c]:5.2f}" for c in CLASSES)
          + f" (合计 {sum(exact[i].values()):.2f}) → 取整 " + "  ".join(f"{c[:4]} {rounded[i][c]:>2}" for c in CLASSES)
          + f" (合计 {sum(rounded[i].values())})   no class {100.0*(data[name]['cls']=='na').mean():4.1f}%")
ax.set_yticks(ys); ax.set_yticklabels([n for n, _, _ in DS], fontsize=5.5)
ax.set_xlim(0, 100); ax.set_xlabel('share of series (%)', fontsize=6.5)
ax.text(0.5, -0.22, '(d)', transform=ax.transAxes, ha='center', va='top', fontsize=7)
ax.tick_params(labelsize=5.5, length=2, pad=1)

fig.savefig(os.path.join(OUT, 'fig_profiles.pdf'))
fig.savefig(os.path.join(OUT, 'fig_profiles.png'), dpi=300)
print('wrote', os.path.join(OUT, 'fig_profiles.pdf'))
