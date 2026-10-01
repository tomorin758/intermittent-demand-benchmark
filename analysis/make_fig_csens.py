"""Companion figure for the threshold-sensitivity subsection (Section 4.5).

Reads the pre-computed sweep `results/cscan.csv` (one row per dataset x model x threshold)
and draws, per dataset, the zero-detection F1 of every configuration against the truncation
threshold c.  The configurations that hold the best F1 at some threshold are coloured and
labelled; the rest stay grey.  The shaded band is the best-to-worst spread, which shrinks as
c grows -- the two readings the subsection reports in text.

Conventions match the other figures of the paper: Okabe-Ito palette, panel letters *below*
each panel (the descriptive titles live in the caption), no in-figure descriptions.
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
CSV = os.environ.get('CSCAN_CSV') or next(
    (p for p in (os.path.join(BUNDLE, 'results', 'cscan.csv'),
                 os.path.join(BUNDLE, '..', 'release', 'results', 'cscan.csv')) if os.path.exists(p)),
    os.path.join(BUNDLE, 'results', 'cscan.csv'))
OUT = os.environ.get('FIG_OUT') or (
    os.path.join(BUNDLE, 'results') if os.path.isdir(os.path.join(BUNDLE, 'results'))
    else os.path.join(HERE, 'bundle_results'))
os.makedirs(OUT, exist_ok=True)

PANELS = [('parts', 'Parts'), ('fresh_retail', 'Fresh retail'), ('m5', 'M5')]
COLOURS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']

rows = [r for r in csv.DictReader(open(CSV))]
CS = sorted({float(r['c']) for r in rows})
X = {c: i for i, c in enumerate(CS)}          # evenly spaced thresholds

fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.5))
plt.subplots_adjust(wspace=0.30, left=0.055, right=0.995, top=0.96, bottom=0.34)

for k, (key, label) in enumerate(PANELS):
    ax = axes[k]
    data = {}
    for r in rows:
        if r['dataset'] != key:
            continue
        data.setdefault(r['model'], {})[float(r['c'])] = float(r['f1'])

    # winners: any configuration that holds the best F1 at some threshold
    best_at = {c: max(data[m][c] for m in data) for c in CS}
    winners = [m for m in data if any(abs(data[m][c] - best_at[c]) < 1e-12 for c in CS)]

    lo = [min(data[m][c] for m in data) for c in CS]
    hi = [max(data[m][c] for m in data) for c in CS]
    ax.fill_between(range(len(CS)), lo, hi, color='0.85', zorder=0, linewidth=0)

    for m in sorted(data):
        if m in winners:
            continue
        ax.plot(range(len(CS)), [data[m][c] for c in CS], color='0.62', linewidth=0.8,
                zorder=1)
    for i, m in enumerate(sorted(winners)):
        col = COLOURS[i % len(COLOURS)]
        ys = [data[m][c] for c in CS]
        ax.plot(range(len(CS)), ys, color=col, linewidth=1.5, marker='o', markersize=2.4,
                zorder=3)
    # end labels, decluttered vertically so that near-tied winners stay readable
    ends = sorted(((data[m][CS[-1]], m, COLOURS[i % len(COLOURS)])
                   for i, m in enumerate(sorted(winners))), reverse=True)
    placed = []
    for v, m, col in ends:
        y = v
        while placed and y > placed[-1] - 0.058:
            y = placed[-1] - 0.058
        y = max(y, 0.02)
        placed.append(y)
        ax.annotate(m, (len(CS) - 1, y), xytext=(3, 0), textcoords='offset points',
                    fontsize=5.2, color=col, va='center', zorder=4)

    ax.set_xticks(range(len(CS)))
    ax.set_xticklabels([('%.2f' % c).rstrip('0').rstrip('.') for c in CS], fontsize=5.5)
    ax.set_xlabel('truncation threshold $c$', fontsize=6.5)
    if k == 0:
        ax.set_ylabel('zero-detection $F_1$', fontsize=6.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.25, len(CS) + 2.1)
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.grid(axis='y', linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.text(0.5, -0.40, '(%s)' % 'abc'[k], transform=ax.transAxes, ha='center', va='top',
            fontsize=7)

fig.savefig(os.path.join(OUT, 'fig_csens.pdf'))
fig.savefig(os.path.join(OUT, 'fig_csens.png'), dpi=300)
print('wrote', os.path.join(OUT, 'fig_csens.pdf'))
for key, label in PANELS:
    sub = [r for r in rows if r['dataset'] == key]
    best = {c: max(float(r['f1']) for r in sub if float(r['c']) == c) for c in CS}
    worst = {c: min(float(r['f1']) for r in sub if float(r['c']) == c) for c in CS}
    print(f'  {label:13s} spread {best[CS[0]]-worst[CS[0]]:.3f} -> {best[CS[-1]]-worst[CS[-1]]:.3f}')
