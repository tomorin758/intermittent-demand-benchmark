"""Emit Sheet-format CSVs (same layout as the current "*- Sheet1.csv" files) from the
unified metric JSONs, so that gen_tables.py / make_fig_reorder.py can consume them.

  data/parts_unified.csv, data/fresh_retail_unified.csv, data/m5_unified.csv

Row order follows the existing Sheet files (model order preserved); values come from
data/metrics_final/<ds>/<model>.json (metric_comprehensive output on the unified window).
"""
import csv, json, os, re, sys

PAPER = '${PAPER_ROOT}/Projects/paper'
DS = {'parts': ('parts.csv', 'Parts', 'Parts - Sheet1.csv'),
      'fresh_retail': ('fresh_retail.csv', 'Fresh', 'Fresh Retail - Sheet1.csv'),
      'm5': ('m5.csv', 'M5', 'M5 - Sheet1.csv')}
BLANK = ['', '']
HEAD0 = ['Dataset', 'Model', 'Overall Metrics', 'Zero Metrics', '', '', 'Nonzero Metrics', ''] + [''] * 10
HEAD1 = ['', '', 'QL(0.5)', 'Precision', 'Recall', 'F1 score', 'MAE', 'RMSE',
         '[0, q10] RMSE', '(q10, q25] RMSE', '(q25, q50] RMSE', '(q50, q75] RMSE',
         '(q75, q90] RMSE', '(q90, ∞) RMSE', 'intermittent RMSE', 'lumpy RMSE',
         'smooth RMSE', 'erratic RMSE', '', '']


def seg_keys(met, prefix):
    ks = [k for k in met if k.startswith(prefix + '_Seg')]
    ks.sort(key=lambda k: int(re.search(r'Seg(\d+)', k).group(1)))
    return ks


def fmt(v):
    if v is None:
        return '-'
    try:
        f = float(v)
    except (TypeError, ValueError):
        return '-'
    if f != f:                     # NaN
        return '-'
    return f'{f:.3f}'


def row_for(ds_label, model, variant, met):
    name = RENAME.get(model, model)
    r = ['', name if variant == 'raw' else f'{name}(τ=0.5)']
    if variant == 'raw':
        vals = [met['QL(0.5)'], None, None, None, met['MAE_nonzero'], met['RMSE_nonzero']]
        pfx = 'NT'
    else:
        vals = [met['QL(0.5,c=0.5)'], met['0Precision(c=0.5)'], met['0Recall(c=0.5)'],
                met['0F1(c=0.5)'], met['MAE_nonzero(c=0.5)'], met['RMSE_nonzero(c=0.5)']]
        pfx = 'T'
    vals += [met.get(k) for k in seg_keys(met, pfx)]
    vals += [met.get(f'{pfx}_{n} RMSE') for n in ['Intermittent', 'Lumpy', 'Smooth', 'Erratic']]
    r += [fmt(v) for v in vals] + BLANK
    return r


# every model is reported in BOTH settings: the untruncated row carries the headline point
# forecast (ranking convention), the truncated row re-scores the same forecast at c=0.5 and
# carries the zero-detection metrics.
RENAME = {'DeepAR-negbin': 'DeepAR(negbin)'}
TRUNC_ONLY = {'Croston', 'SBA', 'TSB'}

_only = [a for a in sys.argv[1:] if a in DS]
for ds, (hf, label, sheet) in DS.items():
    if _only and ds not in _only:
        continue
    mdir = os.path.join(PAPER, 'data/metrics_final', ds)
    if not os.path.isdir(mdir):
        print(f'[skip] {ds}: no metrics_final yet'); continue
    have = {f[:-5] for f in os.listdir(mdir) if f.endswith('.json')}
    # model order from the existing sheet
    order = []
    with open(os.path.join(PAPER, sheet), encoding='utf-8-sig') as fh:
        for r in csv.reader(fh):
            if len(r) > 1 and r[1].strip():
                name = r[1].replace('(τ=0.5)', '').strip()
                if name and name not in order and name in have:
                    order.append(name)
    order += sorted(m for m in have if m not in order)
    out = os.path.join(PAPER, f'data/{ds}_unified.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(HEAD0); w.writerow(HEAD1)
        first = True
        for m in order:
            met = json.load(open(os.path.join(mdir, f'{m}.json')))
            for variant in ('raw', 'trunc'):
                r = row_for(label, m, variant, met)
                r[0] = f'{label}\n' if first else ''
                w.writerow(r); first = False
    print(f'{out}: {len(order)} models, {sum(1 for _ in open(out))-2} rows')
print('[done]')
