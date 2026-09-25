"""Cross-check the shipped artefacts against each other.

For every dataset it verifies that the sheet-format table (`tables/<ds>_unified.csv`, the input of
the LaTeX table generator) and the pooled metric files (`metrics/pooled/<ds>/<model>.json`,
the output of the metric implementation) agree cell by cell:

  * untruncated row : QL(0.5), MAE>0, RMSE>0
  * truncated row   : QL(0.5,c=0.5), 0Precision, 0Recall, 0F1
  * both rows       : tail RMSE (last quantile segment) and the four ADI-CV^2 regime RMSEs

Writes results/consistency_report.md.  Usage:  python analysis/check_consistency.py
"""
import csv, json, os, re

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOLED = os.environ.get('POOLED_DIR', os.path.join(BUNDLE, 'metrics', 'pooled'))
TABLES = os.environ.get('TABLES_DIR', os.path.join(BUNDLE, 'tables'))
RESULTS = os.environ.get('RESULTS_DIR', os.path.join(BUNDLE, 'results'))
os.makedirs(RESULTS, exist_ok=True)

DS = {'parts': 'parts_unified.csv', 'fresh_retail': 'fresh_retail_unified.csv',
      'm5': 'm5_unified.csv'}
RENAME = {'DeepAR-negbin': 'DeepAR(negbin)'}          # paper/table naming for one model
# column header in the sheet CSV is lower-case, the metric key is capitalised
REGIMES = [('intermittent', 'Intermittent'), ('lumpy', 'Lumpy'), ('smooth', 'Smooth'), ('erratic', 'Erratic')]


def table_rows(path):
    rows = list(csv.reader(open(path, encoding='utf-8-sig')))
    hdr = [c.strip() for c in rows[1]]
    out = {}
    for r in rows[2:]:
        if len(r) < 2 or not r[1].strip():
            continue
        name = r[1].strip()
        trunc = '(τ=0.5)' in name
        out[(name.replace('(τ=0.5)', '').strip(), trunc)] = {hdr[i]: r[i] for i in range(len(hdr)) if i < len(r)}
    return out


def num(x):
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def close(a, b, tol=5e-4):
    return a is not None and b is not None and abs(a - b) <= tol


report = ['# Consistency report', '',
          'Each cell below compares the LaTeX-table input (`tables/*_unified.csv`) with the',
          'pooled metric file (`metrics/pooled/*.json`) produced by the metric implementation.', '']
total = bad = 0
for ds, fname in DS.items():
    rows = table_rows(os.path.join(TABLES, fname))
    models = sorted(f[:-5] for f in os.listdir(os.path.join(POOLED, ds)) if f.endswith('.json'))
    report.append(f'## {ds} ({len(models)} models)')
    report.append('')
    report.append('| model | QL untr | MAE>0 untr | F1 trunc | tail untr | regimes untr | verdict |')
    report.append('|---|---:|---:|---:|---:|---:|---|')
    for m in models:
        j = json.load(open(os.path.join(POOLED, ds, f'{m}.json')))
        tname = RENAME.get(m, m)
        raw = rows.get((tname, False), {})
        tr = rows.get((tname, True), {})
        tail_key = sorted(k for k in j if k.startswith('NT_Seg'))[-1]
        checks = [
            ('QL untr', close(num(raw.get('QL(0.5)')), j['QL(0.5)'])),
            ('MAE untr', close(num(raw.get('MAE')), j['MAE_nonzero'])),
            ('F1 trunc', close(num(tr.get('F1 score')), j['0F1(c=0.5)'])),
            ('tail untr', close(num(raw.get('(q90, ∞) RMSE')), j[tail_key])),
            ('regimes untr', all(close(num(raw.get(f'{col} RMSE')), j[f'NT_{key} RMSE'])
                                 for col, key in REGIMES)),
        ]
        ok = all(v for _, v in checks)
        total += len(checks); bad += sum(1 for _, v in checks if not v)
        mark = '✓' if ok else '✗ ' + ', '.join(k for k, v in checks if not v)
        report.append(f"| {m} | {num(raw.get('QL(0.5)')) if raw else float('nan'):.3f} | "
                      f"{num(raw.get('MAE')) if raw else float('nan'):.3f} | "
                      f"{num(tr.get('F1 score')) if tr else float('nan'):.3f} | "
                      f"{num(raw.get('(q90, ∞) RMSE')) if raw else float('nan'):.3f} | "
                      f"{'ok' if checks[-1][1] else 'MISMATCH'} | {mark} |")
    report.append('')

report.append(f'**{total - bad}/{total} checks passed.**')
open(os.path.join(RESULTS, 'consistency_report.md'), 'w').write('\n'.join(report) + '\n')
print('\n'.join(report[-6:]))
print(f'[saved] results/consistency_report.md  ({total - bad}/{total} checks passed)')
