"""Lightweight audit: every 3-decimal number quoted in the results/discussion text must appear
among the metrics of the dataset that section is about (raw or truncated variant)."""
import json, os, re, sys

PAPER = '${PAPER_ROOT}/Projects/paper'
tex = open(os.path.join(PAPER, 'bench2026-paper/main.tex'), encoding='utf-8').read().split('\n')
M = {}
for ds in ['parts', 'fresh_retail', 'm5']:
    d = os.path.join(PAPER, 'data/metrics_final', ds)
    vals = set()
    for f in os.listdir(d):
        if f.endswith('.json'):
            for v in json.load(open(os.path.join(d, f))).values():
                if isinstance(v, float):
                    vals.add(round(v, 3)); vals.add(round(v, 2))
    M[ds] = vals

SEC = [('parts', 'Very sparse data: spare parts'), ('fresh_retail', 'Dense-but-intermittent retail'),
       ('m5', 'M5 (Walmart)')]
marks = []
for ds, title in SEC:
    i = next(k for k, l in enumerate(tex) if title in l)
    j = next(k for k in range(i + 1, len(tex)) if tex[k].startswith('\\subsection'))
    marks.append((ds, i, j))
marks.append(('ALL', next(k for k, l in enumerate(tex) if 'Cross-dataset findings' in l),
              next(k for k, l in enumerate(tex) if l.startswith('\\section{Discussion'))))
bad = 0
for ds, i, j in marks:
    scope = M[ds] if ds != 'ALL' else set().union(*M.values())
    for k in range(i, j):
        line = tex[k]
        if line.strip().startswith('%%'):
            continue
        for num in re.findall(r'\$?(\d\.\d\d\d)\$?', line):
            v = float(num)
            if v not in scope:
                print(f'  [{ds}] line {k+1}: {num} not found in metrics  -> {line.strip()[:90]}')
                bad += 1
print(f'\n{"audit clean" if bad == 0 else str(bad) + " suspicious number(s)"}')
