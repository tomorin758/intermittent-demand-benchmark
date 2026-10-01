import numpy as np, os, json, sys, csv
ROOT=__import__('os').environ.get('PRED_DIR','data/final')   # prediction arrays are NOT shipped; see results/cscan.csv
DS=[a for a in sys.argv[1:] if a] or ['parts','fresh_retail','m5']
CS=[0.1,0.25,0.5,0.75,1.0,1.5,2.0]
_missing=[d for d in DS if not os.path.isdir(os.path.join(ROOT,d))]
if _missing:
    raise SystemExit(
        'missing prediction arrays in %s: %s\n'
        'This bundle ships the evaluation outputs, not the per-model prediction arrays, so the '
        'threshold sweep cannot be re-run from the bundle alone; its output ships pre-computed as '
        'results/cscan.csv (see the README, "Scope of this bundle"). Point PRED_DIR at a directory '
        'of <dataset>/<model>/{pred,true}.npy to re-run the sweep.'
        % (ROOT, ', '.join(_missing)))
def metrics(p,t,c):
    pt=np.where(p<c,0.0,p); tt=np.where(t<c,0.0,t)
    ql=0.5*float(np.mean(np.abs(pt-tt)))
    zt=t<c; zp=p<c
    tp=int(np.sum(zt&zp)); fp=int(np.sum(~zt&zp)); fn=int(np.sum(zt&~zp))
    pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0
    f1=2*pr*rc/(pr+rc) if pr+rc else 0.0
    return ql,pr,rc,f1
def avgrank(v):
    n=len(v); idx=sorted(range(n),key=lambda i:v[i]); r=[0.0]*n; i=0
    while i<n:
        j=i
        while j+1<n and v[idx[j+1]]==v[idx[i]]: j+=1
        for k in range(i,j+1): r[idx[k]]=(i+j)/2.0+1
        i=j+1
    return r
def pear(a,b):
    n=len(a); ma=sum(a)/n; mb=sum(b)/n
    da=(sum((x-ma)**2 for x in a))**.5; db=(sum((y-mb)**2 for y in b))**.5
    return 0.0 if da*db==0 else sum((x-ma)*(y-mb) for x,y in zip(a,b))/(da*db)
OUT={}
for ds in DS:
    d=os.path.join(ROOT,ds)
    models=sorted(m for m in os.listdir(d) if os.path.isdir(os.path.join(d,m)))
    res={}
    for m in models:
        pp=os.path.join(d,m,'pred.npy'); tp=os.path.join(d,m,'true.npy')
        if not (os.path.exists(pp) and os.path.exists(tp)):
            print(f"  [skip] {ds}/{m} missing files"); continue
        p=np.load(pp).ravel().astype(np.float64); t=np.load(tp).ravel().astype(np.float64)
        res[m]={c:metrics(p,t,c) for c in CS}
        del p,t
    OUT[ds]=res
    n=len(next(iter(res.values()))) if res else 0
    print(f"\n{'='*100}\n{ds}   models={len(res)}   test points={n}\n{'='*100}")
    print(f"{'model':18s}" + "".join(f"{'F1@'+str(c):>9}" for c in CS) + f"{'span':>8}{'@0.5 rank':>11}")
    order=sorted(res, key=lambda m: -res[m][0.5][3])
    for i,m in enumerate(order):
        f=[res[m][c][3] for c in CS]
        print(f"{m:18s}" + "".join(f"{v:9.3f}" for v in f) + f"{max(f)-min(f):8.3f}{i+1:9d}")
    print("\n  best zero detector at each c:")
    for c in CS:
        b=max(res,key=lambda m:res[m][c][3])
        spread=max(res[m][c][3] for m in res)-min(res[m][c][3] for m in res)
        print(f"    c={c:<5} {b:18s} F1={res[b][c][3]:.3f}   field spread={spread:.3f}")
    print("\n  rank correlation with the c=0.5 F1 ordering:")
    ms=list(res); base=avgrank([res[m][0.5][3] for m in ms])
    for c in CS:
        if c==0.5: continue
        rc=avgrank([res[m][c][3] for m in ms])
        mv=max(abs(base[i]-rc[i]) for i in range(len(ms)))
        print(f"    c={c:<5} rho={pear(base,rc):+.3f}  max rank move={mv:.0f}")
_resdir = os.environ.get('RESULTS_DIR', 'results')
os.makedirs(_resdir, exist_ok=True)
json.dump({ds:{m:{str(c):v for c,v in r.items()} for m,r in OUT[ds].items()} for ds in OUT},
          open(os.path.join(_resdir,'cscan_all.json'),'w'))
print(f"\n[saved] {_resdir}/cscan_all.json")

# ---------------------------------------------------------------------------
# Machine-readable dump of the sweep. The prediction arrays are not part of the
# artefact bundle, so this CSV (results/cscan.csv) is what makes the
# threshold-sensitivity claims in the paper checkable without re-running models.
# ---------------------------------------------------------------------------
_resdir = os.environ.get('RESULTS_DIR', 'results')
os.makedirs(_resdir, exist_ok=True)
with open(os.path.join(_resdir, 'cscan.csv'), 'w', newline='') as _fh:
    _w = csv.writer(_fh)
    _w.writerow(['dataset', 'model', 'c', 'ql_trunc', 'precision0', 'recall0', 'f1'])
    for _ds, _models in OUT.items():
        for _m, _per_c in _models.items():
            for _c, (_ql, _pr, _rc, _f1) in sorted(_per_c.items()):
                _w.writerow([_ds, _m, _c, f'{_ql:.6g}', f'{_pr:.6g}', f'{_rc:.6g}', f'{_f1:.6g}'])
print(f'[saved] {_resdir}/cscan.csv')
