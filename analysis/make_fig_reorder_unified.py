import csv, math, itertools, json
import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt

import os as _os
_BUNDLE=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_TABLES=_os.environ.get('TABLES_DIR', _os.path.join(_BUNDLE,'tables'))
_RESULTS=_os.environ.get('RESULTS_DIR', _os.path.join(_BUNDLE,'results'))
POOLED=_os.environ.get('POOLED_DIR', _os.path.join(_BUNDLE,'metrics','pooled'))
_os.makedirs(_RESULTS, exist_ok=True)
FILES={'Parts':_os.path.join(_TABLES,'parts_unified.csv'),
       'Fresh':_os.path.join(_TABLES,'fresh_retail_unified.csv'),
       'M5':_os.path.join(_TABLES,'m5_unified.csv')}
AXES=['(q90, ∞) RMSE','intermittent RMSE','lumpy RMSE','smooth RMSE']
LABEL={'(q90, ∞) RMSE':'Tail\n$(q_{90},\\infty)$','intermittent RMSE':'Intermittent',
       'lumpy RMSE':'Lumpy','smooth RMSE':'Smooth'}

# ---- full-precision inputs -------------------------------------------------
# tables/*_unified.csv is rounded to three decimals, which creates spurious ties (two models whose
# non-zero MAE rounds to the same value) and average ranks then shift by half a position -- enough
# to flip a "moved five or more places" count and to pull Spearman correlations towards zero.
# metrics/pooled/ carries the unrounded values, so it is preferred when present; the CSVs in
# tables/ remain the fallback and keep this script runnable from the tables alone.
POOLED = _os.environ.get('POOLED_DIR', _os.path.join(_BUNDLE, 'metrics', 'pooled'))
_SEG_ORDER = ['[0, q10] RMSE', '(q10, q25] RMSE', '(q25, q50] RMSE',
              '(q50, q75] RMSE', '(q75, q90] RMSE', '(q90, \u221e) RMSE']
_AXES_JSON = ['QL(0.5)', 'Precision', 'Recall', 'F1 score', 'MAE', 'RMSE',
              'intermittent RMSE', 'lumpy RMSE', 'smooth RMSE', 'erratic RMSE']
_KEY = {'QL(0.5)': 'QL(0.5)', 'Precision': '0Precision(c=0.5)', 'Recall': '0Recall(c=0.5)',
        'F1 score': '0F1(c=0.5)', 'MAE': 'MAE_nonzero', 'RMSE': 'RMSE_nonzero',
        'intermittent RMSE': 'NT_Intermittent RMSE', 'lumpy RMSE': 'NT_Lumpy RMSE',
        'smooth RMSE': 'NT_Smooth RMSE', 'erratic RMSE': 'NT_Erratic RMSE'}


def load_pooled(dsdir):
    """Full-precision rows from metrics/pooled/<dataset>/*.json (untruncated point metrics and
    the c=0.5 zero-detection metrics, i.e. exactly the convention the paper reports)."""
    files = {}
    for f in sorted(_os.listdir(dsdir)):
        if f.endswith('.json'):
            with open(_os.path.join(dsdir, f)) as fh:
                files[f[:-5]] = json.load(fh)
    if not files:
        return {}
    segs = sorted({k for d in files.values() for k in d if k.startswith('NT_Seg')},
                  key=lambda s: int(s[6:].split('_')[0]))
    key = dict(_KEY)
    for i, c in enumerate(_SEG_ORDER):
        key[c] = segs[i] if i < len(segs) else None
    D = {}
    for m, d in files.items():
        row = {}
        for c in _AXES_JSON + _SEG_ORDER:
            k = key.get(c)
            v = d.get(k) if k else None
            row[c] = float('nan') if v is None else float(v)
        D[m] = row
    return D


def dataset_of(path):
    base = _os.path.basename(path)
    return 'fresh_retail' if 'fresh' in base else ('m5' if base.startswith('m5') else 'parts')


def load_any(path):
    pd = _os.path.join(POOLED, dataset_of(path))
    return load_pooled(pd) if _os.path.isdir(pd) else None

def num(s):
    try: return float(s.strip())
    except ValueError: return float('nan')
def avgrank(vals):
    n=len(vals); order=sorted(range(n), key=lambda i: vals[i]); r=[0.0]*n; i=0
    while i<n:
        j=i
        while j+1<n and vals[order[j+1]]==vals[order[i]]: j+=1
        for k in range(i,j+1): r[order[k]]=(i+j)/2.0+1
        i=j+1
    return r
def pear(a,b):
    p=[(x,y) for x,y in zip(a,b) if not(math.isnan(x) or math.isnan(y))]
    n=len(p); ma=sum(x for x,_ in p)/n; mb=sum(y for _,y in p)/n
    da=(sum((x-ma)**2 for x,_ in p))**.5; db=(sum((y-mb)**2 for _,y in p))**.5
    return 0.0 if da*db==0 else sum((x-ma)*(y-mb) for x,y in p)/(da*db)

DATA={}
for ds,f in FILES.items():
    D=load_any(f)                                    # full precision when metrics/pooled/ exists
    if D is None:
        rows=list(csv.reader(open(f,encoding='utf-8-sig')))
        hdr=[c.strip() for c in rows[1]]
        idx={c:hdr.index(c) for c in ['MAE']+AXES}
        D={}
        for r in rows[2:]:
            if len(r)<2 or not r[1].strip() or '(\u03c4=0.5)' in r[1]: continue   # untruncated = reference
            D[r[1].strip()]={c:num(r[idx[c]]) for c in idx}
    DATA[ds]=D

fig,axes=plt.subplots(1,2,figsize=(7.4,2.9))
# ---- panel (a): Spearman rho vs non-zero MAE ----
ax=axes[0]
hatches=['','///','...','xxx']
# Okabe-Ito palette, same mapping as fig_profiles (Parts blue, Fresh vermillion, M5 green);
# the hatches below keep the three bars apart in black-and-white printing
colors=['#0072B2','#D55E00','#009E73']
w=0.2; xs=range(len(AXES))
for k,ds in enumerate(['Parts','Fresh','M5']):
    D=DATA[ds]; ms=list(D)
    r_mae=avgrank([D[m]['MAE'] for m in ms])
    rhos=[]
    for a in AXES:
        r_a=avgrank([D[m][a] for m in ms]); rhos.append(pear(r_mae,r_a))
    pos=[x+(k-1)*w for x in xs]
    ax.bar(pos,rhos,width=w,color=colors[k],edgecolor='black',linewidth=0.6,
           hatch=hatches[k],label=ds)
ax.axhline(1.0,color='0.30',linewidth=0.7,linestyle=(0,(4,2)),zorder=1)
ax.axhline(0,color='black',linewidth=0.8)
ax.annotate('same order as MAE',xy=(0,1.0),xycoords=('axes fraction','data'),
            xytext=(3,1),textcoords='offset points',fontsize=5.6,color='0.30',
            ha='left',va='bottom',zorder=4)
ax.annotate('unrelated',xy=(0,0.0),xycoords=('axes fraction','data'),
            xytext=(3,-1),textcoords='offset points',fontsize=5.6,color='0.30',
            ha='left',va='top',zorder=4)
ax.set_xticks(list(xs)); ax.set_xticklabels([LABEL[a] for a in AXES],fontsize=7)
ax.set_ylabel('Spearman $\\rho$ vs. non-zero MAE',fontsize=8)
ax.text(0.5,-0.17,'(a)',transform=ax.transAxes,ha='center',va='top',fontsize=8)
ax.tick_params(labelsize=6); ax.legend(fontsize=7,frameon=False,ncol=3,loc='lower left')
ax.set_ylim(-0.25,1.18)
ax.set_xlim(-0.45,3.55)
ax.grid(axis='y',linewidth=0.3,alpha=0.5); ax.set_axisbelow(True)

# ---- panel (b): rank slopegraph, Parts MAE -> lumpy ----
ax=axes[1]
D=DATA['Parts']; ms=list(D)
r_mae=avgrank([D[m]['MAE'] for m in ms]); r_lu=avgrank([D[m]['lumpy RMSE'] for m in ms])
HL={'TCN':'#0072B2','PatchMixer':'#D55E00'}
STYLE={'TCN':'-','PatchMixer':(0,(4,2))}   # solid vs dashed so panel (b) survives greyscale printing
for m in ms:
    hl = m in HL
    ax.plot([0,1],[r_mae[ms.index(m)],r_lu[ms.index(m)]],
            color=HL.get(m,'#BFBFBF'), linewidth=1.8 if hl else 0.8,
            linestyle=STYLE.get(m,'-'),
            marker='o' if hl else None, markersize=4, zorder=3 if hl else 1)
    if hl:
        ax.annotate(m,(0,r_mae[ms.index(m)]),xytext=(-6,0),textcoords='offset points',
                    ha='right',va='center',fontsize=7,fontweight='bold',color=HL[m])
        ax.annotate(m,(1,r_lu[ms.index(m)]),xytext=(6,0),textcoords='offset points',
                    ha='left',va='center',fontsize=7,fontweight='bold',color=HL[m])
# 'rank 1' text removed: it collided with the PatchMixer label (PatchMixer is rank 1 by
# lumpy RMSE) and the y-axis already states rank 1 = best.
ax.invert_yaxis()
ax.set_xlim(-0.35,1.35); ax.set_xticks([0,1])
ax.set_xticklabels(['by non-zero\nMAE','by lumpy\nRMSE'],fontsize=7)
ax.set_ylabel('rank (1 = best)',fontsize=8)
ax.set_yticks([1,5,10,15,20]); ax.tick_params(labelsize=6)
ax.text(0.5,-0.17,'(b)',transform=ax.transAxes,ha='center',va='top',fontsize=8)
for s in ['top','right']: ax.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig(_os.path.join(_RESULTS,'fig_reorder.pdf'),bbox_inches='tight')
plt.savefig(_os.path.join(_RESULTS,'fig_reorder.png'),dpi=200,bbox_inches='tight')

# ---- §9.1 render-then-verify: bbox 重叠检查 ----
import matplotlib.text as mtext
r=fig.canvas.get_renderer()
texts=[(t,t.get_window_extent(r)) for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()]
spines=[(s,s.get_window_extent(r)) for ax in fig.axes for s in ax.spines.values() if s.get_visible()]
tick={ax:set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both')) for ax in fig.axes}
ov=[(a.get_text(),b.get_text()) for i,(a,ba) in enumerate(texts) for b,bb in texts[i+1:] if ba.overlaps(bb)]
ov+=[(t.get_text(),'spine') for t,bt in texts for s,bs in spines if bt.overlaps(bs) and t not in tick.get(s.axes,set())]
print("BBOX 重叠:", ov if ov else "无")
oob=[t.get_text() for t,b in texts if not fig.bbox.contains(*b.p0) or not fig.bbox.contains(*b.p1)]
print("超出画布:", oob if oob else "无")
print("saved results/fig_reorder.pdf and .png")
# print the rho table for the caption
for ds in ['Parts','Fresh','M5']:
    D=DATA[ds]; ms=list(D); r_mae=avgrank([D[m]['MAE'] for m in ms])
    vals=[f"{pear(r_mae,avgrank([D[m][a] for m in ms])):+.2f}" for a in AXES]
    print(f"  {ds:6s} n={len(ms):2d}  " + "  ".join(f"{LABEL[a][:6]}={v}" for a,v in zip(AXES,vals)))
