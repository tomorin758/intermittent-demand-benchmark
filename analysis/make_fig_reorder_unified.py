import csv, math, itertools
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
_os.makedirs(_RESULTS, exist_ok=True)
FILES={'Parts':_os.path.join(_TABLES,'parts_unified.csv'),
       'Fresh':_os.path.join(_TABLES,'fresh_retail_unified.csv'),
       'M5':_os.path.join(_TABLES,'m5_unified.csv')}
AXES=['(q90, ∞) RMSE','intermittent RMSE','lumpy RMSE','smooth RMSE']
LABEL={'(q90, ∞) RMSE':'Tail\n$(q_{90},\\infty)$','intermittent RMSE':'Intermittent',
       'lumpy RMSE':'Lumpy','smooth RMSE':'Smooth'}
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
    rows=list(csv.reader(open(f,encoding='utf-8-sig')))
    hdr=[c.strip() for c in rows[1]]
    idx={c:hdr.index(c) for c in ['MAE']+AXES}
    D={}
    for r in rows[2:]:
        if len(r)<2 or not r[1].strip() or '(\u03c4=0.5)' in r[1]: continue   # untruncated rows = ranking reference
        D[r[1].strip()]={c:num(r[idx[c]]) for c in idx}
    DATA[ds]=D

fig,axes=plt.subplots(1,2,figsize=(7.4,2.9))
# ---- panel (a): Spearman rho vs non-zero MAE ----
ax=axes[0]
hatches=['','///','...','xxx']
grays=['0.15','0.45','0.7','0.9']
w=0.2; xs=range(len(AXES))
for k,ds in enumerate(['Parts','Fresh','M5']):
    D=DATA[ds]; ms=list(D)
    r_mae=avgrank([D[m]['MAE'] for m in ms])
    rhos=[]
    for a in AXES:
        r_a=avgrank([D[m][a] for m in ms]); rhos.append(pear(r_mae,r_a))
    pos=[x+(k-1)*w for x in xs]
    ax.bar(pos,rhos,width=w,color=grays[k],edgecolor='black',linewidth=0.6,
           hatch=hatches[k],label=ds)
ax.axhline(0,color='black',linewidth=0.8)
ax.set_xticks(list(xs)); ax.set_xticklabels([LABEL[a] for a in AXES],fontsize=7)
ax.set_ylabel('Spearman $\\rho$ vs. non-zero MAE',fontsize=8)
ax.set_title('(a) Rank agreement with MAE depends on axis and dataset',fontsize=8,loc='left')
ax.tick_params(labelsize=6); ax.legend(fontsize=7,frameon=False,ncol=3,loc='lower left')
ax.set_ylim(-0.45,1.12)
ax.grid(axis='y',linewidth=0.3,alpha=0.5); ax.set_axisbelow(True)

# ---- panel (b): rank slopegraph, Parts MAE -> lumpy ----
ax=axes[1]
D=DATA['Parts']; ms=list(D)
r_mae=avgrank([D[m]['MAE'] for m in ms]); r_lu=avgrank([D[m]['lumpy RMSE'] for m in ms])
HL={'TCN':'#1f4e9c','PatchMixer':'#b2182b'}
STYLE={'TCN':'-','PatchMixer':(0,(4,2))}   # solid vs dashed so panel (b) survives greyscale printing
for m in ms:
    hl = m in HL
    ax.plot([0,1],[r_mae[ms.index(m)],r_lu[ms.index(m)]],
            color=HL.get(m,'0.78'), linewidth=1.8 if hl else 0.8,
            linestyle=STYLE.get(m,'-'),
            marker='o' if hl else None, markersize=4, zorder=3 if hl else 1)
    if hl:
        ax.annotate(m,(0,r_mae[ms.index(m)]),xytext=(-6,0),textcoords='offset points',
                    ha='right',va='center',fontsize=7,fontweight='bold',color=HL[m])
        ax.annotate(m,(1,r_lu[ms.index(m)]),xytext=(6,0),textcoords='offset points',
                    ha='left',va='center',fontsize=7,fontweight='bold',color=HL[m])
ax.annotate('rank 1',(1,1),xytext=(6,0),textcoords='offset points',ha='left',va='center',fontsize=7)
ax.invert_yaxis()
ax.set_xlim(-0.35,1.35); ax.set_xticks([0,1])
ax.set_xticklabels(['by non-zero\nMAE','by lumpy\nRMSE'],fontsize=7)
ax.set_ylabel('rank (1 = best)',fontsize=8)
ax.set_yticks([1,5,10,15,20]); ax.tick_params(labelsize=6)
ax.set_title('(b) Parts: the same models cross over',fontsize=8,loc='left')
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
