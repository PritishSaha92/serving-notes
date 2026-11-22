"""Build original editable diagrams and the evidence-backed results chart."""
from pathlib import Path
import csv
from sketches import render
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/diagrams'
PAPER='#fffff8'

def results():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    rows=list(csv.DictReader((ROOT/'evidence'/'results.csv').open()))
    best=[(float(r['fused_ratio'])-1)*100 for r in rows[:4]]
    repair=[(float(r['repaired_ratio'])-1)*100 for r in rows[:4]]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'path'})
    fig,ax=plt.subplots(figsize=(10,4.6),facecolor=PAPER)
    ax.set_facecolor(PAPER)
    import numpy as np
    x=np.arange(4); w=.28
    a=ax.bar(x-w/2,best,w,label='Fused expert scheduling',color='#35624f',zorder=3)
    b=ax.bar(x+w/2,repair,w,label='+ Repaired attention preparation',color='#b67d59',zorder=3)
    ax.bar_label(a,labels=[f'{v:.2f}%' for v in best],padding=5,fontsize=10,color='#254f43')
    ax.bar_label(b,labels=[f'{v:.2f}%' for v in repair],padding=5,fontsize=10,color='#875334')
    ax.set_xticks(x,['1 user','4 users','16 users','32 users']);ax.set_ylim(0,4.65)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.0f}%'))
    ax.set_ylabel('Improvement over paired stock',labelpad=10,color='#59625a')
    ax.grid(axis='y',color='#dcded4',linewidth=.6,zorder=0)
    for k in ['top','right','left']:ax.spines[k].set_visible(False)
    ax.spines['bottom'].set_color('#b9c2b4');ax.tick_params(length=0,pad=10)
    ax.legend(loc='upper left',frameon=False,ncol=2,bbox_to_anchor=(0,1.18),fontsize=10)
    fig.subplots_adjust(left=.10,right=.99,bottom=.13,top=.83)
    fig.savefig(OUT/'results.svg',metadata={'Title':'Reported serving improvements; separate runs, no inferred error bars','Date':None})
    plt.close(fig)
if __name__=='__main__':render();results()
