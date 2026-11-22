"""Original vector diagrams with equivalent, editable Excalidraw scenes."""
from pathlib import Path
import json, html, random, csv
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets'/'diagrams'
INK='#34483c'; GREEN='#e5eee1'; BLUE='#e7edf4'; ORANGE='#f4e5d6'; PAPER='#faf9f5'; GRAY='#eeeee7'

class Scene:
    def __init__(self,name,h,title):
        self.name=name;self.h=h;self.elements=[];self.svg=[];self.r=random.Random(name);self.title=title
    def element(self,kind,x,y,w,h,**more):
        i=len(self.elements)+1
        e=dict(id=f'{self.name}-{i}',type=kind,x=x,y=y,width=w,height=h,angle=0,
            strokeColor=INK,backgroundColor='transparent',fillStyle='solid',strokeWidth=1.5,
            strokeStyle='solid',roughness=1,opacity=100,groupIds=[],frameId=None,roundness=None,
            seed=i*1789,version=1,versionNonce=i*2131,isDeleted=False,boundElements=None,
            updated=1795305600000,link=None,locked=False)
        e.update(more);self.elements.append(e);return e
    def text(self,x,y,t,size=22,color=INK,align='left',hand=False):
        lines=t.split('\n');w=max(len(a) for a in lines)*size*.55;h=len(lines)*size*1.3
        self.element('text',x-w/2 if align=='center' else x,y,w,h,text=t,originalText=t,fontSize=size,fontFamily=1 if hand else 2,textAlign=align,verticalAlign='top',containerId=None,autoResize=True,lineHeight=1.3,strokeColor=color)
        family='Segoe Print, Comic Sans MS, cursive' if hand else 'Segoe UI, Arial, sans-serif'
        self.svg.append(f'<text x="{x}" y="{y+size}" font-family="{family}" font-size="{size}" fill="{color}" text-anchor="{"middle" if align=="center" else "start"}">'+''.join(f'<tspan x="{x}" dy="{0 if n==0 else size*1.3}">{html.escape(line)}</tspan>' for n,line in enumerate(lines))+'</text>')
    def box(self,x,y,w,h,label,fill=GREEN,size=21):
        self.element('rectangle',x,y,w,h,backgroundColor=fill,roundness={'type':3})
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{INK}" stroke-width="1.5"/>')
        self.svg.append(f'<path d="M{x+4} {y+2} Q{x+w*.5} {y-1} {x+w-4} {y+2}" fill="none" stroke="{INK}" stroke-width=".7" opacity=".32"/>')
        lines=label.count('\n')+1;self.text(x+w/2,y+(h-lines*size*1.3)/2,label,size,align='center')
    def arrow(self,x,y,x2,y2,points=None):
        pts=points or [(x,y),(x2,y2)]
        self.element('arrow',x,y,max(abs(px-x) for px,py in pts),max(abs(py-y) for px,py in pts),points=[[px-x,py-y] for px,py in pts],lastCommittedPoint=None,startBinding=None,endBinding=None,startArrowhead=None,endArrowhead='arrow',elbowed=False)
        self.svg.append('<polyline points="'+' '.join(f'{px},{py}' for px,py in pts)+'" fill="none" stroke="'+INK+'" stroke-width="1.6" stroke-linejoin="round" marker-end="url(#arrow)"/>')
    def save(self):
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {self.h}" role="img"><title>{html.escape(self.title)}</title><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L8 5 L1 9" fill="none" stroke="{INK}" stroke-width="1.4"/></marker></defs><rect width="1000" height="{self.h}" fill="{PAPER}"/>'+''.join(self.svg)+'</svg>'
        (OUT/f'{self.name}.svg').write_text(svg,encoding='utf-8')
        (OUT/f'{self.name}.excalidraw').write_text(json.dumps(dict(type='excalidraw',version=2,source='https://excalidraw.com',elements=self.elements,appState=dict(viewBackgroundColor=PAPER,gridSize=None),files={}),indent=2),encoding='utf-8')

def diagrams():
    s=Scene('serving-path',430,'The serving loop and the two optimization sites')
    s.text(30,15,'A conversation keeps its state between turns',25,hand=True)
    for x,label in [(25,'Request +\nconversation'),(270,'Schedule +\nprefill'),(515,'Draft\nproposals'),(760,'Target\nverification')]:s.box(x,95,215,82,label,BLUE)
    for x in [240,485,730]:s.arrow(x,136,x+28,136)
    s.box(620,270,300,70,'Accept + update KV state',GRAY,20)
    s.arrow(870,178,870,265)
    s.arrow(620,305,620,181,[(620,305),(570,305),(570,209),(620,209),(620,181)])
    s.text(25,237,'Inside target layers',21)
    s.box(25,282,235,64,'Attention preparation',ORANGE,19)
    s.box(285,282,230,64,'Expert scheduling',GREEN,19)
    s.text(30,365,'Two local changes; one end-to-end score.',22,hand=True)
    s.save()
    s=Scene('expert-routing',410,'Map routes and schedule whole experts by occupancy')
    s.text(30,12,'The token batch is not the expert batch',25,hand=True)
    s.box(30,100,210,80,'Global route IDs\n8 per token',BLUE)
    s.box(300,100,245,80,'Map + histogram\nlocal expert loads',GREEN)
    s.arrow(242,140,296,140)
    s.box(640,65,325,80,'≤ 16 routes / expert\n32-row Marlin tile',GREEN)
    s.box(640,185,325,80,'> 16 routes / expert\nkeep stock tile',ORANGE)
    s.arrow(547,140,635,105,[(547,140),(590,140),(590,105),(635,105)])
    s.arrow(590,140,635,225,[(590,140),(590,225),(635,225)])
    s.text(307,220,'Non-local routes → −1\nRemain inactive in both classes',18)
    s.text(40,325,'Every live route once. Same weights. Same finalization.',23,hand=True)
    s.save()
    s=Scene('attention-fusion',510,'Attention preparation before and after fusion')
    for y,lab in [(50,'STOCK'),(200,'VARIANT A'),(350,'VARIANT B')]:s.text(25,y+18,lab,16)
    labels=['QK norm\n+ RoPE','Cast K','Cast V','KV store','Cast Q','FA3']
    xs=[170,325,450,575,700,835];ws=[130,100,100,100,100,135]
    for x,w,label in zip(xs,ws,labels):s.box(x,50,w,72,label,BLUE if label=='FA3' else GRAY,17)
    for x,w,nx in zip(xs,ws,xs[1:]):s.arrow(x+w+2,86,nx-5,86)
    s.box(170,200,505,72,'QK norm + RoPE + FP8 KV store',GREEN,21)
    s.box(700,200,100,72,'Cast Q',GRAY,17);s.box(835,200,135,72,'FA3',BLUE)
    s.arrow(677,236,695,236);s.arrow(802,236,830,236)
    s.box(170,350,630,72,'QK norm + RoPE + FP8 KV store + FP8 query',GREEN,20)
    s.box(835,350,135,72,'FA3',BLUE);s.arrow(802,386,830,386)
    s.text(175,140,'Intermediate materialization + separate launches',18,hand=True)
    s.text(175,290,'BF16 query retained; ordinary cast still runs',18,hand=True)
    s.text(175,450,'Preserve BF16 rounding before the final FP8 conversion',20,hand=True)
    s.save()
    s=Scene('rounding-boundary',400,'Different row counts can change the reduction layout')
    s.text(35,10,'Same 56 tokens. Different launch configuration.',24,hand=True)
    s.box(35,80,420,100,'Stock: 18 × 56 = 1,008 rows\nBLOCK_M = 4',BLUE,23)
    s.box(540,80,425,100,'Original fusion: 20 × 56 = 1,120\nBLOCK_M = 8',ORANGE,22)
    s.arrow(245,183,245,237);s.arrow(750,183,750,237)
    s.box(35,242,420,65,'Different FP32 reduction grouping',GRAY,19)
    s.box(540,242,425,65,'A value crosses a rounding midpoint',ORANGE,18)
    s.arrow(458,275,535,275)
    s.text(40,339,'One changed query byte is enough to fail an exact comparison.',22,hand=True)
    s.save()
    s=Scene('evidence-ladder',390,'Different evidence answers different questions')
    xs=[25,270,515,760]
    labels=['Source\nreasoning','Exact local\ncomparisons','Changing-input\ngraph replay','Official H200\nserving run']
    notes=['Does the mechanism\nmake sense?','Do these bytes\nmatch stock?','Are state + addresses\ncorrect on replay?','Does this bundle pass\nthe real workload?']
    for x,label,note in zip(xs,labels,notes):
        s.box(x,65,215,90,label,BLUE if x==760 else GREEN,21);s.text(x+107,182,note,18,align='center')
    for x in [240,485,730]:s.arrow(x+2,110,x+25,110)
    s.text(45,292,'Compile ≠ execute. Pass ≠ speedup. One run ≠ certainty.',23,hand=True)
    s.save()

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
if __name__=='__main__':diagrams();results()
