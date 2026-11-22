"""Original, reproducible teaching sketches; SVG plus editable Excalidraw scenes.

These are conceptual diagrams, not GPU traces. Dense tensors are drawn with
small representative grids. Each SVG embeds the OFL Kalam font for portability.
"""
from pathlib import Path
import base64, html, json, random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/diagrams'
PAPER = '#fffff8'
INK = '#29332e'
GREEN = '#2c7652'
BLUE = '#24758e'
ORANGE = '#b36a16'
PURPLE = '#7845a4'
RED = '#bd4b3b'
GRAY = '#737f72'

class Sketch:
    def __init__(self, name, height, title, width=1000):
        self.name, self.h, self.w, self.title = name, height, width, title
        self.svg, self.elements, self.patterns = [], [], {}
        self.random = random.Random(name)

    def element(self, kind, x, y, w, h, **extra):
        n = len(self.elements) + 1
        item = dict(id=f'{self.name}-{n}', type=kind, x=x, y=y, width=w,
                    height=h, angle=0, strokeColor=INK, backgroundColor='transparent',
                    fillStyle='hachure', strokeWidth=1.5, strokeStyle='solid',
                    roughness=1, opacity=100, groupIds=[], frameId=None,
                    roundness=None, seed=n*1789, version=1, versionNonce=n*2131,
                    isDeleted=False, boundElements=None, updated=1795305600000,
                    link=None, locked=False)
        item.update(extra); self.elements.append(item)

    def text(self, x, y, value, size=23, color=INK, anchor='start', mono=False):
        lines = value.split('\n'); w = max(map(len, lines))*size*.51
        self.element('text', x-w/2 if anchor=='middle' else x, y, w,
                     len(lines)*size*1.35, text=value, originalText=value,
                     fontSize=size, fontFamily=3 if mono else 1,
                     textAlign='center' if anchor=='middle' else 'left',
                     verticalAlign='top', containerId=None, autoResize=True,
                     lineHeight=1.35, strokeColor=color)
        family = 'Consolas, monospace' if mono else 'Kalam, cursive'
        self.svg.append(f'<text x="{x}" y="{y+size}" font-family="{family}" font-size="{size}" fill="{color}" text-anchor="{anchor}">' + ''.join(
            f'<tspan x="{x}" dy="{0 if i==0 else size*1.35}">{html.escape(line)}</tspan>'
            for i,line in enumerate(lines)) + '</text>')

    def line(self, x, y, x2, y2, color=INK, dashed=False, arrow=False, width=1.5):
        # Lightly irregular strokes, without distorting coordinates or labels.
        mx,my=(x+x2)/2,(y+y2)/2
        jitter=self.random.uniform(-.7,.7)
        self.svg.append(f'<path d="M{x} {y} Q{mx+jitter} {my+jitter} {x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"' +
                        (' stroke-dasharray="5 6"' if dashed else '') +
                        (f' marker-end="url(#arrow-{color[1:]})"' if arrow else '') + '/>')
        self.element('arrow' if arrow else 'line',x,y,abs(x2-x),abs(y2-y),
                     points=[[0,0],[x2-x,y2-y]],lastCommittedPoint=None,
                     startBinding=None,endBinding=None,startArrowhead=None,
                     endArrowhead='arrow' if arrow else None,elbowed=False,
                     strokeColor=color,strokeStyle='dashed' if dashed else 'solid')

    def rect(self,x,y,w,h,color=INK,fill=None,hatch=True):
        fillvalue='none'
        if fill:
            if hatch:
                self.patterns[fill]=True;fillvalue=f'url(#hatch-{fill[1:]})'
            else:fillvalue=fill
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fillvalue}"/>')
        for a,b,c,d in [(x,y,x+w,y),(x+w,y,x+w,y+h),(x+w,y+h,x,y+h),(x,y+h,x,y)]:
            j=self.random.uniform(-.7,.7)
            self.svg.append(f'<path d="M{a} {b} Q{(a+c)/2+j} {(b+d)/2+j} {c} {d}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        self.element('rectangle',x,y,w,h,strokeColor=color,
                     backgroundColor=fill or 'transparent',fillStyle='hachure' if hatch else 'solid')

    def matrix(self,x,y,w,h,rows=6,cols=8,row=None,col=None,color=BLUE):
        if row is not None:self.rect(x,y+row*h/rows,w,h/rows,color,color)
        if col is not None:self.rect(x+col*w/cols,y,w/cols,h,color,color)
        self.rect(x,y,w,h)
        for r in range(1,rows):self.line(x,y+r*h/rows,x+w,y+r*h/rows,GRAY,True,width=.7)
        for c in range(1,cols):self.line(x+c*w/cols,y,x+c*w/cols,y+h,GRAY,True,width=.7)

    def box(self,x,y,w,h,label,color=GREEN,size=23):
        self.rect(x,y,w,h,color,color)
        self.text(x+w/2,y+(h-size*1.35*(label.count('\n')+1))/2,label,size,anchor='middle')

    def save(self):
        font=base64.b64encode((ROOT/'assets/fonts/kalam-regular.woff2').read_bytes()).decode()
        defs=f'<style>@font-face{{font-family:Kalam;src:url(data:font/woff2;base64,{font}) format("woff2");font-weight:400;font-style:normal}}</style>'
        for color in [INK,GREEN,BLUE,ORANGE,PURPLE,RED,GRAY]:
            defs+=f'<marker id="arrow-{color[1:]}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L8 5 L1 9" fill="none" stroke="{color}" stroke-width="1.6"/></marker>'
        for color in self.patterns:
            defs+=f'<pattern id="hatch-{color[1:]}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(35)"><rect width="6" height="6" fill="{color}" opacity=".035"/><path d="M0 0 V6" stroke="{color}" stroke-width=".85" opacity=".6"/></pattern>'
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" role="img"><title>{html.escape(self.title)}</title><defs>{defs}</defs><rect width="{self.w}" height="{self.h}" fill="{PAPER}"/>'+''.join(self.svg)+'</svg>'
        (OUT/f'{self.name}.svg').write_text(svg,encoding='utf-8')
        (OUT/f'{self.name}.excalidraw').write_text(json.dumps(dict(type='excalidraw',version=2,source='https://excalidraw.com',elements=self.elements,appState=dict(viewBackgroundColor=PAPER,gridSize=None),files={}),indent=2)+'\n',encoding='utf-8')

def serving():
    s=Sketch('serving-path',610,'A request, the target verification loop, and persistent KV state')
    s.text(25,10,'What has to happen before the next token can be returned?',28)
    for x,w,label,c in [(25,175,'Conversation\n+ request',BLUE),(250,185,'Schedule\n+ prefill',BLUE),(490,180,'Draft\nproposals',PURPLE),(735,235,'Target\nverification',GREEN)]:s.box(x,90,w,80,label,c)
    for a,b in [(202,245),(438,485),(673,730)]:s.line(a,130,b,130,arrow=True)
    s.line(850,173,850,220,arrow=True)
    s.rect(25,225,945,235,GRAY)
    s.text(45,240,'Inside a target layer, across eight ranks',23)
    s.box(55,305,215,68,'Prepare Q / K / V',ORANGE,22)
    s.box(315,305,160,68,'FA3 attention',BLUE,22)
    s.box(520,305,175,68,'Route tokens',PURPLE,22)
    s.box(740,305,200,68,'Local experts',GREEN,22)
    for a,b in [(273,310),(478,515),(698,735)]:s.line(a,339,b,339,arrow=True)
    s.text(55,403,'Attention fusion changes this',21,ORANGE)
    s.text(535,403,'Expert scheduling changes this',21,GREEN)
    s.text(45,485,'Collectives and the rest of the model\nstill have to complete.',22)
    s.box(580,500,360,70,'Accept tokens + update state',BLUE,23)
    s.line(850,462,850,495,arrow=True)
    s.line(943,535,990,535,PURPLE);s.line(990,535,990,195,PURPLE);s.line(990,195,580,195,PURPLE);s.line(580,195,580,175,PURPLE,arrow=True)
    s.text(540,576,'Repeat with the accepted prefix',20,PURPLE)
    s.save()

def geometry():
    s=Sketch('expert-geometry',620,'The routed activation count is the row dimension of an expert GEMM')
    s.text(25,12,'One local expert: the weights stay the same; the row count changes.',26)
    s.text(155,80,'Routed activations',24,anchor='middle')
    s.text(505,80,'Logical expert weights',24,anchor='middle')
    s.text(840,80,'Gate + up projection',24,anchor='middle')
    s.matrix(45,155,240,150,6,8,row=2,color=GREEN)
    s.matrix(390,125,235,235,8,8,col=3,color=BLUE)
    s.matrix(720,155,240,150,6,8,row=2,color=PURPLE)
    s.rect(720+3*30,155+2*25,30,25,PURPLE,PURPLE)
    s.text(330,205,'×',34,anchor='middle');s.text(670,205,'=',34,anchor='middle')
    s.text(160,325,'Aₑ: mₑ × 4096',24,anchor='middle')
    s.text(505,380,'W₁₃: 4096 × 8192',24,anchor='middle')
    s.text(840,325,'Hₑ: mₑ × 8192',24,anchor='middle')
    s.line(34,155,34,305,GREEN,arrow=True)
    s.text(40,400,'mₑ is the number of routes\nassigned to this expert.',24,GREEN)
    s.line(220,395,210,350,GREEN,arrow=True)
    s.line(730,435,930,435,PURPLE,arrow=True)
    s.text(720,450,'Split gate / up, activate,\nproject back to 4096.',23,PURPLE)
    s.line(25,523,970,523,GRAY,True)
    s.text(35,546,'12 routes and 50 routes use the same weight dimensions.\nOnly the useful activation rows differ. Grids are illustrative, not to scale.',24)
    s.save()

def routing():
    s=Sketch('expert-routing',715,'Schedule lightly loaded experts with smaller tiles without splitting route ownership')
    s.text(25,10,'Count routes per expert, then choose the execution class.',27)
    s.text(30,65,'Global IDs',23);s.text(335,65,'Local IDs on this rank',23)
    for i,(g,l) in enumerate([(32,0),(33,1),(7,-1),(32,0),(65,-1),(33,1)]):
        c=GRAY if l<0 else GREEN if l==0 else BLUE
        s.box(30+i*43,108,36,40,str(g),c,19)
        s.box(355+i*43,108,36,40,str(l),c,19)
    s.line(292,128,342,128,arrow=True)
    s.text(665,83,'Toy ownership map:\n32 → 0, 33 → 1\nOther ranks’ IDs → −1',22)
    s.text(35,164,'A six-route excerpt; the histogram counts the entire forward.',21,GRAY)
    s.line(25,217,970,217,GRAY,True)
    s.text(35,238,'Suppose the complete histogram gives:',24)
    s.text(70,289,'Expert 0: 12 routes',24,GREEN)
    s.text(555,289,'Expert 1: 50 routes',24,BLUE)
    # Each visible row stands for two physical rows. Counts and labels are exact.
    for x,used,total,color in [(75,12,32,GREEN),(560,50,64,BLUE)]:
        s.rect(x,345,160,210,GRAY,GRAY)
        s.rect(x,345,160,210*used/total,color,color)
        for k in range(1,total//2):s.line(x,345+210*k/(total//2),x+160,345+210*k/(total//2),GRAY,True,width=.6)
        s.text(x+190,345,f'{used} useful rows',22,color)
        s.line(x+180,376,x+162,360+100*used/total,color,arrow=True)
        s.text(x+190,454,f'{total-used} unused\ntile rows',22,GRAY)
        s.line(x+180,490,x+162,530,GRAY,arrow=True)
    s.text(70,575,'12 ≤ 16: use M = 32',24,GREEN)
    s.text(555,575,'50 > 16: retain stock tile',24,BLUE)
    s.text(35,633,'Illustrated stock M = 64. The actual stock family can also be M = 48.\nThese are tile slots, not measured memory traffic or elapsed time.',22)
    s.save()

def attention():
    s=Sketch('attention-fusion',610,'Fuse preparation while leaving FA3 attention and required rounding intact')
    s.text(25,12,'Remove intermediate launches; keep the same consumer.',27)
    s.text(25,75,'STOCK',21)
    xs=[150,325,445,565,690,825];ws=[150,95,95,100,110,145]
    labels=['QK norm\n+ RoPE','K cast','V cast','Store\nK / V','Q cast','FA3']
    for x,w,label in zip(xs,ws,labels):s.box(x,80,w,78,label,BLUE if label=='FA3' else GRAY,22)
    for x,w,n in zip(xs,ws,xs[1:]):s.line(x+w+2,119,n-5,119,arrow=True)
    s.text(325,176,'FP8 cache bytes',21,BLUE);s.text(685,176,'FP8 Q',21,PURPLE)
    s.text(25,260,'A',25)
    s.box(150,250,515,78,'QK norm + RoPE + FP8 KV store',GREEN,26)
    s.box(690,250,110,78,'Q cast',GRAY,22);s.box(825,250,145,78,'FA3',BLUE,23)
    s.line(668,289,685,289,arrow=True);s.line(802,289,820,289,arrow=True)
    s.text(155,346,'BF16 query leaves the kernel; its cast still runs.',23)
    s.text(25,430,'B',25)
    s.box(150,420,650,78,'QK norm + RoPE + FP8 KV store + FP8 Q',GREEN,25)
    s.box(825,420,145,78,'FA3',BLUE,23);s.line(802,459,820,459,arrow=True)
    s.text(155,516,'BF16 rounding still happens inside the fused kernel.',24,PURPLE)
    s.text(155,563,'Schematic launch sequence: widths do not represent duration.',21,GRAY)
    s.save()

def cache():
    s=Sketch('cache-writes',560,'A cache store must preserve addresses, padding and untouched slots')
    s.text(25,12,'Fusing the store does not give the kernel a new address map.',27)
    s.text(35,78,'Prepared token rows',24);s.text(365,78,'Cache locations',24)
    for y,label,loc,color in [(135,'token A',17,GREEN),(230,'padding',0,GRAY),(325,'token B',42,PURPLE)]:
        s.box(40,y,230,62,label,color,24);s.box(370,y,85,62,str(loc),color,24)
        s.line(275,y+31,362,y+31,color,arrow=True)
    s.text(620,78,'The selected KV pool',24)
    for y,label,color in [(125,'slot 0: unchanged',GRAY),(205,'… untouched slots …',GRAY),(285,'slot 17: write A',GREEN),(365,'slot 42: write B',PURPLE)]:
        s.rect(625,y,325,56,color,color if color!=GRAY else None)
        s.text(642,y+10,label,23,color)
    s.line(460,166,600,166,GREEN);s.line(600,166,600,313,GREEN);s.line(600,313,620,313,GREEN,arrow=True)
    s.line(460,356,565,356,PURPLE);s.line(565,356,565,393,PURPLE);s.line(565,393,620,393,PURPLE,arrow=True)
    s.text(480,240,'skip',23,RED);s.line(462,261,550,261,RED,True)
    s.text(35,455,'Use the backend’s full / sliding-window locations.\nWait for cache ownership before writing. Keep other slots intact.',25)
    s.text(35,529,'Example addresses only; this is not the physical cache layout.',20,GRAY)
    s.save()

def boundary():
    s=Sketch('rounding-boundary',625,'Adding V rows moves the launch across a tile threshold at 56 tokens')
    s.text(25,10,'The extra V rows change more than the amount of work.',27)
    s.text(35,74,'Rows per token',23)
    for i in range(20):
        col=BLUE if i<16 else GREEN if i<18 else ORANGE
        s.rect(35+i*44,123,42,50,col,col)
    s.text(35,186,'16 Q heads',23,BLUE);s.text(737,186,'2 K',23,GREEN);s.text(840,186,'2 V',23,ORANGE)
    s.text(35,245,'At T = 56 tokens',26)
    s.line(55,399,950,399,INK,arrow=True)
    # Linear numeric axis from 900 to 1200 rows, positions exact.
    def axis(n):return 55+(n-900)/300*870
    for n in [900,1008,1024,1120,1200]:
        s.line(axis(n),390,axis(n),410,GRAY)
    s.line(axis(1024),310,axis(1024),454,RED,True)
    s.text(axis(1024),465,'1,024-row threshold',23,RED,anchor='middle')
    s.line(axis(1008),375,axis(1008),391,BLUE,arrow=True)
    s.text(55,292,'Stock: 18 × 56 = 1,008',24,BLUE)
    s.text(55,328,'BLOCK_M = 4',23,BLUE)
    s.line(axis(1120),375,axis(1120),391,ORANGE,arrow=True)
    s.text(620,292,'Fused: 20 × 56 = 1,120',24,ORANGE)
    s.text(620,328,'BLOCK_M = 8',23,ORANGE)
    s.text(60,535,'Same Q and K inputs. Different tile → potentially different reduction grouping.\nThe threshold tells us where to test; it does not predict every failure.',24)
    s.save()

def midpoint():
    s=Sketch('rounding-midpoint',530,'A tiny reduction difference can cross a stored-value rounding boundary')
    s.text(25,10,'Same formula, slightly different finite-precision result.',27)
    s.text(35,80,'sum of squares → RMS scale → normalized value',26)
    s.text(35,128,'A changed reduction grouping perturbs the scale by a tiny amount.',23,GRAY)
    s.line(110,260,895,260,INK,arrow=True)
    for x,c in [(190,BLUE),(500,GRAY),(810,PURPLE)]:s.line(x,246,x,274,c)
    s.text(190,300,'lower BF16 value',23,BLUE,anchor='middle')
    s.text(810,300,'upper BF16 value',23,PURPLE,anchor='middle')
    s.line(500,195,500,288,RED,True)
    s.text(500,165,'rounding midpoint',22,RED,anchor='middle')
    s.line(475,215,475,249,BLUE,arrow=True);s.line(525,215,525,249,PURPLE,arrow=True)
    s.line(475,275,205,290,BLUE,arrow=True);s.line(525,275,795,290,PURPLE,arrow=True)
    s.text(330,202,'x − ε',23,BLUE);s.text(580,202,'x + ε',23,PURPLE)
    s.text(35,380,'Most values still round identically. Values near a midpoint can separate.\nIf those BF16 neighbors straddle an FP8 rounding boundary too,\nthe final query byte changes.',25)
    s.text(35,494,'Schematic number line; distances are exaggerated, not fixture values.',21,GRAY)
    s.save()

def evidence():
    s=Sketch('evidence-ladder',500,'Separate source, local and serving evidence instead of treating one as proof of the others')
    s.text(25,10,'What did each experiment actually test?',27)
    cols=[(30,285,'LOCAL L40',GREEN),(355,285,'SM90 COMPILE',PURPLE),(680,285,'OFFICIAL H200',BLUE)]
    for x,w,label,c in cols:
        s.text(x,78,label,23,c);s.line(x,119,x+w,119,c)
    s.text(30,146,'Exact Q / K / cache bytes\nSaved failing fixtures\nChanging-input replay\nModel fallback dispatch',24)
    s.text(355,146,'Can target code compile?\nInspect generated code\n\nNo target execution',24)
    s.text(680,146,'Real FA3 + PDL\nEight-rank serving\nLater-turn correctness\nMemory + reported score',24)
    s.line(25,330,970,330,GRAY,True)
    s.text(35,364,'A local pass cannot replace integrated H200 execution.\nAn official pass cannot tell us which kernel caused a score change.',25)
    s.text(35,456,'Different questions need different evidence.',24,GREEN)
    s.save()

def miniatures():
    s=Sketch('tile-note',355,'Small-expert threshold and row-tile size are different quantities',width=420)
    s.text(15,8,'Threshold ≠ tile size',27)
    s.rect(28,75,120,205,GRAY,GRAY);s.rect(28,75,120,77,GREEN,GREEN)
    for i in range(1,16):s.line(28,75+i*205/16,148,75+i*205/16,GRAY,True,width=.7)
    s.text(180,80,'12 routes',24,GREEN);s.line(175,116,152,116,GREEN,arrow=True)
    s.text(180,184,'32-row tile',24);s.line(175,223,152,255,INK,arrow=True)
    s.text(25,306,'12 ≤ 16 selects this class.',24)
    s.save()
    s=Sketch('rounding-note',290,'Preserve the intermediate BF16 rounding before FP8 conversion',width=420)
    s.text(15,8,'Keep the rounding step',26)
    s.box(15,65,110,55,'FP32',GRAY,24);s.box(155,65,110,55,'BF16',BLUE,24);s.box(295,65,110,55,'FP8',PURPLE,24)
    s.line(128,92,150,92,arrow=True);s.line(268,92,290,92,arrow=True)
    s.line(70,138,70,180,RED);s.line(70,180,350,180,RED,True);s.line(350,180,350,135,RED,arrow=True)
    s.text(180,162,'×',32,RED)
    s.text(20,218,'Direct FP32 → FP8 is not\ngenerally the same conversion.',23)
    s.save()

def render():
    OUT.mkdir(parents=True,exist_ok=True)
    for fn in [serving,geometry,routing,attention,cache,boundary,midpoint,evidence,miniatures]:fn()

if __name__=='__main__':render()
