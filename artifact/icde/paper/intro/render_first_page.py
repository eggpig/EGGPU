"""Render a fully specified ownership counterexample, never performance data."""
from pathlib import Path
import argparse, json, hashlib, itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COLORS = dict(topology='#B8D3E7', weights='#BED4AA', other='#C9C3D8',
              actual='#356A9A', referenced='#D3D3D3', threshold='#C88646', ink='#20252B')

def bind(path):
    data = path.read_bytes()
    return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

def compute(model):
    arrays=model['arrays']; views=set().union(*(set(a['owners']) for a in arrays))
    assert all(a['units']>0 and a['owners'] for a in arrays)
    total=sum(a['units'] for a in arrays); gap=total-model['budget_units']
    def released(evicted):
        return sum(a['units'] for a in arrays if set(a['owners']) <= set(evicted))
    def referenced(evicted):
        return sum(a['units'] for v in evicted for a in arrays if v in a['owners'])
    rows=[dict(evicted=e,referenced=referenced(e),released=released(e),
               remaining=total-released(e),budget_met=released(e)>=gap)
          for e in model['candidates']]
    assert total==16 and gap==6
    assert [(r['referenced'],r['released'],r['remaining'],r['budget_met']) for r in rows]==[(9,1,15,False),(18,10,6,True),(6,6,10,True)]
    assert referenced(views)==24 and released(views)==16 and released([])==0
    # Every subset is checked: each array is reclaimed once and only after all owners leave.
    all_sets=[set(c) for n in range(len(views)+1) for c in itertools.combinations(sorted(views),n)]
    assert all(0<=released(s)<=total for s in all_sets)
    assert all(released(a)<=released(b) for a in all_sets for b in all_sets if a<=b)
    return dict(physical_total=total,logical_reference_total=referenced(views),budget=model['budget_units'],required_release=gap,candidates=rows)

def box(ax,x,y,w,h,label,color,fs=8.3):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.018',
                         fc=color,ec=COLORS['ink'],lw=.7,zorder=3)
    ax.add_patch(patch);ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,zorder=4,color=COLORS['ink'])

def arrow(ax,start,end,style='-|>',connection='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=7,lw=.75,
                                color='#59616B',connectionstyle=connection,zorder=2))

def ownership(ax,wide=False):
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(0,1.015,'(a) Shared physical state',ha='left',va='bottom',fontsize=8.5)
    ax.text(.34,.925,'A weight update creates a new version',ha='center',fontsize=7.3,color='#4A515A')
    box(ax,.08,.69,.22,.14,r'View $G^0$','white')
    box(ax,.41,.69,.22,.14,r'View $G^1$','white')
    box(ax,.75,.69,.20,.14,r'View $H$','white')
    arrow(ax,(.305,.765),(.402,.765))
    box(ax,.13,.38,.46,.15,r'Shared topology $T$: 8',COLORS['topology'])
    box(ax,.08,.13,.22,.14,r'$W^0$: 1',COLORS['weights'])
    box(ax,.41,.13,.22,.14,r'$W^1$: 1',COLORS['weights'])
    box(ax,.75,.31,.20,.22,'State H\n6',COLORS['other'])
    arrow(ax,(.19,.685),(.265,.535));arrow(ax,(.52,.685),(.445,.535))
    # Distinct weight-owner arrows route around, rather than through, topology.
    arrow(ax,(.09,.70),(.09,.275),connection='arc3,rad=.25')
    arrow(ax,(.62,.70),(.62,.275),connection='arc3,rad=-.25')
    arrow(ax,(.85,.685),(.85,.535))
    ax.text(.50,.025,'Physical total: 16    Retention budget: 10',ha='center',fontsize=7.6)

def releases(ax,result,wide=False):
    xs=range(3);w=.29
    ref=[r['referenced'] for r in result['candidates']];actual=[r['released'] for r in result['candidates']]
    ax.bar([x-w/2 for x in xs],ref,width=w,color=COLORS['referenced'],edgecolor=COLORS['ink'],lw=.55,label='Referenced',hatch='///',zorder=2)
    ax.bar([x+w/2 for x in xs],actual,width=w,color=COLORS['actual'],edgecolor=COLORS['ink'],lw=.55,label='Released',zorder=3)
    for x,a,b in zip(xs,ref,actual):
        ax.text(x-w/2,a+.45,str(a),ha='center',fontsize=7.8)
        ax.text(x+w/2,b+.45,str(b),ha='center',fontsize=7.8)
    ax.axhline(result['required_release'],color=COLORS['threshold'],lw=.85,ls=(0,(3,2)),zorder=1)
    ax.text(2.47,8.1,'Need 6',color=COLORS['threshold'],ha='right',va='bottom',fontsize=7.2)
    ax.set_xticks(list(xs));ax.set_xticklabels([r'Drop $G^0$'+'\nBudget: no',r'Drop $G^0,G^1$'+'\nBudget: yes',r'Drop $H$'+'\nBudget: yes'],fontsize=7.5)
    ax.set_ylabel('Buffer units',fontsize=8.2,labelpad=3)
    ax.set_ylim(0,22);ax.set_xlim(-.55,2.55);ax.set_yticks([0,6,12,18]);ax.tick_params(axis='y',labelsize=7.5,width=.6,length=2.5);ax.tick_params(axis='x',length=0,pad=5)
    for name in ['top','right']:ax.spines[name].set_visible(False)
    for name in ['bottom','left']:ax.spines[name].set_linewidth(.6)
    ax.set_title('(b) Logical removal vs. physical release',loc='left',fontsize=8.3,pad=22 if not wide else 23)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.13),ncol=2,frameon=False,fontsize=7.5,handlelength=1.4,columnspacing=1.25,borderaxespad=0)

def render(root,name,width_mm,result):
    wide=width_mm>100
    if wide:
        fig=plt.figure(figsize=(width_mm/25.4,69/25.4));left=fig.add_axes([.025,.15,.46,.71]);right=fig.add_axes([.565,.25,.41,.51])
    else:
        fig=plt.figure(figsize=(width_mm/25.4,104/25.4));left=fig.add_axes([.06,.54,.90,.39]);right=fig.add_axes([.16,.12,.80,.29])
    ownership(left,wide);releases(right,result,wide)
    fig.text(.5,.018,'Constructed example; all shown views are evictable.',ha='center',fontsize=6.8,color='#4A515A')
    files=[]
    for ext in ['pdf','svg','png']:
        p=root/(name+'.'+ext);fig.savefig(p,dpi=300,facecolor='white');files.append(p)
    plt.close(fig);return files

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'dejavuserif'})
    m=json.loads(a.input.read_text());assert m['evidence_kind']=='CONSTRUCTED_COUNTEREXAMPLE_NOT_MEASUREMENT';r=compute(m)
    files=render(a.output,'first_page_problem_85mm',85,r)+render(a.output,'first_page_problem_175mm',175,r)
    (a.output/'example_calculation.json').write_text(json.dumps(r,indent=2)+'\n')
    receipt=dict(status='PASS_DETERMINISTIC_COUNTEREXAMPLE_RENDER',performance_measurement=False,gpu_executed=False,claims='Exact set-ownership arithmetic for the stated constructed model only; no empirical superiority claim.',input=bind(a.input),generator=bind(Path(__file__)),widths_mm=[85,175],files={x.name:bind(x) for x in files})
    (a.output/'figure_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'files':len(files),'actual_releases':[x['released'] for x in r['candidates']]}))
if __name__=='__main__':main()
