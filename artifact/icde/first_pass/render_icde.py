"""Static execution panels: paper E1 single GPU; paper E2(a) eight GPUs."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GRAPHS=('US','EP','LJ','OR','IN','UK','TW');ALGORITHMS=('bfs','sssp','wcc','pr')

def render(report_path,output):
 report=json.loads(Path(report_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
  'legend.fontsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,
  'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'savefig.pad_inches':.035})
 colors={'eggpu':'#2166AC','best':'#7B8794'};width=.34;records=[];figures=[];captions=[]
 for count,stem,paper_panel in ((1,'e1_single_gpu','E1'),(8,'e2_eight_gpu_comparison','E2(a)')):
  fig,axes=plt.subplots(1,4,figsize=(7.0,2.15),layout='constrained')
  for ci,algorithm in enumerate(ALGORITHMS):
   ax=axes[ci];known=[]
   for si,system in enumerate(('eggpu','best')):
    values=[]
    for graph in GRAPHS:
     group=[r for r in report['cells'] if r['dataset']==graph and r['algorithm']==algorithm and r['gpu_count']==count]
     if system=='eggpu':row=next((r for r in group if r['system']=='eggpu'),None)
     else:
      available=[r for r in group if r['role']=='external' and r['status']=='AVAILABLE'];row=min(available,key=lambda r:r['ready_seconds']) if available else None
     value=row.get('ready_seconds') if row and row['status']=='AVAILABLE' else None
     values.append(value if value is not None else np.nan)
     if value is not None:
      known.append(value);records.append(dict(figure=stem,paper_panel=paper_panel,gpu_count=count,dataset=graph,algorithm=algorithm,system=row['label'],metric='ready_seconds',value=value,raw_receipt=row['raw_receipt']))
    ax.bar(np.arange(len(GRAPHS))+(si-.5)*width,values,width=width,color=colors[system],edgecolor='white',linewidth=.35,label='EGGPU' if system=='eggpu' else 'Best observed',zorder=3)
   ax.set_title('('+chr(97+ci)+') '+algorithm.upper(),pad=4)
   ax.set_xticks(range(len(GRAPHS)),GRAPHS);ax.tick_params(axis='both',length=2.5,pad=2)
   ax.set_xlim(-.6,len(GRAPHS)-.4);ax.grid(axis='y',color='#E3E6EA',linewidth=.45,zorder=0)
   if known:ax.set_yscale('log');ax.set_ylim(min(known)*.55,max(known)*1.9)
  fig.supylabel('Time(s)',fontsize=8);fig.supxlabel('Graph',fontsize=8)
  handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False,columnspacing=2,handlelength=1.5)
  for suffix in ('pdf','svg','png'):fig.savefig(out/(stem+'.'+suffix),dpi=220)
  plt.close(fig)
  own=sum(r['system']=='eggpu' and r['gpu_count']==count and r['status']=='AVAILABLE' for r in report['cells'])
  figures.append(dict(stem=stem,paper_panel=paper_panel,gpu_count=count,eggpu_available=own,eggpu_required=28))
  captions.append(f'{paper_panel}：{count} 卡静态图分析执行时间（n=1）。灰色为各格最快已测对手，具体身份与覆盖见补充表。BFS/SSSP 为三个固定源的等权汇总。空位表示当前无可用观测；EGGPU 覆盖为 {own}/28。')
 # Replace only obsolete derived views; every measurement and the previous run remain.
 for suffix in ('pdf','svg','png'):
  obsolete=out/('e1_first_pass_best_observed.'+suffix)
  if obsolete.exists():obsolete.unlink()
 (out/'FIGURE_CAPTION_CN.txt').write_text('\n\n'.join(captions)+'\n\n论文 E2(a) 使用原始 E1-P8 证据，raw ID 不变；E2(b) 另用固定后端的 1/2/4/8 卡扩展证据。首次任务时间保留在补充表。\n')
 (out/'figure_data.json').write_text(json.dumps({'source':str(Path(report_path).resolve()),'figures':figures,'records':records,'n_independent':1,'error_bars':False,'missing_rendered_as_zero':False,'metric':'ready_seconds','first_times_retained_in':'all_systems.csv'},ensure_ascii=False,indent=2)+'\n')
 return len(records)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(render(a.report,a.output))
