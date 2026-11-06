"""Static execution panels: paper E1 single GPU; paper E2(a) eight GPUs."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GRAPHS=('US','EP','LJ','OR','IN','UK','TW');ALGORITHMS=('bfs','sssp','wcc','pr')

def configure_style():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
  'legend.fontsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,
  'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'savefig.pad_inches':.035})

def render(report_path,output):
 report=json.loads(Path(report_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
 configure_style()
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
 remaining=out/'remaining'/'remaining_results.json'
 if remaining.is_file():render_remaining(remaining,out/'remaining')
 return len(records)

def render_remaining(report_path,out):
 report=json.loads(Path(report_path).read_text());rows=[r for r in report['cells'] if r['status']=='AVAILABLE'];figures=[];data=[];captions=[]
 configure_style();palette=('#2166AC','#D98C2B','#6F7A85','#40927E','#8B6BA4')
 def save(fig,stem,caption,records):
  if not records:plt.close(fig);return
  fig.supylabel('Time(s)',fontsize=8)
  for suffix in ('pdf','svg','png'):fig.savefig(out/(stem+'.'+suffix),dpi=220)
  plt.close(fig);figures.append(stem);captions.append(stem+'：'+caption+' 每格 n=1，无误差条；未完成、失败或待复核格留空。');data.extend(records)
 def decorate(ax,title,xs,labels):
  ax.set_title(title,pad=4);ax.set_xticks(xs,labels);ax.tick_params(length=2.5,pad=2);ax.grid(axis='y',color='#E3E6EA',linewidth=.45,zorder=0)
 def record(stem,row,metric):return dict(figure=stem,dataset=row['dataset'],system=row['label'],gpu_count=row['gpu_count'],algorithm=row.get('algorithm'),batch=row.get('batch'),ablate=row.get('ablate'),coverage_backend=row.get('coverage_backend'),metric=metric,value=row[metric],raw_receipt=row['raw_receipt'])
 # Five fixed graphs and one fixed native endpoint for all four GPU counts.
 for metric,suffix in (('ready_seconds','ready'),('first_seconds','first_supplement')):
  stem='e2_scaling_'+suffix;fig,axes=plt.subplots(1,4,figsize=(7,2.15),layout='constrained');records=[]
  for ax,algorithm in zip(axes,ALGORITHMS):
   known=[]
   for i,graph in enumerate(('US','OR','IN','UK','TW')):
    matching={r['gpu_count']:r for r in rows if r['experiment']=='E2' and r['algorithm']==algorithm and r['dataset']==graph}
    values=[matching[p][metric] if p in matching else np.nan for p in (1,2,4,8)]
    if matching:
     ax.plot(range(4),values,'o-',color=palette[i],markersize=3,linewidth=.9,label=graph)
     for row in matching.values():records.append(record(stem,row,metric));known.append(row[metric])
   decorate(ax,algorithm.upper(),range(4),('1','2','4','8'));ax.set_xlabel('GPUs')
   if known:ax.set_yscale('log')
   else:ax.set_yticks([])
  handles,labels=[],[]
  for ax in axes:
   h,l=ax.get_legend_handles_labels()
   for hi,li in zip(h,l):
    if li not in labels:handles.append(hi);labels.append(li)
  if handles:fig.legend(handles,labels,loc='outside upper center',ncol=5,frameon=False)
  save(fig,stem,'固定原生静态协议的 1/2/4/8 卡完整结果时间；P8 从同源码主计划按明确续跑映射复用。BFS/SSSP 为三个固定源的等权均值。'+('首结果时间含准备。' if suffix.startswith('first') else 'ready 保留图状态。'),records)
 for metric,suffix in (('ready_seconds','query_batch_ready'),('first_seconds','first_supplement')):
  stem='e4s_'+suffix;fig,axes=plt.subplots(1,4,figsize=(7,2.15),layout='constrained');records=[]
  for ax,graph in zip(axes,('US','OR','IN','UK')):
   known=[]
   for i,system in enumerate(('eggpu','kgraph_sparse' if graph=='US' else 'kgraph_dense','c2graph_pm')):
    matching={r['batch']:r for r in rows if r['experiment']=='E4S' and r['dataset']==graph and r['system']==system}
    values=[matching[q][metric] if q in matching else np.nan for q in (32,128,512)]
    if matching:
     ax.plot(range(3),values,'o-',color=palette[i],markersize=3,linewidth=.9,label=next(iter(matching.values()))['label'])
     for row in matching.values():records.append(record(stem,row,metric));known.append(row[metric])
   decorate(ax,graph,range(3),('32','128','512'));ax.set_xlabel('Queries')
   if known:ax.set_yscale('log')
   else:ax.set_yticks([])
  handles,labels=[],[]
  for ax in axes:
   h,l=ax.get_legend_handles_labels()
   for hi,li in zip(h,l):
    if li not in labels:handles.append(hi);labels.append(li)
  if handles:fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
  save(fig,stem,'冻结请求序列上的完整 SSSP 批次。'+('补充：从共同 canonical CSR 到全部持有输出，含系统准备。' if metric=='first_seconds' else '主图：各系统正常图准备后到全部请求完整持有输出；不是第二次热批次，按需准备仍在计时内。')+' KGraph 使用固定 16 线程 CPU；EGGPU/C²graph-PM 使用一张 GPU。',records)
 for metric,suffix in (('ready_seconds','ready'),('first_seconds','first_supplement')):
  timing_caption='主图采用完整热身后保留图状态的独立 ready 工作流实测区间；每次重新产生全部候选与输出。' if metric=='ready_seconds' else '补充首次完整工作流，包含弱投影与后端准备。'
  stem='e6_systems_'+suffix;fig,axes=plt.subplots(1,3,figsize=(7,2.15),layout='constrained');records=[]
  systems=('eggpu','igraph','cugraph_igraph','cugraph_composite');short=('EGGPU','igraph','cuG+ig','cuG+CuPy')
  for ax,graph in zip(axes,('EP','LJ','OR')):
   matching={r['system']:r for r in rows if r['experiment']=='E6' and r['dataset']==graph and r['ablate']=='none' and r['gpu_count'] in (0,1)}
   for i,system in enumerate(systems):
    if system in matching:row=matching[system];ax.bar(i,row[metric],color=palette[i],width=.6,zorder=3);records.append(record(stem,row,metric))
   decorate(ax,graph,range(4),short)
   if matching:ax.set_yscale('log')
   else:ax.set_yticks([])
  save(fig,stem,timing_caption+'完整依赖工作流：结构指标、真实候选筛选、候选 closeness、胜者选择、BFS 与 WCC。EGGPU/cuGraph 组合为 P1；igraph 为 CPU 对手，混合实现保留实际回退。',records)
  stem='e6_scaling_'+suffix;fig,axes=plt.subplots(1,3,figsize=(7,2.15),layout='constrained');records=[]
  for ax,graph in zip(axes,('EP','LJ','OR')):
   matching={r['gpu_count']:r for r in rows if r['experiment']=='E6' and r['dataset']==graph and r['system']=='eggpu' and r['ablate']=='none'}
   if matching:
    ax.plot(range(4),[matching[p][metric] if p in matching else np.nan for p in (1,2,4,8)],'o-',color=palette[0],markersize=3,linewidth=.9)
    records.extend(record(stem,row,metric) for row in matching.values());ax.set_yscale('log')
   else:ax.set_yticks([])
   decorate(ax,graph,range(4),('1','2','4','8'));ax.set_xlabel('GPUs')
  save(fig,stem,timing_caption+'EGGPU 完整工作流扩展：图复制到各卡，结构指标与候选 closeness 分工，BFS/WCC 使用逻辑 GPU0；不表示分区容量扩展。',records)
  stem='e6_coverage_2x2_'+suffix;fig,axes=plt.subplots(2,3,figsize=(7,4.1),layout='constrained');records=[]
  for ri,backend in enumerate(('igraph','cugraph_composite')):
   for ax,graph in zip(axes[ri],('EP','LJ','OR')):
    matching={r['ablate']:r for r in rows if r['experiment']=='E6' and r['dataset']==graph and r['system']=='eggpu' and r['gpu_count']==1 and (r['coverage_backend']==backend or r['ablate']=='none')}
    for i,arm in enumerate(('none','structural','closeness','structural_closeness')):
     if arm in matching:row=matching[arm];ax.bar(i,row[metric],color=palette[i],width=.6,zorder=3);records.append(record(stem,row,metric))
    decorate(ax,graph+' / '+('CPU fallback' if ri==0 else 'GPU fallback'),range(4),('N/N','F/N','N/F','F/F'))
    if matching:ax.set_yscale('log')
    else:ax.set_yticks([])
  save(fig,stem,timing_caption+'结构组与 closeness 的 2×2 覆盖消融，N 为 EGGPU，F 为指定外部回退；两组共用原生 P1 控制，BFS/WCC 始终保持 EGGPU。',records)
 (out/'FIGURE_CAPTION_CN.txt').write_text('\n\n'.join(captions)+'\n')
 (out/'figure_data.json').write_text(json.dumps({'figures':figures,'records':data,'primary_metric':'ready_seconds','supplementary_metric':'first_seconds','n_independent':1,'error_bars':False,'missing_rendered_as_zero':False},ensure_ascii=False,indent=2)+'\n')
 html_path=out/'remaining_results.html'
 if html_path.exists():
  with html_path.open('a') as stream:
   for stem in figures:stream.write('<p><a href="'+stem+'.pdf">'+stem+' PDF</a> · <a href="'+stem+'.svg">SVG</a></p><img style="max-width:100%" src="'+stem+'.png">')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(render(a.report,a.output))
