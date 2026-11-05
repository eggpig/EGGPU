"""Vector figures/CSV/captions from complete, explicitly selected formal evidence."""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import tempfile
import platform
import importlib.metadata
import normalize as n
import analysis

PALETTE = {'blue':'#356A9A','green':'#6B8E67','orange':'#C88646','red':'#B76A64',
           'purple':'#8C79A6','teal':'#76B7B2','gray':'#8A8A8A'}
GRAPH_COLORS = dict(zip(n.DATASETS,PALETTE.values()))
ARM_COLORS = {'D0C0':'#D3D3D3','D1C0':'#B8D3E7','D0C1':'#BED4AA','D1C1':PALETTE['blue']}
ARM_NAMES = {'D0C0':'Full + LRU','D1C0':'D + LRU','D0C1':'Full + C','D1C1':'D + C'}
ARM_COLORS.update(Rebuild='#8A8A8A',D1Cnone='#D3D3D3',D1CviewLRU='#C88646',D1Clogical='#8C79A6')
ARM_NAMES.update(Rebuild='Rebuild',D1Cnone='No retention',D1CviewLRU='View LRU',D1Clogical='Logical score')
MARKERS = ('o','s','^','D','v','P','X')
METRICS = {'first':('Time (s)',1.),'ready':('Time (s)',1.),'workflow':('Workflow time (s)',1.),
    'maintenance_h2d_bytes':('Weight H2D (GiB)',1/(1<<30)),
    'cow_d2d_bytes':('COW D2D (GiB)',1/(1<<30)),
    'shared_array_new_bytes':('New shared arrays (GiB)',1/(1<<30)),
    'freed_bytes':('Freed memory (GiB)',1/(1<<30))}
METRICS['commit_rebuild_freed_bytes']=('Rebuild release (GiB)',1/(1<<30))


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csv_write(path,fields,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({key:row.get(key,'') for key in fields})


def write_tables(result,out):
    repetitions=[];cells=[]
    for row in result['records']:
        for metric,value in (row['metrics'].items() if row['valid'] else [('',None)]):
            repetitions.append(dict(cell=row['cell'],repeat=row['repeat'],status=row['status'],valid=row['valid'],selected=row.get('selected',True),
                exclusion_reason=row.get('exclusion_reason',''),
                metric=metric,value=value,output_evidence=row.get('output_evidence',''),source_path=row['binding']['path']))
    for row in result['cells']:
        for metric,stats in (row['metrics'].items() if row['metrics'] else [('',{})]):
            cells.append(dict(cell=row['cell'],experiment=row['experiment'],metric=metric,n=stats.get('n',0),
                mean=stats.get('mean'),ci_low=stats.get('ci95',[None,None])[0],ci_high=stats.get('ci95',[None,None])[1],
                valid_repetitions=row['valid_repetitions'],observed_repetitions=row['observed_repetitions'],
                complete_formal=row['complete_formal'],terminal_status=row['terminal_status'],
                dimensions=json.dumps(row['definition'],sort_keys=True)))
    csv_write(out/'repetitions.csv',['cell','repeat','status','selected','valid','metric','value','output_evidence','exclusion_reason','source_path'],repetitions)
    csv_write(out/'cells.csv',['cell','experiment','metric','n','mean','ci_low','ci_high','valid_repetitions','observed_repetitions','complete_formal','terminal_status','dimensions'],cells)
    csv_write(out/'coverage.csv',['cell','experiment','expected_repetitions','observed_repetitions','valid_repetitions','complete_formal','resolved','terminal_status'],result['cells'])


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.linewidth':.65,
        'lines.linewidth':1.35,'lines.markersize':3.5,'grid.linewidth':.4,'grid.alpha':.35,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})
    return plt


def stat(row,metric,scale=1.):
    if not row['complete_formal'] or metric not in row['metrics']:return None
    s=row['metrics'][metric]
    return s['mean']*scale,[max(0.,(s['mean']-s['ci95'][0])*scale),max(0.,(s['ci95'][1]-s['mean'])*scale)]


def figure_save(fig,out,name,caption,captions,result,selected,interval='95% Student-t intervals for independent repeat means'):
    plt=pyplot()
    valid=sum(x['complete_formal'] for x in selected)
    if result.get('synthetic_fixture'):
        fig.text(.5,.5,'SYNTHETIC TEST DATA — NOT A RESULT',ha='center',va='center',rotation=20,
                 fontsize=18,color='#B2182B',alpha=.6)
        caption='SYNTHETIC TEST DATA — NOT AN EXPERIMENTAL RESULT. '+caption
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',metadata={'Creator':'EGGPU source-bound paper renderer','CreationDate':None})
    fig.savefig(out/(name+'.svg'),bbox_inches='tight',metadata={'Creator':'EGGPU source-bound paper renderer','Date':None})
    plt.close(fig)
    captions.append((name,caption+f' Complete formal cells: {valid}/{len(selected)}. Means use ten independent complete repetitions; error bars show {interval}. Failed or unresolved cells receive no timing value.'))


def panels(groups,metric,xkey,serieskey,out,name,caption,result,captions,xlabel,
           color_map=None,series_names=None,logx=False,logy=False,columns=None):
    plt=pyplot();keys=list(groups);columns=columns or (2 if len(keys)==4 else min(3,max(1,len(keys))))
    rows=math.ceil(len(keys)/columns)
    fig,axes=plt.subplots(rows,columns,figsize=(6.89,2.25*rows),squeeze=False,constrained_layout=True)
    label,scale=METRICS[metric];selected=[];handles=[];labels=[]
    for ax,title in zip(axes.flat,keys):
        subset=groups[title];selected.extend(subset)
        series=sorted({x['definition'][serieskey] for x in subset},key=str)
        all_x=sorted({x['definition'][xkey] for x in subset})
        failures=[]
        for index,variant in enumerate(series):
            points=sorted((x for x in subset if x['definition'][serieskey]==variant),key=lambda x:x['definition'][xkey])
            xs=[];ys=[];lo=[];hi=[]
            for row in points:
                s=stat(row,metric,scale)
                if s is None:
                    failures.append(str(variant)+': '+str(row['terminal_status'] or 'INCOMPLETE'));continue
                xs.append(row['definition'][xkey]);ys.append(s[0]);lo.append(s[1][0]);hi.append(s[1][1])
            color=(color_map or {}).get(variant,list(PALETTE.values())[index%len(PALETTE)])
            legend=(series_names or {}).get(variant,str(variant))
            if xs:
                h=ax.errorbar(xs,ys,yerr=[lo,hi],marker=MARKERS[index%len(MARKERS)],color=color,capsize=2,label=legend)
                if legend not in labels:handles.append(h);labels.append(legend)
        ax.set_title(str(title));ax.set_xlabel(xlabel);ax.set_ylabel(label);ax.grid(axis='y')
        if logx:ax.set_xscale('log')
        if xkey=='gpu_count':ax.set_xscale('log',base=2)
        if logy and all(stat(x,metric,scale) is None or x['metrics'][metric]['ci95'][0]>0 for x in subset):
            from matplotlib.ticker import LogLocator, NullLocator, FuncFormatter
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1.,2.,5.),numticks=6))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:.3g}'))
            ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks(all_x,labels=[f'{x:g}' if isinstance(x,(int,float)) else str(x) for x in all_x])
        if failures:ax.text(.02,.98,'\n'.join(sorted(set(failures))),transform=ax.transAxes,va='top',fontsize=6,color='#9C2D21')
    for ax in list(axes.flat)[len(keys):]:ax.set_visible(False)
    if handles:fig.legend(handles,labels,loc='outside upper center',ncol=min(5,len(labels)),frameon=False)
    figure_save(fig,out,name,caption,captions,result,selected)


def bar_panel(groups,metric,out,name,caption,result,captions,key='arm',labels=None,colors=None):
    plt=pyplot();titles=list(groups);columns=min(3,len(titles));rows=math.ceil(len(titles)/columns)
    fig,axes=plt.subplots(rows,columns,figsize=(6.89,2.5*rows),squeeze=False,constrained_layout=True)
    ylabel,scale=METRICS[metric];selected=[]
    for ax,title in zip(axes.flat,titles):
        subset=groups[title];selected.extend(subset)
        for index,row in enumerate(subset):
            identity=row['definition'][key];s=stat(row,metric,scale)
            if s:
                ax.bar(index,s[0],color=(colors or {}).get(identity,PALETTE['blue']),edgecolor='black',linewidth=.4,
                       yerr=[[s[1][0]],[s[1][1]]],capsize=2,width=.68)
            else:ax.text(index,.02,row['terminal_status'] or 'INCOMPLETE',transform=ax.get_xaxis_transform(),ha='center',rotation=90,fontsize=6,color='#9C2D21')
        names=[(labels or {}).get(x['definition'][key],str(x['definition'][key])) for x in subset]
        ax.set_xticks(range(len(subset)),names,rotation=25,ha='right');ax.set_title(title)
        ax.set_ylabel(ylabel);ax.grid(axis='y');ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(titles):]:ax.set_visible(False)
    figure_save(fig,out,name,caption,captions,result,selected)


def render_e2(result,out,captions,inference):
    selected=[x for x in result['cells'] if x['experiment']=='E2' and x['definition']['role']=='eggpu']
    if not selected:return
    groups={f.upper():[x for x in selected if x['definition']['algorithm']==f] for f in n.core.ALGORITHMS}
    for metric in ('first','ready'):
        panels(groups,metric,'gpu_count','dataset',out,'Fig4_E2_'+metric+'_time',
            'Strong scaling of a fixed native backend on US, OR, IN, UK and TW. '+
            ('The first interval includes actual partitioning, required weak projection, upload and full host output.' if metric=='first' else
             'Ready time is measured separately after warmup, including complete host output.')+
            ' Native P1 is the scaling reference; the optimized single-GPU arena is not substituted. Each BFS/SSSP repetition is the equal-weight mean of the three frozen sources.',
            result,captions,'GPUs',color_map=GRAPH_COLORS,logy=True,columns=4)
        plt=pyplot();fig,axes=plt.subplots(1,4,figsize=(6.89,2.35),constrained_layout=True)
        for ax,function in zip(axes,n.core.ALGORITHMS):
            for index,item in enumerate(x for x in inference['scaling'] if x['algorithm']==function and x['metric']==metric and x['complete']):
                ys=[item['points'][str(p)]['speedup']['mean'] for p in (1,2,4,8)]
                lo=[ys[j]-item['points'][str(p)]['speedup']['ci95'][0] for j,p in enumerate((1,2,4,8))]
                hi=[item['points'][str(p)]['speedup']['ci95'][1]-ys[j] for j,p in enumerate((1,2,4,8))]
                ax.errorbar([1,2,4,8],ys,yerr=[lo,hi],marker=MARKERS[index%7],color=GRAPH_COLORS[item['dataset']],capsize=2,label=item['dataset'])
            ax.plot([1,8],[1,8],color='#777777',linestyle='--',linewidth=.8,label='Ideal')
            ax.set_title(function.upper());ax.set_xlabel('GPUs');ax.set_ylabel('Speedup');ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8],[1,2,4,8]);ax.grid(axis='y')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=6,frameon=False)
        figure_save(fig,out,'Fig4_E2_'+metric+'_speedup',
            'Speedup is native P1 mean time divided by the corresponding P-GPU mean time. P1/P1 is exactly one. Non-monotonic measured curves are retained.',captions,result,selected,
            interval='95% independent-group log delta ratio intervals')


def render_state(result,out,captions,inference):
    rows=[x for x in result['cells'] if x['experiment'] in ('E3','E4','E3R','E5','E3_PR_CONTROL')]
    e3=[x for x in rows if x['experiment']=='E3']
    options=sorted({(x['definition']['gpu_count'],x['definition']['old_readers'],x['definition']['locality']) for x in e3})
    for p,old,locality in options:
        subset=[x for x in e3 if (x['definition']['gpu_count'],x['definition']['old_readers'],x['definition']['locality'])==(p,old,locality)]
        for positive in (True,False):
            picked=[dict(x,definition=dict(x['definition'],rho_percent=100*x['definition']['rho'])) for x in subset if (x['definition']['rho']>0)==positive]
            if not picked:continue
            groups={name:[x for x in picked if x['definition']['datasets']==[name]] for name in sorted({x['definition']['datasets'][0] for x in picked})}
            for metric in ('workflow','maintenance_h2d_bytes','cow_d2d_bytes'):
                title=f'Fig5_E3_p{p}_{old}_{locality}_{"positive" if positive else "zero"}_{metric}'
                caption=f'Effect of update density at {p} GPUs, reader policy {old}, locality {locality}. Each trace includes 20 commits and 80 complete queries; the shared four-query initialization is excluded from workflow time. Rebuild reconstructs canonical host partitions and device state with the catalog disabled; D0+C and D1+C retain the same C policy. '
                caption += 'Positive density uses a logarithmic x-axis; the zero-update control is a separate companion panel. '
                if metric=='maintenance_h2d_bytes':caption+='Weight H2D sums measured full-weight and packet transfers, without adding the overlapping dense-lease subset; it is not total GPU transfer. '
                if positive:panels(groups,metric,'rho_percent','arm',out,title,caption,result,captions,'Updated edges (%)',ARM_COLORS,ARM_NAMES,logx=True)
                else:bar_panel(groups,metric,out,title,caption,result,captions,labels=ARM_NAMES,colors=ARM_COLORS)
    e4=[x for x in rows if x['experiment']=='E4']
    for order in sorted({x['definition']['order'] for x in e4}):
        selected=[x for x in e4 if x['definition']['order']==order]
        groups={f'{p} GPU'+('s' if p!=1 else ''):[x for x in selected if x['definition']['gpu_count']==p] for p in sorted({x['definition']['gpu_count'] for x in selected})}
        for metric in ('workflow','shared_array_new_bytes','freed_bytes'):
            caption=f'No retention, individual-view LRU, closure-LRU, logical scoring and C-Retain process the fixed US/OR/IN/UK pool in {order} order with D fixed. '
            caption+=('The once control has4 visits/16 queries and is not pooled with128-query traces. ' if order=='once' else 'Each trace has32 visits/128 complete queries. ')
            caption+='Per-device budgets are frozen from the source-matched physical-byte pilot: floor(Bmin+beta*(Ball-Bmin)). '
            if metric=='shared_array_new_bytes':caption+='New shared-array bytes are measured allocation evidence, not an estimate of all reload/H2D traffic. '
            if order=='once':bar_panel(groups,metric,out,'Fig6_E4_once_'+metric,caption,result,captions,labels=ARM_NAMES,colors=ARM_COLORS)
            else:panels(groups,metric,'beta','arm',out,'Fig6_E4_'+order+'_'+metric,caption,result,captions,'Budget fraction',ARM_COLORS,ARM_NAMES)
    readonly=[x for x in rows if x['experiment']=='E3R']
    for p in sorted({x['definition']['gpu_count'] for x in readonly}):
        picked=[]
        for row in readonly:
            if row['definition']['gpu_count']!=p:continue
            d=dict(row['definition']);d['display_arm']=ARM_NAMES[d['arm']]+' / '+d['order'];picked.append(dict(row,definition=d))
        groups={g:[x for x in picked if x['definition']['datasets']==[g]] for g in sorted({x['definition']['datasets'][0] for x in picked})}
        bar_panel(groups,'workflow',out,f'Fig5S_E3R_p{p}',
            f'Read-only order control at {p} GPUs. Grouped and interleaved schedules contain the same 16 requests and no commits. The two D routes use the same C policy; these observations are not pooled with the update experiment.',
            result,captions,key='display_arm')
    e5=[x for x in rows if x['experiment']=='E5']
    for rho,beta in sorted({(x['definition']['rho'],x['definition']['beta']) for x in e5}):
        subset=[x for x in e5 if (x['definition']['rho'],x['definition']['beta'])==(rho,beta)]
        groups={f'{p} GPU'+('s' if p!=1 else ''):sorted((x for x in subset if x['definition']['gpu_count']==p),key=lambda x:('D0C0','D1C0','D0C1','D1C1').index(x['definition']['arm'])) for p in sorted({x['definition']['gpu_count'] for x in subset})}
        for metric in ('workflow','maintenance_h2d_bytes'):
            bar_panel(groups,metric,out,f'Fig8_E5_rho{rho:g}_beta{beta:g}_{metric}',
                f'Joint D/C effects for the IN/UK/OR workload:24 commits and96 complete queries, rho={rho:g}, beta={beta:g}. D0 uses full weight maintenance with reuse; C0 uses closure-LRU. A faster joint configuration is reported separately from the factorial interaction; no sign is imposed on either result.',result,captions,labels=ARM_NAMES,colors=ARM_COLORS)
    interactions=[x for x in inference['joint'] if x.get('interaction')]
    if interactions:
        plt=pyplot();fig,ax=plt.subplots(figsize=(6.89,2.6),constrained_layout=True)
        for index,item in enumerate(interactions):
            s=item['interaction'];ax.errorbar(index,s['mean'],yerr=[[s['mean']-s['ci95'][0]],[s['ci95'][1]-s['mean']]],
                fmt='o',color=PALETTE['green'],capsize=3)
        ax.axhline(0,color='#666666',linewidth=.8,linestyle='--')
        ax.set_xticks(range(len(interactions)),[f"P{x['gpu_count']}, ρ={x['rho']:g}, β={x['beta']:g}" for x in interactions],rotation=20,ha='right')
        ax.set_ylabel('Interaction (s)');ax.grid(axis='y')
        figure_save(fig,out,'Fig8S_E5_interaction',
            'Factorial time interaction I=T11−T10−T01+T00, computed only from ten verified four-arm execution blocks. A confidence interval below zero supports superadditive time saving; lower joint mean alone does not establish it. Configurations without verified blocks remain explicitly unavailable in comparisons.json.',
            captions,result,e5,interval='95% Student-t intervals over ten explicitly matched block contrasts')
    controls=[x for x in rows if x['experiment']=='E3_PR_CONTROL']
    if controls:
        groups={g:sorted((x for x in controls if x['definition']['dataset']==g),key=lambda x:x['definition']['arm']) for g in ('US','IN','OR')}
        bar_panel(groups,'workflow',out,'Fig5S_unweighted_PR_control',
            'Independent weight-invariance control on the original unweighted topology: one base PR query and20 weight commits each followed by a newly executed PR query. All21 complete vectors are checked against the independent unit-weight oracle and base output. Only reader policy none is applicable. This control does not alter the80-query E3 main trace.',result,captions,labels=ARM_NAMES,colors=ARM_COLORS)


def render_e4s(result,out,captions):
    rows=[x for x in result['cells'] if x['experiment']=='E4S']
    if not rows:return
    displayed=[]
    for row in rows:
        d=dict(row['definition']);d['display_system']='KGraph (CPU)' if d['variant'].startswith('kgraph') else 'C²graph-PM' if d['variant']=='c2graph_pm' else 'EGGPU'
        displayed.append(dict(row,definition=d))
    groups={g:[x for x in displayed if x['definition']['dataset']==g] for g in ('US','OR','IN','UK')}
    for metric in ('first','ready'):
        panels(groups,metric,'batch','display_system',out,'Fig7_E4S_'+metric,
            'Complete same-graph SSSP batches with 32/128/512 distinct frozen sources; 128 is the primary point. Initial partitioning, pivots, compression and memoization belong to first-complete time; prepared query-batch time is separate. KGraph uses disclosed CPU resources and is not an E1 same-GPU win. Every complete distance vector is restored from a pinned lossless archive and checked byte-for-byte against the measured output. Formal archive and maintenance gates must both pass; recovery occurs outside the unchanged measured interval.',result,captions,'Queries',{'EGGPU':PALETTE['blue'],'C²graph-PM':PALETTE['orange'],'KGraph (CPU)':PALETTE['green']})


def render_e6(result,out,captions):
    import sys
    from e6_panels import draw
    return draw(result,out,captions,sys.modules[__name__])


def report(result,inference,out):
    lines=['# EGGPU实验结果报告','',f"源码：`{result['source_tree_sha256']}`。正式完整cell：{result['formal_cells']}/{result['expected_cells']}。所有原始重复、失败与覆盖见CSV。",'',
        'E2的均值下降、统计区间、资源效率分别判断，不将理想线写成实测趋势。']
    if result.get('synthetic_fixture'):lines.insert(0,'**合成测试数据；不是实验结果，不可用于论文性能结论。**\n')
    complete=[x for x in inference['scaling'] if x['complete']]
    if complete:lines+=['',f"E2：{sum(x['mean_monotone_nonincreasing'] for x in complete)}/{len(complete)} 条（图×函数×first/ready）均值曲线在1/2/4/8卡各步非增。绝对时间、速度比和相邻差值区间保留在comparisons.json；未通过或不完整曲线不进入该计数。"]
    for ex in ('E3','E4','E3R','E5','E4S','E6','E3_PR_CONTROL'):
        cells=[x for x in result['cells'] if x['experiment']==ex]
        if cells:lines+=['',f"{ex}：{sum(x['complete_formal'] for x in cells)}/{len(cells)} 个正式完整cell；{sum(x['terminal_status'] is not None for x in cells)} 个显式终止。每个成功cell有10个独立完整重复，qualification不进入均值。"]
    for item in inference['joint']:
        if not item['complete']:continue
        statement=f"E5 P{item['gpu_count']}、ρ={item['rho']:g}、β={item['beta']:g}：joint均值快于两种单模块配置={item['joint_mean_faster_than_both_single']}。"
        if item['interaction']:
            s=item['interaction'];statement+=f" 配对交互I={s['mean']:.6g}s，95%CI=[{s['ci95'][0]:.6g},{s['ci95'][1]:.6g}]；超加和时间收益成立={item['interaction_established']}。"
        else:statement+=' 配对证据不足，不输出伪配对交互区间。'
        lines+=['',statement]
    for ex in ('E3','E4','E3_PR_CONTROL','E6'):
        contrasts=[x for x in inference['contrasts'] if x['experiment']==ex]
        if not contrasts:continue
        lines+=['',f'## {ex} 预先规定的逐配置对照','',
            '| 对照 → 处理 | 指标 | 对照均值 | 处理均值 | 节省量（95% CI） |',
            '|---|---|---:|---:|---:|']
        for item in contrasts:
            s=item['saved']
            lines.append(f"| {item['control']} → {item['treatment']} | {item['metric']} | {item['control_mean']:.6g} | {item['treatment_mean']:.6g} | {s['mean']:.6g} [{s['ci95'][0]:.6g}, {s['ci95'][1]:.6g}] |")
        lines+=['','节省量为对照减处理；负值如实保留。每项区间为独立组 Welch 区间，未作跨配置多重比较校正。']
    for item in inference.get('e6_factorial',[]):
        if not item['complete']:continue
        for metric,contrast in item['contrasts'].items():
            lines+=['',f"E6 {item['dataset']} / {item['coverage_backend']} / {metric}：双外部−外部SH−外部CL+全native = {contrast['mean']:.6g}s，95%CI [{contrast['ci95'][0]:.6g}, {contrast['ci95'][1]:.6g}]。这是独立组线性对照；不把相同repeat编号当配对，不预设交互符号。"]
    lines+=['','E6正文区分完整依赖任务、igraph 与 cuGraph composite 两套双因素 2×2、两项遍历补充与多卡执行；不将复制数据执行宣称为聚合显存容量。',
        '',*result['limitations'],'','本报告不声称不存在的总体SOTA、单调性或协同效应；所有失败和缺测都有明确状态。']
    (out/'report_CN.md').write_text('\n'.join(lines)+'\n')


def diagnostics(result,out):
    out.mkdir(parents=True,exist_ok=False);write_tables(result,out)
    dump(out/'normalized.json',result)
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(x[k]))+'</td>' for k in ('cell','valid_repetitions','expected_repetitions','terminal_status'))+'</tr>' for x in result['cells'])
    (out/'checks.html').write_text('<!doctype html><meta charset="utf-8"><h1 style="color:#a11">QUALIFICATION / INCOMPLETE — NOT PAPER RESULTS</h1><table><tr><th>Cell</th><th>Valid</th><th>Expected</th><th>Status</th></tr>'+rows+'</table>')


def render(result,out):
    n.need(result['mode']=='formal' and result['paper_ready'],'No paper figure without complete formal coverage/explicit terminal states')
    n.need(all(not x['complete_formal'] or x['valid_repetitions']==10 for x in result['cells']),'No under-repeated paper cell')
    out.mkdir(parents=True,exist_ok=False);write_tables(result,out)
    inference=analysis.analyze(result);dump(out/'comparisons.json',inference);dump(out/'normalized.json',result)
    captions=[]
    render_e2(result,out,captions,inference);render_state(result,out,captions,inference)
    render_e4s(result,out,captions);render_e6(result,out,captions)
    (out/'captions.md').write_text('\n\n'.join('**'+name+'** '+caption for name,caption in captions)+'\n')
    report(result,inference,out)
    files={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.rglob('*') if p.is_file()}
    paths=[*Path(__file__).parent.glob('*.py'),n._path,
           Path(__file__).resolve().parents[1]/'campaign/state_protocol_completion_v1/state_plan.py']
    tools={str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
    dump(out/'artifact_manifest.json',dict(schema='eggpu.paper.artifacts.v1',source_tree_sha256=result['source_tree_sha256'],
        selection=result['selection_binding'],synthetic_fixture=bool(result.get('synthetic_fixture')),files=files,
        rendering_tools=tools,python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','matplotlib')}))


def managed_check(path):
    manifest=json.loads((path/'artifact_manifest.json').read_text())
    n.need(manifest['schema']=='eggpu.paper.artifacts.v1','Refuse to delete unmanaged previous directory')
    expected=set(manifest['files'])|{'artifact_manifest.json'}
    observed={str(x.relative_to(path)) for x in path.rglob('*') if x.is_file()}
    n.need(expected==observed,'Previous output contains unrelated files')
    for relative,item in manifest['files'].items():
        p=path/relative;n.need(p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'Previous artifact was edited; refuse automatic deletion')


def managed_remove(path):
    managed_check(path);shutil.rmtree(path)


def publish(result,destination,scratch):
    n.need(not result.get('synthetic_fixture'),'Synthetic tests can never publish current/previous')
    n.need(result['paper_ready'],'No current publication of incomplete results')
    destination.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='eggpu-paper-',dir=scratch));stage=temp/'bundle'
    try:
        render(result,stage)
        current=destination/'current';previous=destination/'previous'
        if previous.exists():managed_check(previous)
        if current.exists():managed_check(current)
        # Copy to a same-filesystem staging name so promotion itself is atomic.
        incoming=destination/('.incoming-'+stage.parent.name)
        shutil.copytree(stage,incoming)
        if previous.exists():managed_remove(previous)
        if current.exists():current.rename(previous)
        incoming.rename(current)
    finally:shutil.rmtree(temp)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--mode',choices=('formal','qualification'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path)
    a=p.parse_args();result=n.normalize(a.selection,a.mode)
    if result['paper_ready']:
        n.need(a.scratch is not None,'Explicit scratch directory required for publication')
        publish(result,a.output,a.scratch)
        print(json.dumps({'status':'FORMAL_ARTIFACTS_PUBLISHED','current':str(a.output/'current'),'cells':result['formal_cells']}))
    else:
        n.need(not {'paper','current','previous'}&set(a.output.parts),'Qualification/incomplete outputs must remain outside paper/current/previous')
        diagnostics(result,a.output)
        print(json.dumps({'status':'DIAGNOSTICS_ONLY','formal_cells':result['formal_cells'],'expected_cells':result['expected_cells']}))


if __name__=='__main__':main()
