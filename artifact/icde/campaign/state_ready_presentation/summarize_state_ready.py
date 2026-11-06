#!/usr/bin/env python3
"""Read recorded target audits once per changed receipt; expose all 504 cells."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import csv
import hashlib
import html
import json
import math
import os
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'state_main_first_pass'))
from prepare import read,bind

LABELS={'Rebuild':'Rebuild','D0C1':'Reuse+Full','D1C1':'EGGPU',
    'D1Cnone':'No-retention','D1CviewLRU':'view-LRU','D1C0':'closure-LRU','D1Clogical':'logical-size demand',
    'D0C0':'D0C0'}
COLORS={'Rebuild':'#505050','D0C1':'#999999','D1C1':'#2166AC','D1Cnone':'#404040',
    'D1CviewLRU':'#707070','D1C0':'#999999','D1Clogical':'#b8b8b8','D0C0':'#505050'}
STYLES={'Rebuild':('s','--'),'D0C1':('^','-.'),'D1C1':('o','-'),
    'D1Cnone':('s','--'),'D1CviewLRU':('^',':'),'D1C0':('D','-.'),'D1Clogical':('v','--')}


def figure_captions(report):
    rows=report['rows']
    e3=[r for r in rows if r['experiment']=='E3' and r['partitions']==8 and r['locality']=='uniform' and r['old_readers']=='none']
    e4=[r for r in rows if r['experiment']=='E4' and r['partitions']==8 and r['order']=='interleaved']
    e5=[r for r in rows if r['experiment']=='E5' and r['partitions']==8 and r['rho']==1e-4 and r['beta']==.25]
    count=lambda values:sum(r['status']=='COMPLETE_N1' for r in values)
    interaction=next(r for r in report['interactions'] if r['partitions']==8 and r['rho']==1e-4 and r['beta']==.25)
    joint=f"{interaction['interaction_seconds']:.4g} s" if interaction['interaction_seconds'] is not None else 'pending matched four arms'
    note=' Eight GPUs; n=1 per available configuration. Blue: EGGPU; gray: controls. Missing observations are blank. Time is the directly measured continuous workflow after initialization; first-complete time is supplemental, and whole-job wall time is separate.'
    return {
        'e3_maintenance':f'E3: IN and OR; uniform updates without old readers. Available {count(e3)}/36.'+note,
        'e4_retention':f'E4: US / OR / IN / UK; interleaved requests. Available {count(e4)}/25. Budget is the retained-byte fraction defined in the protocol.'+note,
        'e5_joint':f'E5: IN / UK / OR; update fraction 1e-4, budget fraction 0.25. Available {count(e5)}/4. Matched interaction: {joint}. Byte counters show weight-maintenance H2D and copy-on-write D2D, not all traffic.'+note}


def atomic(path,text):
    temporary=path.with_name(path.name+'.partial');temporary.write_text(text);os.replace(temporary,path)


def assess(path,row,selection,freeze,runner):
    raw=read(path)
    result=dict(status=raw.get('status','UNKNOWN'),raw_receipt=bind(path),time_seconds=None,
        maintenance_h2d_bytes=None,cow_d2d_bytes=None,shared_array_new_bytes=None,freed_bytes=None,
        budget=None,paired_block_sha256=None)
    if raw.get('status')!='PASS_STATE_GPU_QUALIFICATION':return result
    try:
        assert raw['phase']=='target' and raw['repeat']==0 and raw['execution_complete'] and raw['quality_pass']
        assert raw['datasets']==row['datasets'] and raw['experiment']==row['experiment'] and raw['arm']==row['arm']
        assert raw['inputs']==row['inputs'] and len(raw['devices'])==row['partitions']
        selected_policy=selection['measurement_policy']
        assert bind(selected_policy['path'])==selected_policy
        assert raw['measurement_policy']['binding']==selected_policy
        assert raw['measurement_policy']['actual']['status']=='PASS_ACTUAL_PR_POLICY'
        config=raw['configuration']
        for key in ('old_readers','beta','order'):assert config[key]==row[key]
        source=raw['source_binding']
        assert source['dso']==selection['native_dso']
        for name,expected in freeze['source_files'].items():
            assert all(source['files'][name][key]==expected[key] for key in ('bytes','sha256'))
        assert set(source['benchmark'])==set(runner['files'])
        for name,record in source['benchmark'].items():
            assert all(record[key]==runner['files'][name][key] for key in ('bytes','sha256'))
        assert raw['continuous_interval'] is True and raw['sum_of_phases_used_as_total'] is False
        value=raw['workflow_complete_host_seconds'];assert math.isfinite(value) and value>0
        first=raw['first_complete_workflow_seconds'];assert math.isfinite(first) and first>0
        queries=[e for e in raw['events'] if e['kind']=='query']
        expected=80 if row['experiment']=='E3' else 16 if row['experiment']=='E3R' or row['order']=='once' else 128 if row['experiment']=='E4' else 96
        assert len(queries)==len(raw['checks'])==expected
        for event,check in zip(queries,raw['checks']):
            assert all(event[k]==v for k,v in check['event'].items())
            assert check['quality']['passed'] and check['survives_session_close']
            assert check['output']['lossless'] and check['output']['format']=='npz:values'
            assert check['output_sha256']==check['output']['vector_sha256']
            quality=check['quality'];assert quality['reference']['independent']
            function=event['function']
            if function in ('bfs','sssp'):assert quality['exact']
            elif function=='wcc':assert quality['partition_equivalent']
            else:
                assert quality['residual_l1']<=1e-6 and quality['mass_error']<=1e-6
                if 'reference_l1_upper_bound' in quality:
                    amendment=raw['audit_amendment'];assert amendment['method']=='state-pr-full-residual-certificate-v1'
                    assert bind(amendment['manifest']['path'])==amendment['manifest']
                    audit=read(amendment['manifest']['path']);reference=quality['reference']
                    assert reference['solver']==amendment['method'] and reference['full_vector'] and reference['power_solve_executed'] is False
                    assert reference['certificate_source']==audit['files']['pr_certificate.py']
                    assert quality['reference_l1_kind']=='certified_upper_bound_not_measured_difference'
                    assert 0<=quality['reference_l1_upper_bound']<=1e-5 and reference['proof']['certificate_pass']
                    assert quality['reference_l1_upper_bound']==reference['proof']['reference_l1_upper_bound']
                else:assert quality['reference_l1']<=1e-5
        totals=defaultdict(int)
        shared_complete=True
        for event in queries:
            metrics=event['details']['metrics']
            for delta in metrics['native_dynamic_patch']['per_device_delta'].values():
                for key in ('weight_full_h2d_bytes','patch_h2d_bytes','cow_d2d_bytes'):totals[key]+=delta.get(key,0)
            if 'shared_array_new_bytes' not in metrics:shared_complete=False
            else:totals['shared_array_new_bytes']+=metrics['shared_array_new_bytes']
            for action in metrics['resource_management']['actions']:totals['freed_bytes']+=action['freed_bytes']
        if row['old_readers']=='per_update':assert len(raw['leases'])==20 and all(x['released'] for x in raw['leases'])
        result.update(status='COMPLETE_N1',time_seconds=value,workflow_seconds=value,first_complete_workflow_seconds=first,
            maintenance_h2d_bytes=totals['weight_full_h2d_bytes']+totals['patch_h2d_bytes'],
            cow_d2d_bytes=totals['cow_d2d_bytes'],shared_array_new_bytes=totals['shared_array_new_bytes'] if shared_complete else None,
            freed_bytes=totals['freed_bytes'],budget=raw.get('budget'),
            paired_block_sha256=raw['first_pass']['paired_block_sha256'],
            audit_method=raw.get('audit_amendment',{}).get('method','original_independent_reference'),measurement_policy=selected_policy['sha256'])
    except (AssertionError,KeyError,TypeError,ValueError) as error:
        result.update(status='RECEIPT_NOT_ADMISSIBLE',reason=type(error).__name__+': '+str(error))
    return result


def export(bundle,output,render=False):
    output.mkdir(parents=True,exist_ok=True)
    coverage=read(bundle/'coverage.json');selection=read(bundle/'selection.json')
    freeze=read(selection['source_freeze']['path']);runner=read(selection['runner_freeze']['path'])
    cachepath=output/'receipt_cache_ready_v1.json';cache=read(cachepath) if cachepath.exists() else {}
    rows=[]
    for cell in coverage['configurations']:
        result=dict(cell,status='PENDING',time_seconds=None)
        path=bundle/'results'/('target_'+cell['id'])/'receipt.json'
        if path.exists():
            stat=path.stat();key=str(path);identity=[stat.st_size,stat.st_mtime_ns]
            if key not in cache or cache[key]['identity']!=identity:
                cache[key]=dict(identity=identity,result=assess(path,cell,selection,freeze,runner))
            result.update(cache[key]['result'])
        rows.append(result)
    blocks=defaultdict(list)
    for row in rows:
        if row['experiment']=='E5':
            key=(row['partitions'],row['rho'],row['beta'])
            blocks[key].append(row)
    interactions=[]
    for key,values in blocks.items():
        record=dict(partitions=key[0],rho=key[1],beta=key[2],status='PENDING_MATCHED_FOUR_ARMS',interaction_seconds=None)
        if len(values)==4 and all(r['status']=='COMPLETE_N1' for r in values):
            hashes={r['paired_block_sha256'] for r in values}
            if len(hashes)==1 and None not in hashes:
                times={r['arm']:r['time_seconds'] for r in values}
                record.update(status='COMPLETE_N1',paired_block_sha256=next(iter(hashes)),
                    interaction_seconds=times['D1C1']-times['D1C0']-times['D0C1']+times['D0C0'])
            else:record['status']='UNMATCHED_BLOCKS'
        interactions.append(record)
    counts={stage:dict(Counter(r['status'] for r in rows if r['priority_stage']==stage)) for stage in coverage['stages']}
    report=dict(schema='state.first.observation.report.v1',observed_unix=time.time(),source_tree_sha256=selection['source_tree_sha256'],
        observations_per_available_cell=1,full_denominator=504,priority_denominator=78,counts=counts,rows=rows,interactions=interactions,
        evidence_policy='Read existing full-vector CPU audit receipts; no GPU runs, no repeated array hashing/decompression, no new numerical claims.',
        primary_timing='workflow_complete_host_seconds: directly measured continuous ready workflow, no phase subtraction',
        supplemental_timing='first_complete_workflow_seconds: initialization plus complete workflow, recorded directly',
        elapsed_total_estimate_seconds=None,estimated_finish=None,
        budget_note='Per-device managed retained bytes: floor(max(S_gd)+beta*(sum(S_gd)-max(S_gd))); not VRAM capacity or peak.',
        byte_note='New shared-array allocation is not all H2D or reload traffic. Weight H2D and COW D2D retain actual counter identities.')
    atomic(cachepath,json.dumps(cache,indent=2)+'\n');atomic(output/'report.json',json.dumps(report,indent=2)+'\n')
    fields=('id','experiment','datasets','partitions','rho','locality','old_readers','arm','beta','order','priority_stage','status',
        'time_seconds','first_complete_workflow_seconds','maintenance_h2d_bytes','cow_d2d_bytes','shared_array_new_bytes','freed_bytes','paired_block_sha256','audit_method','measurement_policy')
    with (output/'all_cells.csv.partial').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    os.replace(output/'all_cells.csv.partial',output/'all_cells.csv')
    table=['<table><tr>'+''.join('<th>'+x+'</th>' for x in ('Configuration','Stage','Status','Time(s)'))+'</tr>']
    for row in rows:
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (row['id'],row['priority_stage'],row['status'],row['time_seconds'] if row['time_seconds'] is not None else '—'))+'</tr>')
    table.append('</table>')
    captions=figure_captions(report)
    atomic(output/'FIGURE_CAPTION.txt','\n\n'.join(captions.values())+'\n')
    images=''.join('<figure><img src="'+name+'.svg" style="max-width:100%"><figcaption>'+html.escape(caption)+'</figcaption></figure>' for name,caption in captions.items()) if render else ''
    atomic(output/'index.html','<!doctype html><meta charset="utf-8"><title>State first observations</title><style>body{font:14px system-ui;margin:32px;color:#203040}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>E3/E4/E5: n=1</h1><p>78 priority targets / 504 frozen targets. Missing and failed cells remain visible. No repeated-run range or significance claim.</p><pre>'+html.escape(json.dumps(counts,indent=2))+'</pre>'+images+''.join(table))
    if render:draw(report,output)
    return {k:v for k,v in report.items() if k not in ('rows','interactions')}


def draw(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
        'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42})
    def finish(fig,name):
        fig.tight_layout()
        for extension in ('svg','pdf','png'):fig.savefig(output/(name+'.'+extension),dpi=180,bbox_inches='tight')
        plt.close(fig)
    rows=report['rows']
    fig,axes=plt.subplots(1,2,figsize=(7,2.4))
    for ax,graph in zip(axes,('IN','OR')):
        selected=[r for r in rows if r['experiment']=='E3' and r['partitions']==8 and r['datasets']==[graph] and r['locality']=='uniform' and r['old_readers']=='none']
        for arm in ('Rebuild','D0C1','D1C1'):
            values=sorted([r for r in selected if r['arm']==arm],key=lambda r:r['rho'])
            marker,line=STYLES[arm]
            ax.plot([r['rho'] for r in values],[r['time_seconds'] if r['status']=='COMPLETE_N1' else float('nan') for r in values],marker=marker,linestyle=line,markersize=3.5,linewidth=1.2,label=LABELS[arm],color=COLORS[arm])
        ax.set_xscale('symlog',linthresh=1e-6);ax.set(xlabel='Updated arc fraction',ylabel='Time(s)',title=graph)
    axes[0].legend();finish(fig,'e3_maintenance')
    fig,ax=plt.subplots(figsize=(3.5,2.4))
    selected=[r for r in rows if r['experiment']=='E4' and r['partitions']==8 and r['order']=='interleaved']
    for arm in ('D1Cnone','D1CviewLRU','D1C0','D1Clogical','D1C1'):
        values=sorted([r for r in selected if r['arm']==arm],key=lambda r:r['beta'])
        marker,line=STYLES[arm]
        ax.plot([r['beta'] for r in values],[r['time_seconds'] if r['status']=='COMPLETE_N1' else float('nan') for r in values],marker=marker,linestyle=line,markersize=3.5,linewidth=1.2,label=LABELS[arm],color=COLORS[arm])
    ax.set(xlabel='Budget fraction',ylabel='Time(s)');ax.set_xticks([0,.25,.5,.75,1]);ax.legend()
    finish(fig,'e4_retention')
    fig,axes=plt.subplots(1,2,figsize=(7,2.4));arms=('D0C0','D1C0','D0C1','D1C1')
    selected={r['arm']:r for r in rows if r['experiment']=='E5' and r['partitions']==8 and r['rho']==1e-4 and r['beta']==.25}
    for i,arm in enumerate(arms):
        row=selected[arm];good=row['status']=='COMPLETE_N1'
        if good:
            axes[0].bar(i,row['time_seconds'],color=COLORS[arm])
            axes[1].bar(i-.17,row['maintenance_h2d_bytes']/(1<<30),width=.34,color=COLORS[arm],edgecolor='#404040',linewidth=.4)
            axes[1].bar(i+.17,row['cow_d2d_bytes']/(1<<30),width=.34,color=COLORS[arm],edgecolor='#404040',linewidth=.4,hatch='///')
    for ax in axes:ax.set_xticks(range(4),('D0C0','D1C0','D0C1','EGGPU'))
    axes[0].set_ylabel('Time(s)');axes[1].set_ylabel('Data transfer (GiB)')
    axes[1].legend(handles=[Patch(facecolor='#999999',edgecolor='#404040',label='Weight H2D'),Patch(facecolor='#999999',edgecolor='#404040',hatch='///',label='COW D2D')])
    finish(fig,'e5_joint')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render',action='store_true');parser.add_argument('--watch',action='store_true');parser.add_argument('--poll-seconds',type=int,default=30)
    args=parser.parse_args()
    assert 1<=args.poll_seconds<=60
    while True:
        print(json.dumps(export(args.bundle,args.output,args.render)),flush=True)
        if not args.watch:break
        time.sleep(args.poll_seconds)
