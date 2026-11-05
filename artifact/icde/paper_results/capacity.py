"""Source-bound feasibility evidence; never convert a failure to a timing win.

Selection is explicit. Classification cites an exact exception/log substring.
Only observed device-allocation errors are OOM; generic GPU errors stay Failed.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import hashlib
import normalize as n
import render as r

STATUS = {'Complete','OOM','Timeout','Index limit','Failed','Pending','Infeasible'}
OOM_MARKERS = ('cudaErrorMemoryAllocation', 'CUDA_ERROR_OUT_OF_MEMORY',
               'out_of_memory: CUDA error', 'RMM_ALLOC', 'cudaMalloc failed: out of memory')
COLORS = {'Complete':'#BED4AA','OOM':'#E8B2AB','Timeout':'#E7D1A7','Index limit':'#D4C8E0',
          'Failed':'#BFBFBF','Pending':'#F2F2F2','Infeasible':'#D7D7D7'}
CELL_LABEL = {'Complete':'OK','OOM':'OOM','Timeout':'TO','Index limit':'IDX',
              'Failed':'FAIL','Pending':'—','Infeasible':'N/A'}


def classify(status, raw, text='', needle=''):
    n.need(status in STATUS,'Unknown feasibility state')
    if status=='Pending':
        n.need(raw is None,'Pending must not hide an existing selected attempt')
        return
    n.need(raw is not None,'Observed capacity state needs a bound execution receipt')
    if status=='Complete':
        n.need(raw.get('status') in ('PASS','PASS_QUALIFICATION','PASS_STATE_COMPLETE_WORKFLOW','PASS_STATE_GPU_QUALIFICATION')
               and raw.get('quality_pass') is True,'Complete needs a finished correct task')
    else:
        n.need(raw.get('quality_pass') is not True,'Failure cannot relabel a complete correct result')
        n.need(bool(needle) and needle in text,'Classification must quote retained evidence exactly')
        if status=='OOM':
            n.need(any(x in needle for x in OOM_MARKERS),'Generic GPU, host allocation, or index errors are not device OOM')


def sampled_peak(e, binding, receipt_binding=None):
    if binding is None:return None,'not collected'
    record=e.read(binding)
    n.need(record.get('kind')=='device_memory_samples','Wrong memory evidence type')
    n.need(receipt_binding is not None and record.get('execution_receipt') == receipt_binding,
           'Memory sampling must bind the exact selected execution receipt')
    e.verify(record['execution_receipt'])
    stream=e.verify(record['samples_file']).open() if record.get('samples_file') else None
    samples=(json.loads(line) for line in stream) if stream else iter(record['samples'])
    peak=None;devices=None;count=0
    try:
        for sample in samples:
            values=sample.get('used_bytes');n.need(isinstance(values,dict) and values,'No observed memory samples')
            if devices is None:devices=set(values)
            n.need(set(values)==devices,'Observed device UUID set changed')
            n.need(all(type(v) is int and v>=0 for v in values.values()),'Invalid memory sample')
            peak=max(values.values()) if peak is None else max(peak,*values.values());count+=1
    finally:
        if stream:stream.close()
    n.need(count>0,'No observed memory samples')
    if record.get('samples_file'):
        n.need(record.get('formal_measurement') is False and record.get('performance_eligible') is False,
               'Sampled diagnostic must be excluded from formal timing')
        n.need(record.get('sample_count')==count and record.get('busiest_device_peak_bytes')==peak,
               'Sampled summary differs from retained frames')
    return peak,'complete task' if record.get('covers_complete_task') is True else 'partial sampling'


def normalize(selection_path):
    selection_path=Path(selection_path);selection=json.loads(selection_path.read_text())
    n.need(selection.get('schema')=='eggpu.capacity.selection/v1','Wrong explicit capacity selection')
    e=n.core.Evidence(selection_path.parent,selection.get('path_remap',[]))
    freeze=e.read(selection['source_freeze']);n.need(n.digest(freeze['source_files'])==freeze['source_tree_sha256'],'Invalid release freeze')
    domains=n.core.input_domains(e,selection['input_registry']);rows=[];seen=set()
    for cell in selection['cells']:
        key=(cell['system'],cell['algorithm'],cell['gpu_count'],tuple(cell['datasets']),cell.get('budget_fraction'),cell.get('arm'))
        n.need(key not in seen,'Duplicate capacity cell; no fastest/latest picking');seen.add(key)
        n.need(set(cell['datasets'])<=set(n.DATASETS),'Capacity must retain the fixed graph suite')
        n.need(cell['scope'] in ('single_task','multi_graph_state'),'Unknown capacity boundary')
        if cell['scope']=='single_task':n.need(len(cell['datasets'])==1,'Single-task capacity requires one graph')
        raw=e.read(cell['receipt']) if cell.get('receipt') else None
        if raw:
            epoch=n.bind_identity(e,cell,raw,freeze)
            if cell['scope']=='single_task':
                n.need(raw.get('input_sha256')==domains[cell['datasets'][0]]['sha256'],'Capacity input binding differs')
            else:
                n.need(cell['adapter']=='state','Resident-state capacity needs the actual state runner')
                n.validate_state_input(e,raw,domains)
                n.need(epoch['experiment']=='E4','Resident-state pressure requires the frozen E4 workflow')
                n.need(cell.get('arm')==epoch['arm'] and cell.get('budget_fraction')==epoch['beta'],
                       'Resident-state policy or budget fraction differs from execution')
                if raw.get('budget'):
                    n.need(cell.get('per_device_budget_bytes')==raw['budget']['per_device'],
                           'Reported per-device budget differs from measured pilot budget')
        annotation=cell.get('classification',{})
        text=e.verify(annotation['evidence']).read_text(errors='replace') if annotation.get('evidence') else ''
        classify(cell['status'],raw,text,annotation.get('needle',''))
        peak,coverage=sampled_peak(e,cell.get('memory_samples'),cell.get('receipt'))
        # An allocation request is diagnostic, never used as a peak measurement.
        allocation=annotation.get('failed_allocation_bytes')
        if allocation is not None:n.need(cell['status']=='OOM' and isinstance(allocation,int) and allocation>0,'Allocation diagnostic differs')
        canonical=sum(sum(domains[g]['identity']['files'][k]['bytes'] for k in ('offsets','indices','weights')) for g in cell['datasets'])
        rows.append(dict(system=cell['system'],algorithm=cell['algorithm'],gpus=cell['gpu_count'],datasets=','.join(cell['datasets']),
            scope=cell['scope'],arm=cell.get('arm'),status=cell['status'],stage=annotation.get('stage','—'),phase=raw.get('phase') if raw else None,
            canonical_csr_bytes=canonical,budget_fraction=cell.get('budget_fraction'),per_device_budget_bytes=cell.get('per_device_budget_bytes'),
            sampled_peak_vram_bytes=peak,sampling_coverage=coverage,failed_allocation_bytes=allocation,
            receipt=cell.get('receipt'),classification_evidence=annotation.get('evidence'),timing_mean=None,speedup=None))
    if not selection.get('synthetic_fixture'):
        groups={(x['system'],x['algorithm'],x['gpus']) for x in rows if x['scope']=='single_task'}
        for group in groups:
            n.need({x['datasets'] for x in rows if x['scope']=='single_task' and
                (x['system'],x['algorithm'],x['gpus'])==group}==set(n.DATASETS),
                'A capacity row must explicitly include all seven graphs, including Pending states')
    return dict(schema='eggpu.capacity.normalized/v1',source_tree_sha256=freeze['source_tree_sha256'],
        selection=dict(path=str(selection_path.resolve()),bytes=selection_path.stat().st_size,
            sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest()),rows=rows,synthetic_fixture=bool(selection.get('synthetic_fixture')),
        graph_order=sorted(domains,key=lambda g:domains[g]['identity'].get('m',0)),
        policy='Feasibility only. No qualification timing means, zero failure times, infinite speedups, or invented peak VRAM.')


def render(result,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);rows=result['rows']
    (out/'capacity.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=['system','algorithm','gpus','datasets','scope','arm','status','stage','phase','canonical_csr_bytes',
        'budget_fraction','per_device_budget_bytes','sampled_peak_vram_bytes','sampling_coverage','failed_allocation_bytes']
    with (out/'capacity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:row.get(k) for k in fields})
    plt=r.pyplot()
    for p in sorted({x['gpus'] for x in rows if x['scope']=='single_task'}):
        selected=[x for x in rows if x['gpus']==p and x['scope']=='single_task']
        graphs=result.get('graph_order') or sorted({x['datasets'] for x in selected},key=lambda g:next(x['canonical_csr_bytes'] for x in selected if x['datasets']==g))
        methods=sorted({(x['system'],x['algorithm']) for x in selected})
        fig,ax=plt.subplots(figsize=(min(6.89,max(3.35,.65*len(graphs)+1.8)),max(1.5,.3*len(methods)+.5)),constrained_layout=True)
        from matplotlib.patches import Rectangle
        for i,method in enumerate(methods):
            for j,g in enumerate(graphs):
                matches=[x for x in selected if (x['system'],x['algorithm'])==method and x['datasets']==g]
                status=matches[0]['status'] if matches else 'Pending'
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=COLORS[status],edgecolor='white',linewidth=1,
                    hatch='///' if status=='OOM' else None))
                ax.text(j,i,CELL_LABEL[status],ha='center',va='center',fontsize=7)
        ax.set_xlim(-.5,len(graphs)-.5);ax.set_ylim(len(methods)-.5,-.5)
        ax.set_aspect('equal',adjustable='box')
        ax.set_xticks(range(len(graphs)),graphs);ax.set_yticks(range(len(methods)),[s+' / '+f.upper() for s,f in methods])
        ax.set_xlabel('Graph');ax.tick_params(length=0)
        if result.get('synthetic_fixture'):fig.text(.5,.5,'SYNTHETIC TEST DATA — NOT A RESULT',ha='center',color='#B2182B',rotation=15)
        for extension in ('pdf','svg'):fig.savefig(out/f'E1_capacity_p{p}.{extension}',bbox_inches='tight')
        plt.close(fig)
    for p in sorted({x['gpus'] for x in rows if x['scope']=='multi_graph_state'}):
        selected=[x for x in rows if x['gpus']==p and x['scope']=='multi_graph_state']
        budgets=sorted({x['budget_fraction'] for x in selected})
        methods=sorted({(x['system'],x['arm']) for x in selected})
        fig,ax=plt.subplots(figsize=(3.35,max(1.5,.36*len(methods)+.6)),constrained_layout=True)
        from matplotlib.patches import Rectangle
        for i,method in enumerate(methods):
            for j,budget in enumerate(budgets):
                matches=[x for x in selected if (x['system'],x['arm'])==method and x['budget_fraction']==budget]
                n.need(len(matches)<=1,'Ambiguous multi-graph panel; one fixed graph pool and trace required')
                status=matches[0]['status'] if matches else 'Pending'
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=COLORS[status],edgecolor='white',linewidth=1,
                    hatch='///' if status=='OOM' else None))
                ax.text(j,i,CELL_LABEL[status],ha='center',va='center',fontsize=7)
        ax.set_xlim(-.5,len(budgets)-.5);ax.set_ylim(len(methods)-.5,-.5)
        ax.set_xticks(range(len(budgets)),[f'{x:g}' for x in budgets])
        ax.set_yticks(range(len(methods)),[r.ARM_NAMES.get(arm,system) for system,arm in methods])
        ax.set_xlabel('Budget fraction');ax.tick_params(length=0)
        if result.get('synthetic_fixture'):fig.text(.5,.5,'SYNTHETIC TEST DATA — NOT A RESULT',ha='center',color='#B2182B',rotation=15,fontsize=8)
        for extension in ('pdf','svg'):fig.savefig(out/f'E4_residency_capacity_p{p}.{extension}',bbox_inches='tight')
        plt.close(fig)
    (out/'caption.md').write_text('Execution feasibility on the fixed graph suite, ordered by normalized stored arcs. OK: completed with validated output; OOM: observed device-memory allocation failure; TO: timeout; IDX: index limit; FAIL: other observed failure; —: pending; N/A: infeasible configuration. Failed stages and exact source-bound logs are listed in capacity.csv/capacity.json. Qualification failures establish feasibility boundaries and do not provide formal time averages. Sampled peak VRAM is the maximum observed single-device usage, with sampling coverage stated separately. Missing peaks remain unavailable; failed allocation size is not peak usage. Multi-graph resident-state pressure concerns the managed retention budget, not a changed physical VRAM limit, and is a separate boundary from fitting one task.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(normalize(a.selection),a.output)
