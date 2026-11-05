#!/usr/bin/env python3
"""Version-explicit serial author baseline controller. Default is CPU plan-only."""
from pathlib import Path
import argparse, json, subprocess, sys
import sg_author_worker as w

VARIANTS={
 'cggraph_legacy_full':('cggraph',('bfs','sssp','wcc','pr')),
 'cggraph_v15':('cggraph',('bfs','sssp')),
 'gunrock_essentials':('gunrock',('bfs','sssp','pr')),
 'gunrock_legacy_cc':('gunrock',('wcc',)),
 'sygraph_author':('sygraph',('bfs','sssp','wcc')),
 'grapin_author':('grapin',('bfs','sssp','wcc','pr')),
 'hytgraph_author':('hytgraph',('bfs','sssp','wcc','pr'))}


def plan(a):
    variants=a.variants.split(',');datasets=a.datasets.split(',');algorithms=a.algorithms.split(',')
    if len(set(variants))!=len(variants) or not set(variants)<=set(VARIANTS):raise ValueError('Unknown/duplicate variant')
    if not set(datasets)<=set(w.common.DATASETS+w.common.TINY_DATASETS):raise ValueError('Dataset outside frozen suite')
    proofs={}
    for item in json.loads(a.proofs.read_text()) if a.proofs else []:
        path=Path(item);row=json.loads(path.read_text())
        if row.get('status')!='PASS_QUALIFICATION':raise ValueError('Proof did not pass')
        key=(row['phase'],row['variant'],row['algorithm'],row['dataset'])
        if key in proofs:raise ValueError('Choose exactly one explicit proof per version/operator/input, not latest/fastest')
        proofs[key]=path
    repeats=list(range(10)) if a.phase=='formal' and a.repeat is None else [a.repeat or 0];jobs=[]
    for repeat in repeats:
        for name in datasets:
            if (name in w.common.TINY_DATASETS)!=(a.phase=='tiny'):raise ValueError('Wrong tiny/real phase')
            meta=json.loads((a.input_root/'canonical'/name/'canonical.json').read_text());tiny='TINY_D' if meta['directed'] else 'TINY_U'
            for variant in variants if repeat%2==0 else list(reversed(variants)):
                system,available=VARIANTS[variant]
                for function in algorithms:
                    if function not in available:continue
                    out=a.output/f'{a.phase}_{name}_{function}_{variant}_r{repeat:02d}'
                    argv=[str(a.python),'-B',str(Path(__file__).with_name('sg_author_worker.py')),'--execute-gpu',
                          '--system',system,'--variant',variant,'--algorithm',function,'--dataset',name,'--phase',a.phase,
                          '--repeat',str(repeat),'--output',str(out)]
                    for key in ('root','input_root','setup','reference_cache','lock_file','gpu_uuid','cpus','timeout','min_host_free_gib','clock_build_root'):
                        argv+=['--'+key.replace('_','-'),str(getattr(a,key))]
                    for phase,graph in ([('tiny',tiny)] if a.phase=='target' else [('tiny',tiny),('target',name)] if a.phase=='formal' else []):
                        key=(phase,variant,function,graph)
                        if key not in proofs:raise ValueError('Missing explicit source/variant qualification: '+str(key))
                        argv+=['--'+phase+'-proof',str(proofs[key])]
                    jobs.append({'variant':variant,'system':system,'dataset':name,'algorithm':function,'repeat':repeat,'output':str(out),'argv':argv})
    if not jobs:raise ValueError('No applicable baseline cells')
    return jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-gpu',action='store_true');p.add_argument('--variants',default=','.join(VARIANTS))
    p.add_argument('--datasets',default='TINY_D,TINY_U');p.add_argument('--algorithms',default='bfs,sssp,wcc,pr')
    p.add_argument('--phase',choices=('tiny','target','formal'),default='tiny');p.add_argument('--repeat',type=int)
    for key in ('root','input-root','setup','reference-cache','lock-file','output','python','clock-build-root'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--proofs',type=Path);p.add_argument('--gpu-uuid',required=True);p.add_argument('--cpus',required=True)
    p.add_argument('--timeout',type=int,default=3600);p.add_argument('--min-host-free-gib',type=float,default=8.)
    a=p.parse_args();jobs=plan(a)
    if not a.execute_gpu:print(json.dumps({'gpu_executed':False,'jobs':jobs},indent=2));return
    a.output.mkdir(parents=True,exist_ok=False);receipt={'status':'RUNNING','controller':w.common.binding(__file__),'jobs':jobs,'completed':[],'whole_suite_claim':False}
    try:
        for i,row in enumerate(jobs):
            receipt['active_job']=i;w.common.save(a.output/'controller.json',receipt)
            with (a.output/f'job_{i:04d}.log').open('wb') as log:code=subprocess.run(row['argv'],stdout=log,stderr=subprocess.STDOUT).returncode
            path=Path(row['output'])/'result.json';result=json.loads(path.read_text()) if path.exists() else {}
            receipt['completed'].append({'index':i,'returncode':code,'status':result.get('status'),
                                         'result':w.common.binding(path) if path.exists() else None})
            if code or result.get('status') not in ('PASS','PASS_QUALIFICATION'):raise RuntimeError('Baseline cell failed; original result retained at '+str(path))
        receipt['status']='PASS_SELECTED_CELLS'
    finally:w.common.save(a.output/'controller.json',receipt)


if __name__=='__main__':main()
