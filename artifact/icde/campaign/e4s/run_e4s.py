#!/usr/bin/env python3
"""Plan-only E4-S controller for fixed distinct-source complete SSSP batches."""
from pathlib import Path
import argparse,json,subprocess
import worker


def plan(args):
    proofs={}
    for item in json.loads(args.proofs.read_text()) if args.proofs else []:
        path=Path(item);row=json.loads(path.read_text())
        if row.get('status')!='PASS_E4S_QUALIFICATION':raise ValueError('Invalid E4-S proof')
        key=(row['phase'],row['variant'],row['dataset'],row['batch'])
        if key in proofs:raise ValueError('One explicit proof per configuration; never latest/fastest selection')
        proofs[key]=path
    jobs=[]
    for repeat in range(10) if args.phase=='formal' and args.repeat is None else [args.repeat or 0]:
        for name in args.datasets.split(','):
            if name not in (*worker.inputs.DATASETS,*worker.common.TINY_DATASETS) or (name in worker.common.TINY_DATASETS)!=(args.phase=='tiny'):
                raise ValueError('Outside the fixed E4-S dataset/phase scope')
            graph=worker.common.load_input(args.input_root/'canonical'/name)
            tiny='TINY_D' if graph['directed'] else 'TINY_U'
            source=args.sources_dir/(name+'.json')
            for batch in map(int,args.batches.split(',')):
                worker.inputs.load_sources(source,graph,batch)
                variants=['eggpu','kgraph_sparse','kgraph_dense','c2graph_pm'] if args.phase=='tiny' else [
                    'eggpu','kgraph_sparse' if name=='US' else 'kgraph_dense','c2graph_pm']
                if repeat%2:variants.reverse()
                for variant in variants:
                    label=f'{args.phase}_{name}_{variant}_q{batch}_r{repeat:02d}'
                    output=args.output/label
                    command=[str(args.python),'-B',str(Path(__file__).with_name('worker.py')),'--execute',
                        '--variant',variant,'--dataset',name,'--phase',args.phase,'--batch',str(batch),
                        '--repeat',str(repeat),'--output',str(output),'--sources',str(source)]
                    for key in ('root','input_root','author_build','release','native_dso','arena_library_dir','reference_module',
                                'reference_cache','setup','lock_file','gpu_uuid','cpus','budget_mib','reference_cache_gib','timeout','audit_timeout','min_host_free_gib'):
                        command+=['--'+key.replace('_','-'),str(getattr(args,key))]
                    for phase,dataset in ([('tiny',tiny)] if args.phase=='target' else [('tiny',tiny),('target',name)] if args.phase=='formal' else []):
                        key=(phase,variant,dataset,batch)
                        if key not in proofs:raise ValueError('Missing explicit E4-S proof: '+str(key))
                        command+=['--'+phase+'-proof',str(proofs[key])]
                    jobs.append({'id':label,'command':command,'receipt':str(output/'result.json'),
                        'expected_statuses':['PASS_E4S_FORMAL' if args.phase=='formal' else 'PASS_E4S_QUALIFICATION'],
                        'timeout':args.timeout+args.audit_timeout+120,'variant':variant,'batch':batch,'dataset':name,
                        'gpu_count':0 if variant.startswith('kgraph') else 1})
    return jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--datasets',default='TINY_D,TINY_U');p.add_argument('--batches',default='32,128,512')
    p.add_argument('--phase',choices=('tiny','target','formal'),default='tiny');p.add_argument('--repeat',type=int)
    for key in ('root','input-root','sources-dir','author-build','release','native-dso','arena-library-dir','reference-module',
                'reference-cache','setup','lock-file','output','python'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--proofs',type=Path);p.add_argument('--gpu-uuid',required=True);p.add_argument('--cpus',required=True)
    p.add_argument('--budget-mib',type=int,default=22000);p.add_argument('--reference-cache-gib',type=float,default=4.)
    p.add_argument('--audit-timeout',type=int,default=86400);p.add_argument('--timeout',type=int,default=3600);p.add_argument('--min-host-free-gib',type=float,default=8.)
    args=p.parse_args();jobs=plan(args)
    if not args.execute:print(json.dumps({'gpu_executed':False,'jobs':jobs},indent=2));return
    args.output.mkdir(parents=True,exist_ok=False);receipt={'status':'RUNNING','jobs':jobs,'completed':[]}
    try:
        for index,job in enumerate(jobs):
            with (args.output/f'job_{index:04d}.log').open('wb') as log:code=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT).returncode
            path=Path(job['receipt']);row=json.loads(path.read_text()) if path.exists() else {}
            receipt['completed'].append({'id':job['id'],'status':row.get('status'),'returncode':code})
            worker.common.save(args.output/'controller.json',receipt)
            if code or row.get('status') not in job['expected_statuses']:raise RuntimeError('E4-S cell failed: '+job['id'])
        receipt['status']='PASS_SELECTED_E4S_CELLS'
    finally:worker.common.save(args.output/'controller.json',receipt)


if __name__=='__main__':main()
