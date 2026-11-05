#!/usr/bin/env python3
"""Plan-only by default: fixed C2graph-P E1 tiny, target and ten-repeat jobs."""
from pathlib import Path
import argparse,json,subprocess
import c2p_worker as worker


def plan(args):
    proofs={}
    for item in json.loads(args.proofs.read_text()) if args.proofs else []:
        path=Path(item);row=json.loads(path.read_text())
        if row.get('status')!='PASS_QUALIFICATION' or row.get('variant')!='c2graph_p':raise ValueError('Invalid P qualification')
        key=row['phase'],row['dataset']
        if key in proofs:raise ValueError('Choose one explicit qualification, never latest/fastest')
        proofs[key]=path
    jobs=[]
    for repeat in range(10) if args.phase=='formal' and args.repeat is None else [args.repeat or 0]:
        for name in args.datasets.split(','):
            if name not in worker.common.DATASETS+worker.common.TINY_DATASETS or (name in worker.common.TINY_DATASETS)!=(args.phase=='tiny'):
                raise ValueError('Dataset outside the frozen phase scope')
            meta=json.loads((args.input_root/'canonical'/name/'canonical.json').read_text())
            tiny='TINY_D' if meta['directed'] else 'TINY_U';label=f'{args.phase}_{name}_sssp_c2graph_p_r{repeat:02d}'
            output=args.output/label
            command=[str(args.python),'-B',str(Path(__file__).with_name('c2p_worker.py')),'--execute-gpu',
                '--dataset',name,'--phase',args.phase,'--repeat',str(repeat),'--output',str(output)]
            for key in ('root','input_root','author_build','setup','reference_cache','lock_file','gpu_uuid','cpus','timeout','min_host_free_gib'):
                command+=['--'+key.replace('_','-'),str(getattr(args,key))]
            for phase,graph in ([('tiny',tiny)] if args.phase=='target' else [('tiny',tiny),('target',name)] if args.phase=='formal' else []):
                if (phase,graph) not in proofs:raise ValueError('Missing exact P '+phase+' qualification for '+graph)
                command+=['--'+phase+'-proof',str(proofs[phase,graph])]
            jobs.append({'id':label,'command':command,'receipt':str(output/'result.json'),
                         'expected_statuses':['PASS' if args.phase=='formal' else 'PASS_QUALIFICATION'],
                         'timeout':12*args.timeout+180})
    return jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-gpu',action='store_true')
    p.add_argument('--datasets',default='TINY_D,TINY_U');p.add_argument('--phase',choices=('tiny','target','formal'),default='tiny')
    p.add_argument('--repeat',type=int);p.add_argument('--proofs',type=Path)
    for key in ('root','input-root','author-build','setup','reference-cache','lock-file','output','python'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--cpus',required=True)
    p.add_argument('--timeout',type=int,default=3600);p.add_argument('--min-host-free-gib',type=float,default=8.)
    args=p.parse_args();jobs=plan(args)
    if not args.execute_gpu:print(json.dumps({'gpu_executed':False,'jobs':jobs},indent=2));return
    args.output.mkdir(parents=True,exist_ok=False);receipt={'status':'RUNNING','jobs':jobs,'completed':[]}
    try:
        for index,job in enumerate(jobs):
            with (args.output/f'job_{index:04d}.log').open('wb') as log:code=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT).returncode
            path=Path(job['receipt']);result=json.loads(path.read_text()) if path.exists() else {}
            receipt['completed'].append({'id':job['id'],'returncode':code,'status':result.get('status')})
            worker.common.save(args.output/'controller.json',receipt)
            if code or result.get('status') not in job['expected_statuses']:raise RuntimeError('C2graph-P job failed: '+job['id'])
        receipt['status']='PASS_SELECTED_CELLS'
    finally:worker.common.save(args.output/'controller.json',receipt)


if __name__=='__main__':main()
