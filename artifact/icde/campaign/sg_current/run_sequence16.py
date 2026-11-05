"""Plan/execute independent continuous 16-request E1 workflows; formal defaults to all ten repetitions."""
from pathlib import Path
from types import SimpleNamespace
import json,subprocess
import sg_worker as shared
import sequence16_worker as worker


def proof_index(paths):
    result={}
    for value in paths:
        path=Path(value).resolve();row=json.loads(path.read_text())
        if row.get('status')!='PASS_QUALIFICATION' or row.get('protocol')!=worker.PROTOCOL:
            raise ValueError('Need explicit continuous-16 qualification, not a single-query proof')
        key=(row['phase'],row['system'],row['dataset'])
        if key in result:raise ValueError('Select exactly one explicit proof per identity')
        result[key]=path
    return result


def plan(args):
    systems=args.systems.split(',');datasets=args.datasets.split(',')
    if len(set(systems))!=len(systems) or any(s not in ('eggpu','cugraph') for s in systems):raise ValueError('Invalid systems')
    if len(set(datasets))!=len(datasets) or any(d not in shared.DATASETS+shared.TINY_DATASETS for d in datasets):raise ValueError('Invalid datasets')
    repeats=range(10) if args.phase=='formal' and args.repeat is None else [args.repeat or 0]
    proofs=proof_index(json.loads(args.proofs.read_text()) if args.proofs else []);jobs=[]
    for repeat in repeats:
        for dataset in datasets:
            meta=json.loads((args.input_root/'canonical'/dataset/'canonical.json').read_text())
            tiny='TINY_D' if meta['directed'] else 'TINY_U'
            for system in systems if repeat%2==0 else systems[::-1]:
                local=SimpleNamespace(**vars(args));local.system=system;local.dataset=dataset;local.repeat=repeat
                name=f'{args.phase}_{dataset}_{system}_sequence16_r{repeat:02d}';local.output=args.output/name
                local.tiny_proof=proofs.get(('tiny',system,tiny));local.target_proof=proofs.get(('target',system,dataset))
                worker.configure(local)
                if args.phase in ('target','formal') and not local.tiny_proof:raise ValueError('Missing continuous-workflow tiny proof')
                if args.phase=='formal' and not local.target_proof:raise ValueError('Missing continuous-workflow target proof')
                command=[str(args.python_eggpu if system=='eggpu' else args.python_cugraph),'-B',str(Path(worker.__file__).resolve()),'--execute-gpu']
                for action in worker.parser()._actions:
                    if action.dest in ('help','execute_gpu','child','make_tiny','lease_fd','epoch_sha256'):continue
                    value=getattr(local,action.dest,None)
                    if value is not None:command.extend([action.option_strings[0],str(value)])
                jobs.append(dict(id=name,system=system,dataset=dataset,repeat=repeat,requests=16,output=str(local.output),command=command))
    return jobs


def main():
    p=worker.parser();p.description=__doc__;p.set_defaults(repeat=None)
    p.add_argument('--systems',default='eggpu,cugraph');p.add_argument('--datasets',default='TINY_D,TINY_U')
    p.add_argument('--proofs',type=Path);p.add_argument('--python-eggpu',type=Path,required=True);p.add_argument('--python-cugraph',type=Path,required=True)
    args=p.parse_args()
    for key in ['gpu_uuid','cpu_set','input_root','source','native_dso','arena_library_dir','setup','lock_file','reference_cache','output']:
        if getattr(args,key) is None:p.error('--'+key.replace('_','-')+' required')
    jobs=plan(args)
    if not args.execute_gpu:print(json.dumps(dict(status='PLAN_ONLY',gpu_executed=False,jobs=jobs),indent=2));return
    args.output.mkdir(parents=True,exist_ok=False);record=dict(status='RUNNING',protocol=worker.PROTOCOL,jobs=jobs,completed=[])
    try:
        for job in jobs:
            with (args.output/(job['id']+'.log')).open('x') as log:
                result=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT)
            path=Path(job['output'])/'result.json'
            row=json.loads(path.read_text()) if path.exists() else {}
            record['completed'].append(dict(job=job['id'],returncode=result.returncode,status=row.get('status','MISSING'),
                                           result=shared.binding(path) if path.exists() else None))
            if result.returncode or row.get('status') not in ('PASS_QUALIFICATION','PASS_FORMAL'):
                raise RuntimeError('Workflow failed; all completed and failed evidence is retained: '+job['id'])
            shared.save(args.output/'controller.json',record)
        record['status']='PASS_REQUESTED_JOBS'
    except BaseException as error:record.update(status='FAIL',error=type(error).__name__+': '+str(error));raise
    finally:shared.save(args.output/'controller.json',record)

if __name__=='__main__':main()
