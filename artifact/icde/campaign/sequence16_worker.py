"""Independent candidate E1 continuous [BFS,SSSP,WCC,PR] x 4 worker.

Fresh process and one graph/session per measured 16-request workflow. Every
full host result survives later requests and close; audits happen afterward.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,fcntl,json,os,sys,time
import sg_worker as shared
import sequence16_adapter as adapter

HERE=Path(__file__).resolve().parent
PROTOCOL='e1-continuous-16-public-csr-v1'
SOURCE_INDICES=(0,1,2,0)


def trace(graph):
    return [dict(index=r*4+i,round=r,algorithm=algorithm,
                 source=graph['sources'][SOURCE_INDICES[r]] if algorithm in ('bfs','sssp') else None,
                 source_index=SOURCE_INDICES[r] if algorithm in ('bfs','sssp') else None)
            for r in range(4) for i,algorithm in enumerate(shared.ALGORITHMS)]


def owned_output(value,previous,n):
    import numpy as np
    result=np.asarray(value)
    if result.shape!=(n,):raise ValueError('Every request must return the complete canonical host domain')
    # Borrowed/writeable API results are copied immediately, inside this query,
    # so a later API call cannot overwrite their backing storage. Known public
    # immutable owners can be retained directly; pairwise aliasing is rejected.
    copy=(not result.flags.owndata or not result.flags.c_contiguous or result.flags.writeable
          or any(np.shares_memory(result,x) for x in previous))
    if copy:result=np.array(result,copy=True,order='C')
    result.flags.writeable=False
    return result,copy


def execute_sequence(args,graph,backend_factory=adapter.build):
    """Exactly one measured interval; no oracle, hashing or disk IO inside it."""
    requests=trace(graph);outputs=[];calls=[];backend=None
    start=time.perf_counter_ns();constructed=start;failure=None;contract={}
    try:
        backend=backend_factory(args,graph);constructed=time.perf_counter_ns()
        for request in requests:
            begin=time.perf_counter_ns();raw=backend.query(request)
            values,copied=owned_output(raw,outputs,graph['n']);end=time.perf_counter_ns()
            outputs.append(values);calls.append(dict(request,begin_ns=begin,end_ns=end,
                seconds=(end-begin)/1e9,ownership_copy=copied))
        complete=time.perf_counter_ns()
        contract=backend.contract
    except BaseException as error:
        complete=time.perf_counter_ns();failure=type(error).__name__+': '+str(error)
        if backend is not None:contract=backend.contract
    finally:
        if backend is not None:
            try:backend.close()
            except BaseException as error:failure=(failure or '')+'; close: '+type(error).__name__+': '+str(error)
        closed=time.perf_counter_ns()
    passed=failure is None and len(calls)==16
    return outputs,dict(calls=calls,backend=contract,requests=len(calls),outputs_survive_close=failure is None,
        execution_complete=passed,execution_error=failure,
        workflow_start_ns=start,workflow_complete_ns=complete,
        T_workflow_seconds=(complete-start)/1e9 if passed else None,attempt_seconds=(complete-start)/1e9,
        T_first_result_seconds=(calls[0]['end_ns']-start)/1e9 if calls else None,
        construction_seconds=(constructed-start)/1e9,cleanup_seconds=(closed-complete)/1e9,
        workflow_through_close_seconds=(closed-start)/1e9,
        boundary='normal backend imports and common host CSR through all 16 complete owned immutable host outputs',
        continuous_interval=True,sum_of_query_times_used_as_total=False,warmup_requests=0)



def configure(args):
    args.cpus=[int(x) for x in args.cpu_set.split(',')]
    if len(args.cpus)!=16 or len(set(args.cpus))!=16 or not set(args.cpus)<=set(os.sched_getaffinity(0)):
        raise ValueError('Exactly 16 available CPU cores required')
    if not args.gpu_uuid.startswith('GPU-'):raise ValueError('Explicit GPU UUID required')
    if (args.dataset in shared.TINY_DATASETS)!=(args.phase=='tiny'):raise ValueError('Separate tiny and real input phases')
    if args.repeat<0 or (args.phase=='formal' and args.repeat not in range(10)):raise ValueError('Formal repeats are 0..9')
    if args.timeout<=0 or args.budget_mib<=0 or args.min_host_free_gib<=0:raise ValueError('Invalid resource/time budget')
    return args


def epoch(args):
    proxy=SimpleNamespace(**vars(args));proxy.release=args.source
    data,_=shared.source_epoch(proxy)
    data.update(protocol=PROTOCOL,workflow=['bfs','sssp','wcc','pr']*4,
        source_indices=list(SOURCE_INDICES),input_format='public owned CSR',timeout=args.timeout,
        budget_mib=args.budget_mib,minimum_host_free_gib=args.min_host_free_gib,formal_repetitions=10,
        warmups=0,code_cache_policy='persistent compiled-code cache; fresh process/device graph state per complete 16-request workflow')
    data['preloaded_libraries']={name:shared.binding(name) for name in os.environ.get('LD_PRELOAD','').replace(':',' ').split()}
    data['files'].update(sequence_worker=shared.binding(__file__),sequence_adapter=shared.binding(HERE/'sequence16_adapter.py'))
    return data,shared.digest_json(data)


def validate_proof(path,kind,args,digest,graph):
    proof=json.loads(Path(path).read_text())
    if any(proof.get(k)!=v for k,v in dict(status='PASS_QUALIFICATION',phase=kind,system=args.system,
        protocol=PROTOCOL,epoch_sha256=digest,quality_pass=True,request_count=16).items()):
        raise ValueError('Continuous-workflow qualification does not match this source/system/configuration')
    if kind=='tiny':
        if proof.get('dataset') not in shared.TINY_DATASETS or proof.get('directed')!=graph['directed']:raise ValueError('Tiny directedness mismatch')
    elif proof.get('input_sha256')!=graph['input_sha256']:raise ValueError('Target input/source trace changed')
    for field in ('epoch','workflow'):
        if shared.binding(proof[field]['path'])!=proof[field]:raise ValueError('Workflow proof binding changed')
    if shared.digest_json(json.loads(Path(proof['epoch']['path']).read_text()))!=digest:raise ValueError('Proof epoch differs')
    result=json.loads(Path(proof['workflow']['path']).read_text())
    if result.get('status')!='PASS' or not result.get('continuous_interval') or len(result['calls'])!=16:
        raise ValueError('Proof is not one complete measured workflow')
    for row in result['calls']:
        if not row['quality']['pass'] or shared.binding(row['output']['path'])!=row['output']:raise ValueError('Incomplete request output proof')
    return shared.binding(path)


def inherited_lease(args):
    if args.lease_fd is None:raise ValueError('Source child requires inherited global GPU lease')
    held=os.fstat(args.lease_fd);actual=args.lock_file.stat()
    if (held.st_dev,held.st_ino)!=(actual.st_dev,actual.st_ino):raise ValueError('Inherited FD is not global lock')
    with args.lock_file.open('a') as probe:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        raise ValueError('Parent does not hold exclusive global lease')


def loaded_shared_libraries():
    """Bind actual mapped runtime libraries after timing, before CPU oracle imports."""
    files={};unresolved=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6 or not fields[5].startswith('/') or '.so' not in Path(fields[5]).name:continue
        path=Path(fields[5])
        if path.is_file():
            resolved=str(path.resolve())
            if resolved not in files:files[resolved]=shared.content_binding(path)
        elif fields[5] not in unresolved:unresolved.append(fields[5])
    return dict(files=files,unresolved_mappings=unresolved)


def source_child(args):
    import numpy as np
    inherited_lease(args);configure(args);folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=False)
    record=dict(status='RUNNING',protocol=PROTOCOL,system=args.system,quality_pass=False)
    try:
        graph=shared.load_input(args.input_root/'canonical'/args.dataset);_,digest=epoch(args)
        if digest!=args.epoch_sha256:raise ValueError('Child source/DSO/runtime differs from parent')
        record.update(input_sha256=graph['input_sha256'],epoch_sha256=digest,
            host_ready_checksum=[int(graph['offsets'].sum(dtype=np.int64)),int(graph['indices'].sum(dtype=np.int64)),float(graph['weights'].sum(dtype=np.float64))])
        shared.save(folder/'stage.json',dict(active=True,label='continuous_16_requests',started_monotonic=time.monotonic()))
        with shared.complete_timeout(args.timeout):outputs,measurement=execute_sequence(args,graph)
        shared.save(folder/'stage.json',dict(active=False,label='continuous_16_requests'))
        record.update(measurement)
        record['loaded_shared_libraries']=loaded_shared_libraries()
        for row,values in zip(record['calls'],outputs):
            expected,reference=shared.reference_cached(graph,row['algorithm'],row['source'],args.reference_cache)
            row.update(output=shared.persist_output(folder,values),reference=reference,
                       quality=shared.audit_output(values,expected,graph,row['algorithm']))
        record['quality_pass']=bool(record['execution_complete'] and len(record['calls'])==16 and all(x['quality']['pass'] for x in record['calls']))
        if not record['quality_pass']:raise ValueError('Continuous workflow complete-output quality failed')
        record['status']='PASS'
    except BaseException as error:record.update(status='FAIL',error=type(error).__name__+': '+str(error));raise
    finally:shared.save(folder/'result.json',record)


def child_command(args,folder,digest,lease_fd):
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--execute-gpu','--child',
             '--lease-fd',str(lease_fd),'--epoch-sha256',digest]
    for action in parser()._actions:
        if action.dest in ('help','execute_gpu','child','make_tiny','lease_fd','epoch_sha256','output'):continue
        value=getattr(args,action.dest,None)
        if value is not None:command.extend([action.option_strings[0],str(value)])
    return command+['--output',str(folder)]


def job(args):
    if not args.execute_gpu:raise ValueError('GPU execution requires explicit --execute-gpu by owner')
    configure(args);args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    record=dict(status='RUNNING',protocol=PROTOCOL,system=args.system,dataset=args.dataset,
                phase=args.phase,repeat=args.repeat,request_count=16,quality_pass=False)
    with args.lock_file.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.sched_setaffinity(0,args.cpus)
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
            EGGPU_ARENA_LIBRARY_DIR=str(args.arena_library_dir.resolve()))
        try:
            graph=shared.load_input(args.input_root/'canonical'/args.dataset)
            record['admission']=shared.resource_check(args,graph);data,digest=epoch(args)
            shared.save(args.output/'epoch.json',data);record.update(epoch=shared.binding(args.output/'epoch.json'),
                epoch_sha256=digest,input_sha256=graph['input_sha256'],directed=graph['directed'],trace=trace(graph),qualification={})
            if args.phase in ('target','formal'):
                if not args.tiny_proof:raise ValueError('Continuous-workflow tiny proof required')
                record['qualification']['tiny']=validate_proof(args.tiny_proof,'tiny',args,digest,graph)
            if args.phase=='formal':
                if not args.target_proof:raise ValueError('Same-input workflow target proof required')
                record['qualification']['target']=validate_proof(args.target_proof,'target',args,digest,graph)
            folder=args.output/'workflow'
            with (args.output/'workflow.log').open('x') as log:
                code=shared.run_child(child_command(args,folder,digest,lock.fileno()),log,lock.fileno(),folder,args.timeout)
            if code:raise RuntimeError('Fresh workflow process failed: '+str(code))
            result=json.loads((folder/'result.json').read_text())
            if result.get('status')!='PASS' or not result.get('quality_pass'):raise ValueError('Workflow quality failed')
            record.update(status='PASS_FORMAL' if args.phase=='formal' else 'PASS_QUALIFICATION',quality_pass=True,
                workflow=shared.binding(folder/'result.json'),T_workflow_seconds=result['T_workflow_seconds'])
        except BaseException as error:record.update(status='FAIL',error=type(error).__name__+': '+str(error));raise
        finally:shared.save(args.output/'result.json',record)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-gpu',action='store_true')
    p.add_argument('--make-tiny',type=Path);p.add_argument('--child',action='store_true');p.add_argument('--lease-fd',type=int);p.add_argument('--epoch-sha256')
    p.add_argument('--system',choices=['eggpu','cugraph']);p.add_argument('--dataset',choices=shared.DATASETS+shared.TINY_DATASETS)
    p.add_argument('--phase',choices=['tiny','target','formal'],default='tiny');p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--gpu-uuid');p.add_argument('--cpu-set')
    for key in ['input-root','source','native-dso','arena-library-dir','setup','lock-file','reference-cache','output','tiny-proof','target-proof']:
        p.add_argument('--'+key,type=Path)
    p.add_argument('--budget-mib',type=int,default=22000);p.add_argument('--min-host-free-gib',type=float,default=8)
    p.add_argument('--timeout',type=float,default=1800)
    return p


def main():
    p=parser();args=p.parse_args()
    if args.make_tiny:shared.make_tiny(args.make_tiny);return
    for key in ['system','dataset','gpu_uuid','cpu_set','input_root','source','native_dso','arena_library_dir','setup','lock_file','reference_cache','output']:
        if getattr(args,key) is None:p.error('--'+key.replace('_','-')+' required')
    if args.child:
        if not args.execute_gpu or not args.epoch_sha256:p.error('Child needs explicit execution and parent epoch')
        source_child(args)
    else:job(args)

if __name__=='__main__':main()
