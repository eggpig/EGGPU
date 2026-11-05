"""cuGraph MG canonical complete-task worker; owner-only GPU execution.

Fresh process per source. T_first is one continuous interval from host CSR
through weak projection, cluster startup, MGGraph and complete owned host output.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,fcntl,importlib.metadata,json,os,platform,subprocess,sys,time
import sg_worker as shared
import native_input
import native_worker
import cugraph_mg_adapter as adapter
import cugraph_mg_platform as platform_probe
from mg_process_observer import wait_native, stop
from cugraph_mg_platform import resolve_distribution

HERE=Path(__file__).resolve().parent
PROTOCOL='cugraph-mg-host-csr-complete-v1'


def configure(args):
    args.uuids=args.gpu_uuids.split(',');args.cpus=[int(x) for x in args.cpu_set.split(',')]
    if args.count not in (2,4,8) or len(args.uuids)!=args.count or len(set(args.uuids))!=args.count:
        raise ValueError('cuGraph MG requires P=2/4/8 distinct UUIDs')
    if any(not x.startswith('GPU-') or x.strip()!=x for x in args.uuids):raise ValueError('Use explicit GPU UUIDs')
    if len(args.cpus)!=16 or len(set(args.cpus))!=16 or not set(args.cpus)<=set(os.sched_getaffinity(0)):
        raise ValueError('Exactly sixteen available CPU cores required')
    if (args.dataset in shared.TINY_DATASETS)!=(args.phase=='tiny'):raise ValueError('Separate tiny qualification and real graph phases')
    if args.repeat<0 or (args.phase=='formal' and args.repeat not in range(10)):raise ValueError('Formal repeats are 0..9')
    return args


def epoch(args):
    packages={}
    for name in (*adapter.PACKAGES,'scipy','cloudpickle'):
        distname,version=resolve_distribution(name);dist=importlib.metadata.distribution(distname);files={}
        for entry in dist.files or []:
            path=Path(dist.locate_file(entry))
            if path.is_file() and ('.so' in path.name or path.suffix in ('.py','.pyx','.pxd') or path.name in ('METADATA','RECORD')):
                files[str(entry)]=shared.content_binding(path)
        packages[distname]=dict(version=version,files=files)
    data=dict(protocol=PROTOCOL,system='cugraph_mg',count=args.count,gpu_uuids=args.uuids,
        affinity=sorted(os.sched_getaffinity(0)),python=shared.binding(sys.executable),python_version=platform.python_version(),
        setup=shared.content_binding(args.setup),packages=packages,quality=shared.QUALITY,
        pr_alpha=.85,pr_tolerance_l1=1e-9,pr_max_iter=10000,comms_p2p=False,
        resources=dict(cpu_count=16,rank_host_memory_bytes=32<<30,rank_device_memory_bytes=20<<30,
            rmm_pool_bytes=256<<20,rmm_max_bytes=16<<30,rss_cap_gib=float(args.rss_cap_gib),
            admission_gpu_budget_mib=args.budget_mib,minimum_host_free_gib=float(args.min_host_free_gib),
            per_call_timeout_seconds=args.timeout_seconds),
        files={name:shared.binding(HERE/name) for name in ['cugraph_mg_worker.py','cugraph_mg_adapter.py',
            'cugraph_mg_platform.py','sg_worker.py','native_input.py','mg_process_observer.py']},
        environment={k:v for k,v in os.environ.items() if k.startswith(('CUDA_','EGGPU_','OMP_','MKL_','OPENBLAS_','NCCL_','UCX_','NUMBA_'))
                     or k in ('LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH','CUPY_CACHE_DIR')},
        observed_gpus=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version',
            '--format=csv,noheader,nounits'],text=True).strip())
    return data,shared.digest_json(data)


def prepared_graph(graph,algorithm,folder):
    """Preserve canonical isolated vertices and graph-wide int32 loader contract."""
    import numpy as np
    result=native_input.original_graph(graph,'canonical')
    if algorithm=='wcc':
        result,info=native_input.weak_projection(result,folder/'weak')
        if result['m']>=2**31 or result['offsets'].dtype!=np.dtype('<i4'):
            raise native_input.IndexUnsupported('Existing cuGraph MG descriptor has int32 offsets')
        result['offsets']=np.memmap(result['paths']['offsets'],dtype='<i4',mode='r',shape=(result['n']+1,))
        result['indices']=np.memmap(result['paths']['indices'],dtype='<i4',mode='r',shape=(result['m'],))
        result['weights']=None
    else:
        info=dict(operation='identity canonical stored arcs',stored_arcs=graph['m'])
        result['directed']=True
    return result,info


def source_child(args):
    if args.lease_fd is None:raise ValueError('Source child requires inherited global GPU lease')
    held=os.fstat(args.lease_fd);actual=args.lock_file.stat()
    if (held.st_dev,held.st_ino)!=(actual.st_dev,actual.st_ino):raise ValueError('Child lease is not the global experiment lock')
    fcntl.flock(args.lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    configure(args);folder=args.output.resolve();graph=shared.load_input(args.input_root/'canonical'/args.dataset)
    source=graph['sources'][args.source_index] if args.algorithm in ('bfs','sssp') else 0
    child_epoch,digest=epoch(args)
    if digest!=args.epoch_sha256:
        shared.save(folder/'rejected_child_epoch.json', child_epoch)
        raise ValueError('Child source/environment differs from parent')
    expected,ref=shared.reference_cached(graph,args.algorithm,source,args.reference_cache)
    record=dict(status='RUNNING',system='cugraph_mg',algorithm=args.algorithm,source=source,
        count=args.count,epoch_sha256=digest,calls={},reference=ref)
    backend=None;values={}
    def stage(label,active):shared.save(folder/'stage.json',dict(label=label,active=active,started_monotonic=time.monotonic()))
    try:
        with shared.complete_timeout(args.timeout_seconds):
            stage('first',True);start=time.perf_counter()
            # Charge existing by-value rank serialization setup to T_first.
            import cloudpickle
            cloudpickle.register_pickle_by_value(adapter);cloudpickle.register_pickle_by_value(platform_probe)
            selected,projection=prepared_graph(graph,args.algorithm,folder)
            backend=adapter.CuGraphMG(selected,args.algorithm,source,args.uuids,args.cpus,folder/'dask')
            values['first']=backend.complete_query();duration=time.perf_counter()-start;stage('first',False)
        record['calls']['first']=dict(seconds=duration,metrics=dict(backend.metrics))
        record['preparation']=projection
        for label in ['warmup','ready']:
            with shared.complete_timeout(args.timeout_seconds):
                stage(label,True);start=time.perf_counter();values[label]=backend.complete_query()
                duration=time.perf_counter()-start;stage(label,False)
            record['calls'][label]=dict(seconds=duration,metrics=dict(backend.metrics))
        record['observation']=backend.observe();record['adapter_contract']=backend.contract
        backend.close();backend=None;record['cleanup_completed']=True
        # Audit after close proves full host outputs survive graph/client lifetime.
        for label in ['first','warmup','ready']:
            record['calls'][label].update(quality=shared.audit_output(values[label],expected,graph,args.algorithm),
                output=shared.persist_output(folder,values[label]))
        record['quality_pass']=all(x['quality']['pass'] for x in record['calls'].values())
        if not record['quality_pass']:raise ValueError('cuGraph MG complete output quality failed')
        record.update(status='PASS',T_first_seconds=record['calls']['first']['seconds'],T_ready_seconds=record['calls']['ready']['seconds'])
    except BaseException as error:
        record.update(status='FAIL',error=type(error).__name__+': '+str(error));raise
    finally:
        if backend is not None:
            try:backend.close()
            except BaseException as error:record['cleanup_error']=repr(error)
        shared.save(folder/'result.json',record)


def validate_proof(path,kind,args,digest,graph):
    value=shared.validate_proof(path,kind=kind,system='cugraph_mg',algorithm=args.algorithm,epoch_sha=digest,graph=graph)
    proof=json.loads(Path(path).read_text())
    if proof.get('gpu_count')!=args.count:raise ValueError('MG proof has wrong GPU count')
    for item in proof['sources']:
        child=json.loads(Path(item['result']['path']).read_text())
        if not child.get('cleanup_completed'):raise ValueError('MG cleanup not confirmed')
        for call in child['calls'].values():
            if call['metrics']['devices_used']!=list(range(args.count)):raise ValueError('Incomplete actual MG rank domain')
    return value


def child_command(args,index,folder,digest,lease_fd):
    cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--execute-gpu','--source-index',str(index),'--epoch-sha256',digest,'--lease-fd',str(lease_fd)]
    for action in parser()._actions:
        if action.dest in ('help','execute_gpu','make_tiny','source_index','epoch_sha256','lease_fd','output'):continue
        value=getattr(args,action.dest,None)
        if value is not None:cmd.extend([action.option_strings[0],str(value)])
    return cmd+['--output',str(folder)]


def job(args):
    if not args.execute_gpu:raise ValueError('GPU execution requires explicit --execute-gpu by owner')
    configure(args);args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    record=dict(schema=PROTOCOL,status='RUNNING',system='cugraph_mg',algorithm=args.algorithm,
        gpu_count=args.count,dataset=args.dataset,phase=args.phase,repeat=args.repeat,sources=[])
    with args.lock_file.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.sched_setaffinity(0,args.cpus)
        os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(args.uuids),
            NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_SOCKET_IFNAME='lo',NUMBA_NUM_THREADS='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        os.environ['PYTHONPATH']=str(HERE)+os.pathsep+os.environ.get('PYTHONPATH','')
        try:
            graph=shared.load_input(args.input_root/'canonical'/args.dataset);data,digest=epoch(args)
            shared.save(args.output/'epoch.json',data)
            record.update(epoch=shared.binding(args.output/'epoch.json'),epoch_sha256=digest,
                input_sha256=graph['input_sha256'],directed=graph['directed'],qualification={})
            if args.phase in ('target','formal'):
                if not args.tiny_proof:raise ValueError('Tiny proof required')
                record['qualification']['tiny']=validate_proof(args.tiny_proof,'tiny',args,digest,graph)
            if args.phase=='formal':
                if not args.target_proof:raise ValueError('Target proof required')
                record['qualification']['target']=validate_proof(args.target_proof,'target',args,digest,graph)
            sources=graph['sources'] if args.algorithm in ('bfs','sssp') else [0]
            for index,source in enumerate(sources):
                admission=[]
                for uuid in args.uuids:
                    proxy=SimpleNamespace(**vars(args));proxy.gpu_uuid=uuid;admission.append(shared.resource_check(proxy,graph))
                folder=args.output/f'source_{index}';folder.mkdir()
                shared.save(folder/'admission.json',admission)
                with (folder/'worker.log').open('x') as log:
                    process=None
                    try:
                        process=subprocess.Popen(child_command(args,index,folder,digest,lock.fileno()),stdout=log,stderr=subprocess.STDOUT,
                            pass_fds=(lock.fileno(),),start_new_session=True)
                        observation=wait_native(process,dict(timeout_seconds=3*args.timeout_seconds+600,
                            rss_cap_bytes=int(args.rss_cap_gib*(1<<30))),args.uuids)
                        code=process.wait();shared.save(folder/'process_observer.json',observation)
                    finally:stop(process)
                if code:raise RuntimeError('Owned MG source process failed: '+str(code))
                result=json.loads((folder/'result.json').read_text())
                if result.get('status')!='PASS' or not result.get('quality_pass'):raise ValueError('Incomplete source result')
                record['sources'].append(dict(source=source,result=shared.binding(folder/'result.json')))
                shared.save(args.output/'result.json',record)
            record.update(status='PASS_FORMAL' if args.phase=='formal' else 'PASS_QUALIFICATION',quality_pass=True,
                timing_scope='continuous canonical host CSR through full owned host output; fixed one warmup before T_ready')
        except BaseException as error:record.update(status='FAIL',error=type(error).__name__+': '+str(error));raise
        finally:shared.save(args.output/'result.json',record)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-gpu',action='store_true');p.add_argument('--make-tiny',type=Path)
    for key in ['input-root','setup','lock-file','reference-cache','output']:p.add_argument('--'+key,type=Path)
    p.add_argument('--dataset',choices=shared.DATASETS+shared.TINY_DATASETS);p.add_argument('--algorithm',choices=shared.ALGORITHMS)
    p.add_argument('--count',type=int,choices=[2,4,8]);p.add_argument('--gpu-uuids');p.add_argument('--cpu-set')
    p.add_argument('--phase',choices=['tiny','target','formal'],default='tiny');p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--tiny-proof',type=Path);p.add_argument('--target-proof',type=Path)
    p.add_argument('--budget-mib',type=int,default=22000);p.add_argument('--min-host-free-gib',type=float,default=8.)
    p.add_argument('--rss-cap-gib',type=float,default=180.)
    p.add_argument('--timeout-seconds',type=int,default=1800);p.add_argument('--source-index',type=int);p.add_argument('--epoch-sha256');p.add_argument('--lease-fd',type=int)
    return p


def main():
    p=parser();args=p.parse_args()
    if args.make_tiny:native_worker.make_tiny(args.make_tiny);return
    for key in ('input_root','setup','lock_file','reference_cache','output','dataset','algorithm','count','gpu_uuids','cpu_set'):
        if getattr(args,key) is None:p.error('--'+key.replace('_','-')+' required')
    if args.source_index is not None:
        if not args.execute_gpu or not args.epoch_sha256:p.error('Child requires explicit execution and parent epoch')
        source_child(args)
    else:job(args)

if __name__=='__main__':main()
