"""GUM author-CSR complete-output qualification with cross-process monotonic T_first.

Only the experiment owner may pass --execute-gpu. GUM's native complete-query
rank maximum supplies T_ready. Existing host-complete CLOCK_MONOTONIC events
supply T_first from one parent origin; setup phase durations are never summed.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, fcntl, importlib.util, json, math, os, signal, subprocess, sys, time
import sg_worker as shared
import native_input
import native_worker
import gum_reused as reused
import gum_clock
from mg_process_observer import wait_native, stop

HERE=Path(__file__).resolve().parent
PROTOCOL='gum-author-fp32-complete-monotonic-v1'
DESCRIPTOR_NAME='graph.lawcsr'


def configure(args):
    if args.function != 'pr' or args.phase not in ('tiny','target'):
        raise ValueError('This candidate permits original FP32 PR qualification only; formal requires later qualification review')
    args.uuids=args.gpu_uuids.split(','); args.cpus=[int(x) for x in args.cpu_set.split(',')]
    if args.count not in (1,2,4,8) or len(args.uuids)!=args.count or len(set(args.uuids))!=args.count:
        raise ValueError('Supply exactly P distinct GPU UUIDs for P=1/2/4/8')
    if any(not x.startswith('GPU-') or x.strip()!=x for x in args.uuids):raise ValueError('Explicit GPU UUIDs required')
    if len(args.cpus)!=16 or len(set(args.cpus))!=16 or not set(args.cpus)<=set(os.sched_getaffinity(0)):
        raise ValueError('Exactly 16 available CPU cores are required')
    if (args.dataset in shared.TINY_DATASETS)!=(args.phase=='tiny'):raise ValueError('Separate tiny and real graphs')
    if args.repeat<0 or (args.phase=='formal' and args.repeat not in range(10)):raise ValueError('Formal repeats are 0..9')
    return args


def parse_times(runtime_path,folder,function,count):
    spec=importlib.util.spec_from_file_location('gum_bound_runtime',runtime_path)
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    ranks=runtime.rank_diagnostics(folder,'gum',owning=True,pagerank=function=='pr')
    if sorted(row['rank'] for row in ranks)!=list(range(count)):raise ValueError('Incomplete rank timing domain')
    times=[]
    for i in range(3):
        durations=[row['rounds'][i]['warm_query_seconds'] for row in ranks]
        if any(not math.isfinite(x) or x<=0 for x in durations):raise ValueError('Invalid complete query interval')
        times.append(max(durations))
    return times,dict(rank_diagnostics=ranks,setup=runtime.setup_records(folder))


def precision_evidence(folder):
    import struct
    rows=[json.loads(line) for line in (Path(folder)/'timings.jsonl').read_text().splitlines()]
    queries=[r for r in rows if 'round' in r]
    alpha=struct.unpack('f',struct.pack('f',.85))[0]
    if [r['round'] for r in queries] != [0,1,2]:raise ValueError('Missing complete precision observations')
    for r in queries:
        if (r['rank_bytes'],r['message_bytes'],r['iterations'],r['alpha']) != (4,8,117,alpha):
            raise ValueError('Author FP32 precision, message, iterations or effective damping changed')
    return dict(queries=queries,requested_alpha=.85,effective_author_alpha=alpha,
        compute_dtype='float32',message_bytes=8,final_owned_dtype='float64',
        quality_reference_alpha=.85,output_normalization_inside_query=True)


def descriptor(graph,function,folder):
    """Create the extension-dispatched author CSR descriptor inside continuous T_first."""
    import numpy as np
    original=native_input.original_graph(graph,'canonical')
    if function=='wcc':
        selected,projection=native_input.weak_projection(original,folder/'weak')
        if selected['m']>=2**31 or selected['offsets'].dtype!=np.dtype('<i4'):
            raise native_input.IndexUnsupported('GUM author CSR loader requires int32 offsets')
    else:selected,projection=original,{'operation':'identity canonical stored arcs'}
    paths=selected['paths']; weight_path=Path(graph['directory'])/'weights.bin'
    path=folder/DESCRIPTOR_NAME
    lines=['eggpu-law-csr-v1',str(selected['n']),str(selected['m']),'1' if function=='sssp' else '0',
           str(Path(paths['offsets']).resolve()),str(Path(paths['indices']).resolve()),
           str(weight_path.resolve()) if function=='sssp' else '-']
    path.write_text('\n'.join(lines)+'\n')
    return path,projection


def epoch(args,artifact):
    preload={str(Path(p).resolve()):shared.binding(p) for p in args.ld_preload.replace(':',' ').split()}
    data=dict(protocol=PROTOCOL,system='gum',function=args.function,count=args.count,gpu_uuids=args.uuids,
        cpu_affinity=sorted(args.cpus),artifact=artifact,python=shared.binding(sys.executable),
        core_python=shared.binding(args.core_python),mpiexec=shared.binding(args.mpiexec),
        dependencies=native_worker.runtime_dependencies(artifact['binary']['path']),preload=preload,
        ld_library_path=args.ld_library_path,quality=shared.QUALITY,pr_iterations=117,pr_alpha=.85,
        precision=dict(compute='author float32',damping='author float32',message_bytes=8,
                       final_owned_output='float64',normalization_inside_query=True),
        partitioner='seg',load_balance='cm',work_stealing=False,sssp='author standard uint32',
        resources=dict(rss_cap_gib=args.rss_cap_gib,admission_gpu_budget_mib=args.budget_mib,
            minimum_host_free_gib=args.min_host_free_gib,process_timeout_seconds=args.timeout_seconds),
        timing={'T_first':'parent CLOCK_MONOTONIC origin to existing latest-rank first full host-output event',
                'T_ready':'rank-max native owning full-host-output complete query'},
        clock_source=gum_clock.verify_source(artifact,args.function),
        files={p:shared.binding(HERE/p) for p in ['gum_worker.py','gum_reused.py','gum_rank_exec.py',
            'mg_process_observer.py','sg_worker.py','native_input.py','native_worker.py','gum_clock.py']})
    return data,shared.digest_json(data)


def validate_proof(path,kind,args,digest,graph):
    p=json.loads(Path(path).read_text())
    if any(p.get(k)!=v for k,v in dict(status='PASS_QUALIFICATION',phase=kind,system='gum',
        function=args.function,count=args.count,epoch_sha256=digest,quality_pass=True).items()):
        raise ValueError('GUM qualification source/config/function mismatch')
    if kind=='tiny':
        if p['dataset'] not in shared.TINY_DATASETS or p['directed']!=graph['directed']:raise ValueError('Tiny directedness mismatch')
    elif p['input_sha256']!=graph['input_sha256']:raise ValueError('Target input mismatch')
    if shared.binding(p['epoch']['path'])!=p['epoch'] or shared.digest_json(json.loads(Path(p['epoch']['path']).read_text()))!=digest:
        raise ValueError('Qualification epoch changed')
    if len(p['sources'])!=(3 if args.function in ('bfs','sssp') else 1):raise ValueError('Incomplete source block')
    for item in p['sources']:
        if shared.binding(item['result']['path'])!=item['result']:raise ValueError('Source receipt changed')
        result=json.loads(Path(item['result']['path']).read_text())
        if result['status']!='PASS' or not result['quality_pass'] or len(result['rounds'])!=3:raise ValueError('Failed source')
        timing=result.get('continuous_first',{})
        if timing.get('status')!='PASS_CONTINUOUS_MONOTONIC_INTERVAL':raise ValueError('Missing continuous first-result qualification')
        for rank in timing['ranks']:
            for field in ['launch','markers']:
                if shared.binding(rank[field]['path'])!=rank[field]:raise ValueError('First-result clock evidence changed')
        for row in result['rounds']:
            if not row['quality']['pass'] or shared.binding(row['raw_output']['path'])!=row['raw_output']:
                raise ValueError('Complete native output changed')
    return shared.binding(path)


def execute_source(args,graph,source,artifact,folder):
    folder.mkdir();native=folder/'native';native.mkdir();scratch=folder/'tmp';scratch.mkdir()
    records=[]
    for uuid in args.uuids:
        proxy=SimpleNamespace(**vars(args));proxy.gpu_uuid=uuid
        records.append(shared.resource_check(proxy,graph))
    expected,ref=shared.reference_cached(graph,args.function,source,args.reference_cache)
    env,controls=reused.execution_environment(args,args.uuids,args.cpus,native,scratch)
    command=reused.native_command(args,artifact['binary']['path'],folder/DESCRIPTOR_NAME,graph['n'],source or 0,args.count,native)
    domain=gum_clock.clock_domain()
    record=dict(status='RUNNING',function=args.function,source=source,artifact=artifact,
        cpu_affinity=sorted(args.cpus),gpu_uuids=args.uuids,physical_devices=[r['selected']['index'] for r in records],
        admission=records,controls=controls,command=command,reference=ref,
        T_first_seconds=None,contiguous_canonical_to_first_result=False,
        first_result_timing_status='PENDING_CURRENT_EXECUTION',rounds=[])
    child=None
    try:
        first_start_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        env['EGGPU_GUM_MONOTONIC_START_NS']=str(first_start_ns)
        record.update(parent_monotonic_start_ns=first_start_ns,parent_clock_domain=domain)
        begin=time.perf_counter();path,projection=descriptor(graph,args.function,folder);prep=time.perf_counter()-begin
        record.update(preparation=projection,canonical_projection_seconds=prep)
        with (native/'native.log').open('x') as log:
            start=time.perf_counter()
            child=subprocess.Popen(command,env=env,cwd=scratch,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['observer']=wait_native(child,{'rss_cap_bytes':int(args.rss_cap_gib*(1<<30)),
                'timeout_seconds':args.timeout_seconds},args.uuids)
            code=child.wait();record['process_wall_seconds']=time.perf_counter()-start
        parent_end_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        if code:raise RuntimeError('GUM author process exited '+str(code))
        record['rank_proofs']=reused.native_proofs(native,record)
        durations,record['native_timing']=parse_times(artifact['runtime']['path'],native,args.function,args.count)
        record['precision_evidence']=precision_evidence(native)
        extension='f64' if args.function=='pr' else 'i64'
        for index in range(3):
            raw=native/f'values-{index}.{extension}'
            values=reused.normalise_native(raw,args.function,graph['n']);values.flags.writeable=False
            quality=shared.audit_output(values,expected,graph,args.function)
            record['rounds'].append(dict(round=index,role=['initial_native_query','warmup','ready'][index],
                complete_query_seconds=durations[index],quality=quality,raw_output=shared.binding(raw),
                audit_normalization='Native timer already includes FP32-to-owned-FP64 normalization; Python quality audit retains raw full output'))
        record['continuous_first']=gum_clock.first_result(native,args.count,first_start_ns,parent_end_ns,domain)
        record.update(T_first_seconds=record['continuous_first']['seconds'],contiguous_canonical_to_first_result=True,
            first_result_timing_status='PASS_CONTINUOUS_MONOTONIC_INTERVAL')
        record['T_ready_seconds']=durations[2];record['quality_pass']=all(r['quality']['pass'] for r in record['rounds'])
        if not record['quality_pass']:raise ValueError('Complete-output numerical quality failed')
        record['status']='PASS'
    except BaseException as error:
        record.update(status='FAIL',error=type(error).__name__+': '+str(error));raise
    finally:
        stop(child);shared.save(folder/'result.json',record)
    return shared.binding(folder/'result.json')


def job(args):
    if not args.execute_gpu:raise ValueError('Execution requires explicit --execute-gpu by experiment owner')
    configure(args);args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    record=dict(schema=PROTOCOL,status='RUNNING',system='gum',function=args.function,count=args.count,
        dataset=args.dataset,phase=args.phase,repeat=args.repeat,formal=args.phase=='formal',
        T_first_available=True,sources=[])
    def interrupt(signum,frame):raise InterruptedError('Owned job interrupted')
    signal.signal(signal.SIGTERM,interrupt)
    with args.lock_file.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.sched_setaffinity(0,args.cpus)
        try:
            graph=shared.load_input(args.input_root/'canonical'/args.dataset)
            artifact=reused.artifact_identity(args.build_receipt,args.function)
            if args.function=='sssp':record['weight_domain']=reused.weight_domain(graph)
            data,digest=epoch(args,artifact);shared.save(args.output/'epoch.json',data)
            record.update(input_sha256=graph['input_sha256'],directed=graph['directed'],epoch_sha256=digest,
                          epoch=shared.binding(args.output/'epoch.json'),qualification={})
            if args.phase in ('target','formal'):
                if not args.tiny_proof:raise ValueError('Explicit tiny proof required')
                record['qualification']['tiny']=validate_proof(args.tiny_proof,'tiny',args,digest,graph)
            if args.phase=='formal':
                if not args.target_proof:raise ValueError('Explicit same-input target proof required')
                record['qualification']['target']=validate_proof(args.target_proof,'target',args,digest,graph)
            sources=graph['sources'] if args.function in ('bfs','sssp') else [None]
            for index,source in enumerate(sources):
                result=execute_source(args,graph,source,artifact,args.output/f'source_{index}')
                record['sources'].append(dict(source=source,result=result))
            record.update(status='PASS_FORMAL' if args.phase=='formal' else 'PASS_QUALIFICATION',quality_pass=True,
                performance_scope='Continuous T_first and native complete-query T_ready, source-qualified together')
        except BaseException as error:
            record.update(status='FAIL',error=type(error).__name__+': '+str(error));raise
        finally:shared.save(args.output/'result.json',record)
    return record


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-gpu',action='store_true');p.add_argument('--make-tiny',type=Path)
    for key in ['input-root','setup','lock-file','reference-cache','output','build-receipt','mpiexec','core-python']:
        p.add_argument('--'+key,type=Path)
    p.add_argument('--dataset',choices=shared.DATASETS+shared.TINY_DATASETS)
    p.add_argument('--function',choices=['pr']);p.add_argument('--count',type=int,choices=[1,2,4,8])
    p.add_argument('--gpu-uuids');p.add_argument('--cpu-set');p.add_argument('--phase',choices=['tiny','target'],default='tiny')
    p.add_argument('--repeat',type=int,default=0);p.add_argument('--tiny-proof',type=Path);p.add_argument('--target-proof',type=Path)
    p.add_argument('--ld-preload',default='');p.add_argument('--ld-library-path',default='')
    p.add_argument('--budget-mib',type=int,default=22000);p.add_argument('--min-host-free-gib',type=float,default=8)
    p.add_argument('--rss-cap-gib',type=float,default=180);p.add_argument('--timeout-seconds',type=int,default=1800)
    return p


def main():
    p=parser();args=p.parse_args()
    if args.make_tiny:native_worker.make_tiny(args.make_tiny);return
    for name in ['input_root','setup','lock_file','reference_cache','output','build_receipt','mpiexec','core_python',
                 'dataset','function','count','gpu_uuids','cpu_set']:
        if getattr(args,name) is None:p.error('--'+name.replace('_','-')+' is required')
    job(args)

if __name__=='__main__':main()
