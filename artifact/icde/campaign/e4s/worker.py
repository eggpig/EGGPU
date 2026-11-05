#!/usr/bin/env python3
"""E4-S complete distinct-source SSSP batches; primary-agent execution only.

All author-specific preparation executes afresh inside T_first. All complete
FP64 host answers are materialized and retained before the completion timestamp.
Archive/quality transport uses a pipe after that timestamp, avoiding a 100GiB
duplicate output archive for a large 512-query batch.
"""
from pathlib import Path
import argparse, fcntl, hashlib, importlib.util, json, os, select, shutil, signal, subprocess, sys, time
import numpy as np
import input_contract as inputs
import sg_clock
common=inputs.common
VARIANTS=('eggpu','kgraph_sparse','kgraph_dense','c2graph_pm')


def parse_rows(path,marker):
    return [json.loads(line[len(marker):]) for line in Path(path).read_text().splitlines() if line.startswith(marker)]


def validate(args):
    if not args.execute:raise ValueError('Only primary agent may pass --execute; GPU submission remains globally serialized')
    if args.dataset not in (*inputs.DATASETS,*common.TINY_DATASETS):raise ValueError('Outside frozen E4-S pool')
    if (args.dataset in common.TINY_DATASETS)!=(args.phase=='tiny'):raise ValueError('Tiny/real phase mismatch')
    if args.batch not in inputs.BATCHES or args.repeat<0 or (args.phase=='formal' and args.repeat not in range(10)):raise ValueError('Frozen batch/repetition contract')
    expected='kgraph_sparse' if args.dataset=='US' else 'kgraph_dense'
    if args.dataset in inputs.DATASETS and args.variant.startswith('kgraph') and args.variant!=expected:raise ValueError('KGraph variant is frozen by graph category, not observed speed')
    cpus={int(x) for x in args.cpus.split(',')}
    if len(cpus)!=16 or not cpus<=os.sched_getaffinity(0):raise ValueError('Exactly 16 allowed logical CPUs required')
    os.sched_setaffinity(0,cpus)
    threads='16' if args.variant.startswith('kgraph') else '1'
    os.environ.update(OMP_NUM_THREADS=threads,OMP_THREAD_LIMIT=threads,OMP_DYNAMIC='FALSE',
                     OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',
                     MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                     CUDA_VISIBLE_DEVICES='' if args.variant.startswith('kgraph') else args.gpu_uuid)
    os.environ['EGGPU_ARENA_LIBRARY_DIR']=str(args.arena_library_dir.resolve())


def epoch(args):
    build=json.loads(args.author_build.read_text())
    if build.get('status')!='PASS_BUILD_ONLY' or build.get('schema')!='e4s.author.fp64.build.v1':raise ValueError('Fixed complete FP64 author build required')
    files={'worker':common.binding(__file__),'inputs':common.binding(inputs.__file__),'common':common.binding(common.__file__),
           'reference':common.binding(args.reference_module),'setup':common.binding(args.setup),
           'clock':common.binding(sg_clock.__file__),'launcher':common.binding(Path(__file__).with_name('sg_clock_exec.py'))}
    reference=reference_module(args.reference_module)
    files['reference_protocol']=common.binding(reference.c.__file__)
    files['reference_weight_recipe']=common.binding(Path(reference.c.__file__).with_name('state_inputs.py'))
    entry={'schema':'e4s.epoch.v1','variant':args.variant,'batch':args.batch,'python':common.binding(sys.executable),
       'files':files,'affinity':sorted(os.sched_getaffinity(0)),'quality':{'all_distances_exact':True,'output':'complete owning immutable FP64; +inf'},
       'environment':{k:v for k,v in os.environ.items() if k.startswith(('EGGPU_','EGMGPU_')) or k in
                      ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OMP_DYNAMIC','OMP_WAIT_POLICY','GOMP_SPINCOUNT',
                       'MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','LD_LIBRARY_PATH','LD_PRELOAD')},
       'gpu_uuid':None if args.variant.startswith('kgraph') else args.gpu_uuid,'gpu_count':0 if args.variant.startswith('kgraph') else 1,
       'host_cpu_budget':16,'kgraph_partitions':inputs.KGRAPH_PARTITIONS,'kgraph_pivot_strategy':-1,'kgraph_pivot_count':-1,
       'c2_pm_iterations':8,'reference_cache_gib':args.reference_cache_gib,'timeout':args.timeout,'audit_timeout':args.audit_timeout,
       'reference_versions':{'numpy':reference.np.__version__,'scipy':reference.scipy.__version__}}
    if args.variant=='eggpu':
        sources={}
        for base in (args.release/'easygraph',args.release/'cpp_easygraph',args.release/'gpu_easygraph'):
            for path in sorted(base.rglob('*')):
                if path.is_file() and path.suffix in ('.py','.cpp','.h','.cu','.cuh','.hpp'):
                    sources[str(path.relative_to(args.release))]=common.content_binding(path)
        entry.update(release_sources=sources,native_dso=common.binding(args.native_dso),
                     arena_libraries={p.name:common.binding(p) for p in sorted(args.arena_library_dir.glob('*.so'))},budget_mib=args.budget_mib)
    else:
        item=build['binaries'][args.variant]
        if common.binding(item['path'])!=item:raise ValueError('Fixed author binary changed')
        entry.update(author_build=common.binding(args.author_build),binary=item,author_lineage=build['base_builds'])
        if args.variant=='c2graph_pm':
            prior=json.loads((args.root/'build/c2graph/receipt.json').read_text());compressor=common.binding(prior['compressor'])
            if compressor['sha256']!=prior['compressor_sha256']:raise ValueError('Author compressor changed')
            entry['compressor']=compressor
    return entry,common.digest_json(entry)


def proof(path,phase,args,identity,graph,sequence):
    if path is None:raise ValueError('Missing explicit '+phase+' qualification')
    data=json.loads(path.read_text())
    if data.get('status')!='PASS_E4S_QUALIFICATION' or data['phase']!=phase or data['epoch_sha256']!=identity or data['quality_pass'] is not True:
        raise ValueError('Wrong E4-S source/configuration qualification')
    if data['directed']!=graph['directed'] or data['variant']!=args.variant or data['batch']!=args.batch:raise ValueError('Wrong batch/semantic qualification')
    if phase=='target' and (data['input_sha256']!=graph['input_sha256'] or data['source_sequence']!=sequence):raise ValueError('Target qualification differs from exact input/source sequence')
    return common.binding(path)


def reference_module(path):
    spec=importlib.util.spec_from_file_location('e4s_independent_reference',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def quality_context(args,graph):
    module=reference_module(args.reference_module)
    cache=module.ReferenceCache(args.reference_cache,int(args.reference_cache_gib*(1<<30)))
    row={'n':graph['n'],'m':graph['m'],'canonical':{k:common.binding(Path(graph['directory'])/(k+'.bin')) for k in ('offsets','indices','weights')}}
    def audit(values,sequence,source):
        quality=module.output_quality(values,row,0,'sssp',source,cache)
        digest=hashlib.sha256(memoryview(values).cast('B')).hexdigest()
        return {'sequence':sequence,'source':source,'values':len(values),'output_sha256':digest,'quality':quality}
    return audit


def read_answer(fd,n,process,deadline):
    values=np.empty(n,dtype='<f8');view=memoryview(values).cast('B');position=0
    while position<len(view):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Author batch/output transport deadline')
        ready,_,_=select.select([fd],[],[],min(1.,remaining))
        if not ready:
            if process.poll() is not None:raise RuntimeError('Author exited before complete output')
            continue
        block=os.read(fd,min(1<<20,len(view)-position))
        if not block:raise RuntimeError('Truncated author full-vector output')
        view[position:position+len(block)]=block;position+=len(block)
    values.flags.writeable=False;return values


def wait_native_complete(path,process,deadline):
    """The flushed event precedes all pipe writes and independent oracle work."""
    pending=''
    with Path(path).open() as stream:
        while True:
            pending+=stream.read()
            lines=pending.split('\n');pending=lines.pop()
            for line in lines:
                if 'E4S_COMPLETE_JSON ' in line:
                    row=json.loads(line.split('E4S_COMPLETE_JSON ',1)[1])
                    if row.get('pid')!=process.pid or type(row.get('complete_monotonic_ns'))is not int:
                        raise ValueError('Malformed native full-batch completion event')
                    return row
            if process.poll() is not None:raise RuntimeError('Author exited before its full-batch completion event')
            if time.monotonic()>deadline:raise TimeoutError('Author complete batch timeout; oracle has not started')
            time.sleep(.02)


def author_batch(args,graph,sequence,entry,audit):
    out=args.output;stage=out/'stage';native=entry['binary']['path'];sources_path=out/'sources.txt'
    domain=sg_clock.clock_domain();begin_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC);prepare_begin=time.perf_counter();details={}
    work_deadline=time.monotonic()+args.timeout
    sources_path.write_text('\n'.join(map(str,sequence))+'\n')
    if args.variant.startswith('kgraph'):
        details['partitioning']=inputs.kgraph_layout(graph,stage)
        command=[native,'-p',str(inputs.KGRAPH_PARTITIONS),'-ps','-1','-pn','-1','-qn',str(len(sequence)),
                 '-qf',str(sources_path),str(stage/'inter.adj'),str(stage/'intra.adj'),str(stage/'parts.txt')]
        marker='EGGPU_KGRAPH_JSON '
    else:
        details['conversion']=inputs.c2_layout(graph,stage);compression_started=time.perf_counter()
        with (out/'compression.log').open('wb') as log:
            result=subprocess.run([entry['compressor']['path'],str(stage)],stdout=log,stderr=subprocess.STDOUT,
                                  env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=max(.001,work_deadline-time.monotonic()))
        details['compression_seconds']=time.perf_counter()-compression_started
        if result.returncode:raise RuntimeError('Author compression failed')
        command=[native,str(stage),str(sources_path),'8',str(len(sequence))];marker='C2_PM_JSON '
    prep_seconds=time.perf_counter()-prepare_begin
    if time.monotonic()>=work_deadline:raise TimeoutError('Author required preparation exceeded complete-task timeout')
    read_fd,write_fd=os.pipe();env=dict(os.environ)
    env['EGGPU_KGRAPH_OUTPUT' if args.variant.startswith('kgraph') else 'EGGPU_E4S_OUTPUT']='/proc/self/fd/'+str(write_fd)
    env.update(EGGPU_SG_START_NS=str(begin_ns),EGGPU_SG_LAUNCH=str(out/'launch.json'))
    env['EGGPU_PM_OUTPUT_DIR']=str(out);checks=[];process=None
    try:
        with (out/'native.log').open('wb') as log:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('sg_clock_exec.py')),*command],
                stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=(write_fd,),start_new_session=True)
            os.close(write_fd);write_fd=-1
            complete_event=wait_native_complete(out/'native.log',process,work_deadline)
            deadline=time.monotonic()+args.audit_timeout
            with common.complete_timeout(args.audit_timeout):
                for index,source in enumerate(sequence):checks.append(audit(read_answer(read_fd,graph['n'],process,deadline),index,source))
            while process.poll() is None:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Author completion/archive deadline')
                ready,_,_=select.select([read_fd],[],[],min(.1,remaining))
                if ready and os.read(read_fd,1):raise ValueError('Author returned extra distance data')
            if process.returncode:raise RuntimeError('Author batch exited '+str(process.returncode))
        rows=parse_rows(out/'native.log',marker)
        if len(rows)!=1:raise ValueError('Missing complete batch timestamp')
        summary=rows[0];end_ns=int(summary['complete_monotonic_ns'])
        if complete_event!={'complete_monotonic_ns':end_ns,'pid':summary['pid']}:
            raise ValueError('Archived summary differs from pre-archive completion event')
        parent_end_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        launch=json.loads((out/'launch.json').read_text())
        if (launch['clock_domain']!=domain or launch['parent_monotonic_start_ns']!=begin_ns or
                launch['pid']!=process.pid or summary['pid']!=process.pid or
                not begin_ns<=launch['monotonic_launch_ns']<end_ns<=parent_end_ns):
            raise ValueError('Native batch is not bound to this PID/monotonic namespace/continuous interval')
        if args.variant.startswith('kgraph'):
            if summary['queries']!=len(sequence) or summary['n']!=graph['n']:raise ValueError('KGraph query binding differs')
            native_prep=summary['graph_partition_pivot_prepare_seconds'];batch=summary['complete_host_seconds']
        else:
            requests=parse_rows(out/'native.log','C2_PM_OUTPUT_JSON ')
            if summary['requested']!=len(sequence) or summary['completed']!=len(sequence) or [r['request_id'] for r in requests]!=list(range(len(sequence))) or [r['source'] for r in requests]!=sequence:
                raise ValueError('C2 PM per-buffer source binding differs')
            native_prep=summary['graph_preparation_seconds'];batch=summary['batch_complete_host_seconds'];details['requests']=requests
        total=(end_ns-begin_ns)/1e9
        if total+1e-6<prep_seconds+native_prep+batch:raise ValueError('Preprocessing missing from actual complete interval')
        details.update(command=command,summary=summary,source_file=common.binding(sources_path),
            continuous_interval={'start_ns':begin_ns,'end_ns':end_ns,'parent_end_ns':parent_end_ns,
                'clock_domain':domain,'launch':common.binding(out/'launch.json'),'pid':process.pid,'phase_durations_summed':False},
            adapter_preparation_seconds=prep_seconds,native_preparation_seconds=native_prep,
            query_batch_seconds=batch,first_complete_seconds=total,
            other_first_request_seconds=max(0.,total-prep_seconds-native_prep-batch))
        shutil.rmtree(stage)
        return details,checks
    finally:
        os.close(read_fd)
        if write_fd>=0:os.close(write_fd)
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def eggpu_batch(args,graph,sequence,audit):
    session=None;held=[];requests=[]
    try:
        with common.complete_timeout(args.timeout):
            begin=time.perf_counter();sys.path.insert(0,str(args.release.resolve()));sys.path.insert(1,str(args.native_dso.resolve().parent))
            from easygraph.gpu import Session
            if Path(sys.modules[Session.__module__].__file__).resolve()!=args.release.resolve()/'easygraph/gpu/session.py':raise ValueError('Wrong public EGGPU source')
            native_module=sys.modules.get('cpp_easygraph')
            if native_module is not None and Path(native_module.__file__).resolve()!=args.native_dso.resolve():raise ValueError('Imported EGGPU DSO differs from the frozen build')
            payload={'nodes':np.arange(graph['n'],dtype=np.int64),'offsets':graph['offsets'],'indices':graph['indices'],'weights':graph['weights']}
            session=Session.create({'g':payload},devices=(0,),state_model='static',algorithms=('sssp',),input_format='csr',budget_bytes=args.budget_mib<<20)
            ready=time.perf_counter()
            for index,source in enumerate(sequence):
                answer=session.query('g','sssp',{'source':source});held.append(answer.values)
                requests.append({'sequence':index,'source':source,'seconds':answer.seconds,'backend_metrics':answer.backend_metrics})
            end=time.perf_counter()
        # Full-vector audit is deliberately outside both the measured interval
        # and the complete-task execution timeout; it has its own explicit cap.
        with common.complete_timeout(args.audit_timeout):
            addresses=[int(v.__array_interface__['data'][0]) for v in held]
            if len(set(addresses))!=len(held):raise ValueError('Batch output owners alias')
            before=[hashlib.sha256(memoryview(v).cast('B')).hexdigest() for v in held]
            session.close();session=None
            checks=[]
            for i,(source,values) in enumerate(zip(sequence,held)):
                row=audit(values,i,source)
                if row['output_sha256']!=before[i]:raise ValueError('Output changed after Session.close')
                checks.append(dict(row,survives_session_close=True))
        return {'adapter_preparation_seconds':ready-begin,'query_batch_seconds':end-ready,'first_complete_seconds':end-begin,
                'requests':requests,'all_outputs_retained_through_close':True,'public_api':'Session.create(input_format=csr), static SSSP'},checks
    finally:
        if session is not None:session.close()


def main(args):
    validate(args);args.output.mkdir(parents=True,exist_ok=False)
    record={'schema':'e4s.complete.batch.v1','status':'RUNNING','phase':args.phase,'variant':args.variant,'dataset':args.dataset,
            'batch':args.batch,'repeat':args.repeat,'quality_pass':False,'performance_eligible':False,
            'gpu_executed':False,'cpu_budget':16}
    common.save(args.output/'result.json',record)
    try:
        with args.lock_file.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            graph=common.load_input(args.input_root/'canonical'/args.dataset);sequence=inputs.load_sources(args.sources,graph,args.batch)
            entry,identity=epoch(args);common.save(args.output/'epoch.json',entry)
            record.update(epoch_sha256=identity,epoch=common.binding(args.output/'epoch.json'),input_sha256=graph['input_sha256'],
                          directed=graph['directed'],source_sequence=sequence,sources=common.binding(args.sources))
            if args.phase in ('target','formal'):record['tiny_proof']=proof(args.tiny_proof,'tiny',args,identity,graph,sequence)
            if args.phase=='formal':record['target_proof']=proof(args.target_proof,'target',args,identity,graph,sequence)
            host=common.cgroup_memory();required=max(8<<30,20*graph['n']*args.batch+128*graph['m']+256*graph['n']+(2<<30))
            if host['available_bytes']<required:raise RuntimeError('Insufficient host/cgroup headroom for all native and public complete outputs')
            record['resources']={'host':host,'required_host_bytes':required}
            if not args.variant.startswith('kgraph'):record['resources']['gpu']=common.resource_check(args,graph)
            disk=shutil.disk_usage(args.output).free;required_disk=(24*graph['m']+32*graph['n']+(2<<30)) if args.variant!='eggpu' else 2<<30
            if disk<required_disk:raise RuntimeError('Author preparation disk headroom insufficient; no graph data dropped')
            audit=quality_context(args,graph)
            record['gpu_executed']=not args.variant.startswith('kgraph')
            details,checks=eggpu_batch(args,graph,sequence,audit) if args.variant=='eggpu' else author_batch(args,graph,sequence,entry,audit)
            if epoch(args)[1]!=identity:raise RuntimeError('Source/binary/environment changed during batch')
            record.update(status='PASS_E4S_QUALIFICATION' if args.phase!='formal' else 'PASS_E4S_FORMAL',quality_pass=True,
                          performance_eligible=args.phase=='formal',measurements=details,checks=checks,
                          timing='Actual canonical-CSR-to-all-owning-FP64-results interval. All system-specific partitioning/compression, pivot preparation/memoization, runtime initialization and per-query work charged. Hashing/oracle and post-completion pipe serialization excluded.',
                          output_storage='Complete vectors audited and hashed after native completion; native simultaneous owners retained. No repeated full batch archive.',
                          reference_implementation=common.binding(args.reference_module))
    except BaseException as error:
        record.update(status='FAILED',error=type(error).__name__+': '+str(error));raise
    finally:common.save(args.output/'result.json',record)
    print(json.dumps({k:record[k] for k in ('status','variant','dataset','batch','quality_pass')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--batch',type=int,choices=inputs.BATCHES,default=128)
    p.add_argument('--phase',choices=('tiny','target','formal'),required=True);p.add_argument('--repeat',type=int,default=0)
    for key in ('root','input-root','sources','author-build','release','native-dso','arena-library-dir','reference-module','reference-cache','setup','lock-file','output'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('tiny-proof','target-proof'):p.add_argument('--'+key,type=Path)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--cpus',required=True);p.add_argument('--budget-mib',type=int,default=22000)
    p.add_argument('--reference-cache-gib',type=float,default=4.);p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--audit-timeout',type=int,default=86400)
    p.add_argument('--min-host-free-gib',type=float,default=8.)
    main(p.parse_args())
