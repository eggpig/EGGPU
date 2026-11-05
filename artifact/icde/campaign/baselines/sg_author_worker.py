#!/usr/bin/env python3
"""Canonical E1 adapter for already built author systems; root owns GPU runs.

No author algorithm changes. Inputs/outputs use the current campaign contract;
the exact executable, correctness overlays, CPU layout and timing scopes remain
explicit. Tiny and target proofs are required before ten-repeat formal runs.
"""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, shutil, signal, subprocess, sys, time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import sg_worker as common
import sg_clock

SYSTEMS={'gunrock':('bfs','sssp','wcc','pr'),'cggraph':('bfs','sssp','wcc','pr'),
         'sygraph':('bfs','sssp','wcc'),'grapin':('bfs','sssp','wcc','pr'),'hytgraph':('bfs','sssp','wcc','pr')}
AUTHOR_COMMITS={'cggraph_legacy_full':'3b23a7d7384fab283952c84ae60e491d58e50ccd','cggraph_v15':'5945816f4ccf63fce8d13eac84c09bbbb4c7e58a','gunrock_essentials':'748f79e66cc26762cd3f99422b028e742a737bde','gunrock_legacy_cc':'0c9a96dd64ddeecae9208644631983fc889b32b4','sygraph_author':'71dcb56aed0cd43fefa5752a552e0a10cb69a758','grapin_author':'7ffcb2949837ad9f05d5d77afe54c698e2887479','hytgraph_author':'3e5b0e47035972cd8ba1e0a9c89d99c108041de9'}
SELECT={'gunrock':{'bfs':'gunrock','sssp':'gunrock','pr':'gunrock','wcc':'gunrock_wcc_v2'},
        'cggraph':{'bfs':'cggraph','sssp':'cggraph','wcc':'cggraph_wcc_coverage_v3','pr':'cggraph_pr_coverage_v3'},
        'sygraph':{'bfs':'sygraph_v6','sssp':'sygraph_v6','wcc':'sygraph_coverage_v2'},
        'grapin':{'bfs':'grapin_pma_boundary_v1/static','pr':'grapin_pma_boundary_v1/static',
                  'wcc':'grapin_pma_boundary_v1/cc','sssp':'grapin_static_sssp_v1'}}


def artifact(a):
    if a.system=='cggraph':unit='cggraph_v15' if a.variant=='cggraph_v15' else 'cggraph_legacy_pr' if a.algorithm=='pr' else 'cggraph_legacy_integer'
    elif a.system=='gunrock':unit='gunrock_legacy_cc' if a.algorithm=='wcc' else 'gunrock_essentials'
    elif a.system=='sygraph':unit='sygraph_cc' if a.algorithm=='wcc' else 'sygraph_traversal'
    elif a.system=='grapin':unit='grapin_cc' if a.algorithm=='wcc' else 'grapin_sssp' if a.algorithm=='sssp' else 'grapin_static'
    else:unit='hytgraph'
    receipt=a.clock_build_root/unit/'receipt.json';raw=json.loads(receipt.read_text())
    if raw.get('status')!='PASS_BUILD_ONLY' or raw.get('algorithm_changed') is not False:raise ValueError('Missing frozen observation-only author build')
    key='cc' if a.algorithm=='wcc' else a.algorithm
    item=raw['binaries'].get(key) or raw['binaries'].get('native')
    if item is None:raise ValueError('Missing fixed function binary')
    binary=common.binding(item['path'])
    if binary['sha256']!=item['sha256']:raise ValueError('Author binary hash changed')
    header=common.binding(raw['header']['path'])
    if header['sha256']!=common.binding(HERE/'sg_complete_clock.h')['sha256']:raise ValueError('Unreviewed monotonic observation header')
    return {'build':common.binding(receipt),'binary':binary,'source_commit':AUTHOR_COMMITS[a.variant],'variant':a.variant,
            'clock_header':header,'source_provenance':'Exact v2 build plus observation-only CLOCK_MONOTONIC completion hook; receipts bind every patch and compiler command'}


def environment(a,out=None):
    threads='16' if a.system=='cggraph' else '1'
    env={'CUDA_VISIBLE_DEVICES':a.gpu_uuid,'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
         'OMP_NUM_THREADS':threads,'OMP_THREAD_LIMIT':threads,'TBB_NUM_THREADS':threads,
         'OMP_DYNAMIC':'FALSE','OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0',
         'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HWLOC_COMPONENTS':'-gl'}
    dirs=['/usr/local/cuda/lib64','/usr/lib/x86_64-linux-gnu']
    if a.system in ('grapin','hytgraph','sygraph'):
        dirs=[str(a.root/'envs/cuda128/targets/x86_64-linux/lib'),'/usr/lib/x86_64-linux-gnu']
    if a.system=='sygraph':
        dirs=[str(a.root/'envs/sycl/lib'),str(a.root/'envs/sycl/targets/x86_64-linux/lib')]+dirs
        env.update(ACPP_VISIBILITY_MASK='cuda',HIPSYCL_VISIBILITY_MASK='cuda')
    # AdaptiveCpp requires CXXABI_1.3.15 from its own frozen environment. A
    # system-libstdc++ preload overrides LD_LIBRARY_PATH and prevents startup.
    library=a.root/'envs/sycl/lib/libstdc++.so.6' if a.system=='sygraph' else Path('/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
    env.update(LD_LIBRARY_PATH=':'.join(dirs),LD_PRELOAD=str(library))
    return env


def epoch(a,build):
    return {'protocol':common.PROTOCOL,'worker':common.content_binding(__file__),
            'common_worker':common.content_binding(common.__file__),'clock_helper':common.content_binding(sg_clock.__file__),'launcher':common.content_binding(HERE/'sg_clock_exec.py'),'system':a.system,'algorithm':a.algorithm,'variant':a.variant,
            'build':build,'libstdcxx':common.binding(Path(environment(a)['LD_PRELOAD']).resolve()),
            'runtime_linker':resolved_runtime(build['binary']['path'],environment(a)),
            'configuration':{'cpu_affinity':sorted(os.sched_getaffinity(0)),
             'environment':environment(a),'quality':common.QUALITY,'timeout':a.timeout,
             'hytgraph_segments':1,'hytgraph_streams':1,'hytgraph_hybrid':2,
             'grapin_hybrid':0,'grapin_cache':'all canonical arcs','pr_error':1e-7,
             'cggraph_engine':3,'cggraph_gpu_memory':0,'three_calls':'first,warmup,ready'},
            'setup':common.content_binding(a.setup),'source_count':3 if a.algorithm in ('bfs','sssp') else 1,
            'native_outputs':'Complete owning host result per call, archived after its native timer; no answer cache',
            'timing':'T_first = one CLOCK_MONOTONIC parent-before-layout to author-first-full-owning-host-output interval, including required serialization/exec/format preparation. T_ready = native complete call 2. Old native first/composite sums are decomposition only. Hashing/archive/oracle excluded.'}


def resolved_runtime(binary,settings):
    """Trace the ELF loader without entering main or invoking CUDA."""
    env=dict(os.environ,**settings);env['CUDA_VISIBLE_DEVICES']=''
    output=subprocess.check_output(['ldd',str(binary)],text=True,stderr=subprocess.STDOUT,
                                   env=env)
    if 'not found' in output:raise ValueError('Unresolved author runtime library: '+output)
    libraries={}
    for line in output.splitlines():
        fields=line.strip().split()
        if not fields:continue
        if '=>' in fields:
            name=fields[0];path=fields[fields.index('=>')+1]
        else:path=fields[0];name=Path(path).name
        if path.startswith('/') and Path(path).is_file():libraries[name]=common.binding(Path(path).resolve())
    expected=Path(settings['LD_PRELOAD']).resolve()
    if not any(Path(row['path'])==expected for row in libraries.values()):raise ValueError('ELF loader did not resolve the frozen libstdc++ preload')
    return {'mechanism':'ldd/LD_TRACE_LOADED_OBJECTS under exact library environment; no main/CUDA invocation',
            'libraries':libraries,'libstdcxx_resolved':common.binding(expected)}


def graph_layout(graph,system,algorithm,legacy=False):
    """Prepare the author-required CPU layout; this interval belongs to T_first."""
    import numpy as np
    n=graph['n'];off=graph['offsets'];ind=graph['indices'];weights=graph['weights']
    if algorithm=='wcc' and graph['directed']:
        matrix=common.graph_matrix(graph);weak=matrix.maximum(matrix.T).tocsr();weak.sort_indices()
        off=weak.indptr.astype('<i4',copy=False);ind=weak.indices.astype('<i4',copy=False);weights=np.ones(len(ind),dtype='<f8')
    if len(ind)>=2**31:raise OverflowError('INDEX_UNSUPPORTED: weak projection exceeds int32')
    if algorithm=='sssp' and system in ('cggraph','grapin','hytgraph') and (n-1)*int(np.max(weights,initial=0))>=2**32-1:
        raise OverflowError('Author uint32 distance domain cannot represent the frozen graph; no weights changed')
    layout={'n':n,'m':len(ind),'offsets':off,'indices':ind,'weights':weights}
    if system=='cggraph' and (legacy or algorithm in ('wcc','pr')):
        degree=np.diff(off);order=np.r_[np.flatnonzero(degree>0),np.flatnonzero(degree==0)].astype('<u4')
        inverse=np.empty(n,dtype='<i4');inverse[order]=np.arange(n,dtype='<i4')
        matrix=common.graph_matrix(dict(graph,m=len(ind),offsets=off,indices=ind,weights=weights),weighted=algorithm=='sssp')
        mapped=matrix[order][:,order].tocsr();mapped.sort_indices()
        layout.update(offsets=mapped.indptr.astype('<i4',copy=False),indices=mapped.indices.astype('<i4',copy=False),
                      weights=mapped.data.astype('<f8') if algorithm=='sssp' else np.ones(mapped.nnz,dtype='<f8'),order=order,degrees=degree[order].astype('<u8'))
    if system=='cggraph':layout['weights_u32']=layout['weights'].astype('<u4')
    return layout


def write_layout(layout,stage,system,algorithm):
    """Required process-boundary serialization, inside continuous T_first."""
    import numpy as np
    stage.mkdir(parents=True,exist_ok=False);n=layout['n'];off=layout['offsets'];ind=layout['indices']
    if system in ('hytgraph','grapin'):
        with (stage/'graph.edges').open('x') as f:
            for first in range(0,layout['m'],1<<18):
                last=min(first+(1<<18),layout['m']);pos=np.arange(first,last,dtype=np.int64)
                src=np.searchsorted(off,pos,side='right')-1
                np.savetxt(f,np.column_stack((src,ind[first:last],layout['weights'][first:last].astype('<u4'))),fmt='%d')
        (stage/'empty_updates.txt').write_bytes(b'')
    elif system=='gunrock' and algorithm=='wcc':
        count=0
        for first in range(0,layout['m'],1<<18):
            last=min(first+(1<<18),layout['m']);src=np.searchsorted(off,np.arange(first,last,dtype=np.int64),side='right')-1
            count+=int(np.count_nonzero(src<ind[first:last]))
        with (stage/'graph.mtx').open('x') as f:
            f.write('%%MatrixMarket matrix coordinate pattern symmetric\n');f.write(f'{n} {n} {count}\n')
            for first in range(0,layout['m'],1<<18):
                last=min(first+(1<<18),layout['m']);src=np.searchsorted(off,np.arange(first,last,dtype=np.int64),side='right')-1
                dst=ind[first:last];keep=src<dst;np.savetxt(f,np.column_stack((src[keep]+1,dst[keep]+1)),fmt='%d')
    else:
        for name in ('offsets','indices','weights','weights_u32','order','degrees'):
            if name in layout:layout[name].tofile(stage/(name+'.bin'))
        np.array([0],dtype='<u4').tofile(stage/'sources.bin')
    return sorted(stage.iterdir())


def invocation(a,build,layout,stage,out,source):
    exe=build['binary']['path'];prefix=out/'output';n,m=layout['n'],layout['m'];fn=a.algorithm
    env=environment(a);trace=None;marker='NATIVE_JSON ';pattern='output.{i}.f64';dtype='<f8'
    legacy=a.system=='cggraph' and getattr(a,'variant',None)=='cggraph_legacy_full'
    if a.system in ('gunrock','cggraph','sygraph') and fn in ('bfs','sssp','pr') and not (a.system=='cggraph' and (fn=='pr' or legacy)):
        lines=[f'G target {n} {m} {stage}']+[f'C {i} target {fn} {source or 0} 0.85 1e-12' for i in range(3)]
        if a.system=='gunrock':lines.append('Q')
        trace=out/'trace.tsv';trace.write_text('\n'.join(lines)+'\n')
        command=[exe,str(prefix)] if a.system=='gunrock' else [exe,str(trace),str(prefix)]
    elif a.system=='cggraph':
        marker='RECENT_CG_JSON '
        command=[exe,'--graphName=recent',f'--nodes={n}',f'--entries={m}','--inputPrefix='+str(stage),'--output='+str(prefix),
                 '--algorithm='+str({'bfs':0,'sssp':1,'wcc':2,'pr':3}[fn]),'--computeEngine=3',
                 '--root='+str(10000 if fn=='pr' else int(__import__('numpy').flatnonzero(layout['order']==source)[0]) if fn in ('bfs','sssp') else 0),
                 '--runs=3','--useDeviceId=0','--gpuMemory=0']
    elif a.system=='sygraph':
        marker='COVERAGE_JSON ';command=[exe,str(n),str(m),str(stage),'CC','1','3',str(prefix)]
    elif a.system=='gunrock':
        marker='PANEL_JSON ';pattern='values-{i}.i64';dtype='<i8';env['EGGPU_PANEL_OUTPUT_PREFIX']=str(out/'values')
        command=[exe,'market',str(stage/'graph.mtx'),'--quick','--quiet','--iteration-num=3','--device=0']
    elif a.system=='hytgraph':
        marker='HYTGRAPH_COMPLETE_JSON ';env.update(EGGPU_HYTGRAPH_NODES=str(n),EGGPU_HYTGRAPH_OUTPUT=str(out))
        command=[exe,'--format=market_big','--graphfile='+str(stage/'graph.edges'),'--weight_num=0','--SEGMENT=1',
                 '--n_stream=1','--hybrid=2','--max_iteration='+str(max(2000,n)),'--verbose=false','--check=false']
        if fn in ('bfs','sssp'):command+=['--source_node='+str(source)]
        if fn=='pr':command+=['--error=1e-7']
    else:
        marker='GRAPIN_STATIC_JSON ';env.update(EGGPU_STATIC_OUTPUT=str(out),EGGPU_STATIC_SOURCES=','.join([str(source or 0)]*3))
        command=[exe,'--format=market_big','--graphfile='+str(stage/'graph.edges'),'--eggpu_nodes='+str(n),'--eggpu_cache_edges='+str(m),
                 '--weight_num=1','--weight='+('false' if fn=='bfs' else 'true'),'--SEGMENT=1','--n_stream=1','--wl_alloc_factor=4',
                 '--hybrid=0','--max_iteration='+str(max(2000,n) if fn!='pr' else 2000),
                 '--updatefile='+str(stage/'empty_updates.txt')]
        if fn in ('bfs','sssp'):command+=['--source_node='+str(source or 0)]
        if fn=='sssp':command+=['--update_size='+str(stage/'empty_updates.txt')]
        if fn=='pr':command+=['--error=1e-7']
    return dict(argv=command,environment=env,stdin=str(trace) if a.system=='gunrock' and fn!='wcc' else None,
                marker=marker,output_pattern=pattern,output_dtype=dtype)


def native_times(a,rows,log):
    """Only sum disjoint native wall intervals. Keep all native raw fields."""
    first=rows[0]
    if a.system=='cggraph' and (getattr(a,'variant',None)=='cggraph_legacy_full' or a.algorithm in ('wcc','pr')):
        start=first['backend_initialization_seconds']+first['init_seconds']+first['complete_call_seconds']
        ready=rows[2]['complete_call_seconds']
    elif a.system=='sygraph' and a.algorithm=='wcc':start=first['first_request_seconds'];ready=rows[2]['query_complete_host_seconds']
    elif a.system=='gunrock' and a.algorithm=='wcc':
        events=[json.loads(x.split('CAMPAIGN_CC_FIRST_JSON ',1)[1]) for x in log.splitlines() if 'CAMPAIGN_CC_FIRST_JSON ' in x]
        if len(events)!=3 or events[0]['round']!=0:raise ValueError('Missing complete first-request observation patch')
        start=events[0]['first_request_seconds'];ready=rows[2]['complete_query_seconds']
    elif a.system in ('grapin','hytgraph'):start=first['first_request_seconds'];ready=rows[2]['query_complete_host_seconds']
    else:
        start=first['host_csr_read_seconds']+first['runtime_init_seconds']+first['complete_call_seconds'];ready=rows[2]['complete_call_seconds']
        if a.system=='gunrock':
            registrations=[json.loads(x.split('REGISTER_JSON ',1)[1]) for x in log.splitlines() if 'REGISTER_JSON ' in x]
            if len(registrations)!=1:raise ValueError('Missing graph registration timing')
            start+=registrations[0]['host_csr_read_seconds']+registrations[0]['state_initialization_seconds']
    if not 0<ready<=a.timeout or not 0<start<=a.timeout:raise ValueError('Invalid or timed-out complete request')
    return {'first':start,'ready':ready}


def execute_native(spec,out,timeout,start_ns,parent_domain):
    env=dict(os.environ,**spec['environment']);env.update(TMPDIR=str(out),PYTHONDONTWRITEBYTECODE='1',EGGPU_SG_START_NS=str(start_ns),EGGPU_SG_LAUNCH=str(out/'launch.json'))
    stream=open(spec['stdin'],'rb') if spec['stdin'] else subprocess.DEVNULL
    try:
        with (out/'native.log').open('wb') as log:
            process=subprocess.Popen([sys.executable,'-B',str(HERE/'sg_clock_exec.py'),*spec['argv']],stdin=stream,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=out,start_new_session=True)
            try:returncode=process.wait(timeout=3*timeout+60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise TimeoutError('Author three-call process deadline; killed only this owned process group')
        end_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        if returncode:raise RuntimeError('Author executable returned '+str(returncode))
        return sg_clock.first_result(out,start_ns,end_ns,parent_domain)
    finally:
        if spec['stdin']:stream.close()


def run(a):
    if a.algorithm not in SYSTEMS[a.system]:raise ValueError('This frozen author revision has no requested algorithm')
    a.variant=a.variant or ('cggraph_legacy_full' if a.system=='cggraph' else 'gunrock_legacy_cc' if a.system=='gunrock' and a.algorithm=='wcc' else 'gunrock_essentials' if a.system=='gunrock' else a.system+'_author')
    if a.system=='cggraph' and (a.variant not in ('cggraph_legacy_full','cggraph_v15') or (a.variant=='cggraph_v15' and a.algorithm not in ('bfs','sssp'))):raise ValueError('CGgraph V1.5 is a separate BFS/SSSP-only variant')
    if a.system=='gunrock' and a.variant!=('gunrock_legacy_cc' if a.algorithm=='wcc' else 'gunrock_essentials'):raise ValueError('Gunrock CC and Essentials have distinct source lineages')
    if a.system not in ('cggraph','gunrock') and a.variant!=a.system+'_author':raise ValueError('Unknown fixed author source variant')
    if not a.execute_gpu:raise ValueError('Only the primary agent may pass --execute-gpu')
    if (a.dataset in common.TINY_DATASETS)!=(a.phase=='tiny'):raise ValueError('Tiny/formal input scope mismatch')
    if a.repeat not in range(10):raise ValueError('Repeat must be 0..9')
    if a.cpus:
        selected={int(x) for x in a.cpus.split(',')}
        if len(selected)!=16 or not selected<=os.sched_getaffinity(0):raise ValueError('Exactly 16 available CPU IDs required')
        os.sched_setaffinity(0,selected)
    if len(os.sched_getaffinity(0))!=16:raise ValueError('Freeze exactly sixteen CPU IDs through --cpus or outer taskset')
    import numpy as np
    graph=common.load_input(a.input_root/'canonical'/a.dataset);out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    record={'schema':'eggpu.e1.sg.author.v1','protocol':common.PROTOCOL,'phase':a.phase,'system':a.system,'algorithm':a.algorithm,
            'dataset':a.dataset,'variant':a.variant,'repeat':a.repeat,'gpu_count':1,'gpu_executed':False,'status':'RUNNING','quality_pass':False,
            'performance_eligible':False,'input_sha256':graph['input_sha256'],'directed':graph['directed'],'sources':[]}
    try:
        build=artifact(a);ep=epoch(a,build);key=common.digest_json(ep);common.save(out/'epoch.json',ep)
        record.update(epoch=common.binding(out/'epoch.json'),epoch_sha256=key)
        if a.phase in ('target','formal'):
            if a.tiny_proof is None:raise ValueError('Need explicit new-source tiny proof')
            record['tiny_proof']=common.validate_proof(a.tiny_proof,kind='tiny',system=a.system,algorithm=a.algorithm,epoch_sha=key,graph=graph)
        if a.phase=='formal':
            if a.target_proof is None:raise ValueError('Need explicit matching target proof')
            record['target_proof']=common.validate_proof(a.target_proof,kind='target',system=a.system,algorithm=a.algorithm,epoch_sha=key,graph=graph)
        with a.lock_file.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for index in range(ep['source_count']):
                source=graph['sources'][index] if ep['source_count']==3 else None;child=out/f'source_{index}';child.mkdir()
                record['resources']=common.resource_check(a,graph)
                if shutil.disk_usage(out).free<48*graph['m']+72*graph['n']+(2<<30):raise RuntimeError('Insufficient baseline staging/output disk reserve')
                domain=sg_clock.clock_domain();start_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                begin=time.perf_counter();layout=graph_layout(graph,a.system,a.algorithm,a.variant=='cggraph_legacy_full');layout_seconds=time.perf_counter()-begin
                begin=time.perf_counter();staged=write_layout(layout,child/'stage',a.system,a.algorithm);bridge_seconds=time.perf_counter()-begin
                spec=invocation(a,build,layout,child/'stage',child,source);record['gpu_executed']=True
                continuous=execute_native(spec,child,a.timeout,start_ns,domain)
                if continuous['values']!=graph['n']:raise ValueError('Native first endpoint lacks the full vertex domain')
                common.save(child/'invocation.json',dict(spec,staged=[common.binding(p) for p in staged],layout_seconds=layout_seconds,bridge_serialization_seconds=bridge_seconds))
                log=(child/'native.log').read_text()
                rows=[json.loads(x.split(spec['marker'],1)[1]) for x in log.splitlines() if spec['marker'] in x]
                if len(rows)!=3 or [r.get('round',r.get('sequence')) for r in rows]!=[0,1,2]:raise ValueError('Missing exactly three complete native calls')
                times=native_times(a,rows,log);expected,reference=common.reference_cached(graph,a.algorithm,source,a.reference_cache)
                calls={}
                for i,label in enumerate(('first','warmup','ready')):
                    raw=child/spec['output_pattern'].format(i=i);values=np.fromfile(raw,dtype=spec['output_dtype'])
                    if a.algorithm=='wcc' and values.dtype!=np.float64:values=values.astype('<f8')
                    values.setflags(write=False)
                    calls[label]={'timed':label!='warmup','native':rows[i],
                                  'output':common.persist_output(child,values),'quality':common.audit_output(values,expected,graph,a.algorithm),
                                  'native_output_sha256':common.binding(raw)['sha256']}
                    if label=='first':calls[label].update(seconds=continuous['seconds'],continuous_interval=continuous,native_first_decomposition_seconds=times['first'],legacy_layout_plus_native_first_seconds=layout_seconds+times['first'])
                    elif label=='ready':calls[label]['seconds']=times['ready']
                    raw.unlink() # Exact values remain once in the content-addressed .npy above.
                passed=all(x['quality']['pass'] for x in calls.values())
                result={'status':'PASS' if passed else 'FAIL_QUALITY','quality_pass':passed,'source':source,'calls':calls,
                        'continuous_first':continuous,'reference_receipt':reference,'layout_seconds':layout_seconds,'bridge_serialization_seconds':bridge_seconds,
                        'outputs_survive_native_teardown':True,'input_sha256':graph['input_sha256'],'invocation':common.binding(child/'invocation.json')}
                common.save(child/'result.json',result)
                record['sources'].append({'index':index,'source':source,'status':result['status'],'quality_pass':passed,
                                          'result':common.binding(child/'result.json'),'calls':calls})
                shutil.rmtree(child/'stage') # Only this worker's derived bridge, never the canonical files.
                if not passed:raise ValueError('Common complete-output quality failed; timing is not eligible')
            record.update(status='PASS' if a.phase=='formal' else 'PASS_QUALIFICATION',quality_pass=True,performance_eligible=a.phase=='formal')
    except BaseException as error:
        record.update(status='TIMEOUT' if isinstance(error,TimeoutError) else 'FAILED',error=type(error).__name__+': '+str(error))
        raise
    finally:common.save(out/'result.json',record)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-gpu',action='store_true')
    p.add_argument('--system',choices=tuple(SYSTEMS),required=True);p.add_argument('--algorithm',choices=common.ALGORITHMS,required=True)
    p.add_argument('--dataset',choices=common.DATASETS+common.TINY_DATASETS,required=True);p.add_argument('--phase',choices=('tiny','target','formal'),required=True)
    p.add_argument('--repeat',type=int,default=0);p.add_argument('--timeout',type=int,default=3600)
    for key in ('root','input-root','setup','reference-cache','lock-file','output','clock-build-root'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--cpus');p.add_argument('--variant');p.add_argument('--tiny-proof',type=Path);p.add_argument('--target-proof',type=Path)
    p.add_argument('--min-host-free-gib',type=float,default=8.);p.add_argument('--budget-mib',type=int,default=0)
    return p


if __name__=='__main__':run(parser().parse_args())
