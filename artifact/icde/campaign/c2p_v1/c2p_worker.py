#!/usr/bin/env python3
"""E1 C2graph-P single-source SSSP with continuous complete-host timing.

Each canonical primary source gets a fresh author graph, compression included
in T_first; compressed state is reused for three independently solved requests.
This is the P executor, never a divided PM multi-query batch duration.
"""
from pathlib import Path
import argparse,fcntl,json,os,shutil,signal,subprocess,sys,time
import sg_worker as common
import sg_clock
HERE=Path(__file__).resolve().parent
COMMIT='97af1ebd935a03f5adc180935644e786a624bdbe'


def environment(args):
    return {'CUDA_VISIBLE_DEVICES':args.gpu_uuid,'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
        'OMP_NUM_THREADS':'16','OMP_THREAD_LIMIT':'16','OMP_DYNAMIC':'FALSE',
        'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TBB_NUM_THREADS':'16',
        'LD_PRELOAD':'/usr/lib/x86_64-linux-gnu/libstdc++.so.6',
        'LD_LIBRARY_PATH':'/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu'}


def artifact(args):
    raw=json.loads(args.author_build.read_text())
    if raw.get('status')!='PASS_BUILD_ONLY' or raw.get('variant')!='c2graph_p' or raw.get('source_commit')!=COMMIT:
        raise ValueError('Exact single-source author P build required')
    items={}
    for key in ('binary','compressor','header'):
        items[key]=common.binding(raw[key]['path'])
        if items[key]!=raw[key]:raise ValueError('Frozen P build file changed: '+key)
    if items['header']['sha256']!=common.binding(HERE/'sg_complete_clock.h')['sha256']:
        raise ValueError('Unreviewed complete-output clock observation')
    return {'build':common.binding(args.author_build),'source_commit':COMMIT,'variant':'c2graph_p',**items}


def epoch(args,build):
    files={name:common.content_binding(HERE/name) for name in
        ('c2p_worker.py','sg_worker.py','sg_clock.py','sg_clock_exec.py','sg_complete_clock.h')}
    return {'protocol':common.PROTOCOL,'author_timing_protocol':'single-source-P-continuous-monotonic-v1',
        'system':'c2graph','variant':'c2graph_p','algorithm':'sssp','build':build,'files':files,
        'setup':common.content_binding(args.setup),'quality':common.QUALITY,'environment':environment(args),
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'timeout':args.timeout,
        'semantics':'Canonical positive w16 CSR, full FP64/+inf distances, three distinct primary sources, independent solve on every call',
        'timing':'Continuous canonical CSR before conversion/compression/exec to first complete owning host output. T_ready=third complete native solve; output archive/oracle excluded.',
        'author_configuration':{'compression':'unchanged author purn-sssp','blocks':20,'threads_per_block':512,
            'source_compensation':'existing source-keyed H transitive closure; cost inside each query','calls':['first','warmup','ready']}}


def origin_layout(graph,stage):
    import numpy as np
    if (graph['n']-1)*16>=1999999999:raise OverflowError('Frozen graph exceeds author exact finite distance sentinel')
    stage=Path(stage);(stage/'origin').mkdir(parents=True,exist_ok=False);(stage/'sssp').mkdir()
    directory=Path(graph['directory'])
    for name,role in [('vlist.bin','offsets'),('elist.bin','indices')]:
        (stage/'origin'/name).symlink_to((directory/(role+'.bin')).resolve())
    # Frozen w16 has exact positive integer values. No random author weights.
    weights=graph['weights'].astype('<i4')
    if not np.array_equal(weights,graph['weights']) or np.any(weights<=0):raise ValueError('Unsupported native weight domain')
    weights.tofile(stage/'origin/wlist.bin')


def validate_compression(stage,n):
    """CPU admission/audit after completion; no vectors from the GPU are used."""
    import numpy as np
    records={}
    for prefix in ('','h'):
        paths=[Path(stage)/'sssp'/(prefix+name+'.bin') for name in ('vlist','elist','wlist')]
        row=np.fromfile(paths[0],dtype='<i4');col=np.fromfile(paths[1],dtype='<i4');weights=np.fromfile(paths[2],dtype='<i4')
        if (len(row)!=n+1 or row[0]!=0 or row[-1]!=len(col) or len(col)!=len(weights) or
                np.any(row[1:]<row[:-1]) or np.any(col<0) or np.any(col>=n) or np.any(weights<=0)):
            raise ValueError('Author compression changed full vertex/edge-weight domain')
        records['compressed' if not prefix else 'source_compensation']={
            'vertices':n,'arcs':len(col),'files':[common.binding(p) for p in paths]}
    return records


def invocation(args,build,graph,stage,out,source):
    trace=out/'trace.tsv';trace.write_text('\n'.join([f"G target {graph['n']} {graph['m']} {stage}"]+
        [f'C {i} target sssp {source} 0.85 1e-12' for i in range(3)])+'\n')
    return {'argv':[build['binary']['path'],str(trace),str(out/'output')],
        'environment':environment(args),'single_source':source,'calls':3,'variant':'P'}


def execute_native(args,spec,out,start_ns,domain):
    env=dict(os.environ,**spec['environment'],EGGPU_SG_START_NS=str(start_ns),EGGPU_SG_LAUNCH=str(out/'launch.json'))
    process=None
    try:
        with (out/'native.log').open('wb') as log:
            process=subprocess.Popen([sys.executable,'-B',str(HERE/'sg_clock_exec.py'),*spec['argv']],
                stdout=log,stderr=subprocess.STDOUT,env=env,cwd=out,start_new_session=True)
            code=process.wait(timeout=3*args.timeout+60)
        ended=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        if code:raise RuntimeError('C2graph-P returned '+str(code))
        result=sg_clock.first_result(out,start_ns,ended,domain)
        if result['seconds']>args.timeout:raise TimeoutError('Complete C2graph-P first request exceeded timeout')
        return result
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait()


def run(args):
    if not args.execute_gpu:raise ValueError('Only the primary agent may pass --execute-gpu')
    if (args.dataset in common.TINY_DATASETS)!=(args.phase=='tiny') or args.repeat not in range(10):
        raise ValueError('Tiny/target/formal input or repetition scope differs')
    cpus={int(x) for x in args.cpus.split(',')}
    if len(cpus)!=16 or not cpus<=os.sched_getaffinity(0):raise ValueError('Exactly 16 available CPU IDs required')
    os.sched_setaffinity(0,cpus)
    import numpy as np
    graph=common.load_input(args.input_root/'canonical'/args.dataset)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    record={'status':'RUNNING','schema':'eggpu.e1.c2graph_p.v1','protocol':common.PROTOCOL,
        'phase':args.phase,'system':'c2graph','variant':'c2graph_p','algorithm':'sssp','dataset':args.dataset,
        'directed':graph['directed'],'input_sha256':graph['input_sha256'],'repeat':args.repeat,
        'gpu_count':1,'gpu_executed':False,'quality_pass':False,'performance_eligible':False,'sources':[]}
    try:
        build=artifact(args);entry=epoch(args,build);key=common.digest_json(entry);common.save(out/'epoch.json',entry)
        record.update(epoch=common.binding(out/'epoch.json'),epoch_sha256=key)
        if args.phase in ('target','formal'):
            record['tiny_proof']=common.validate_proof(args.tiny_proof,kind='tiny',system='c2graph',algorithm='sssp',epoch_sha=key,graph=graph)
        if args.phase=='formal':
            record['target_proof']=common.validate_proof(args.target_proof,kind='target',system='c2graph',algorithm='sssp',epoch_sha=key,graph=graph)
        with args.lock_file.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for index,source in enumerate(graph['sources']):
                child=out/('source_'+str(index));child.mkdir()
                record['resources']=common.resource_check(args,graph)
                if shutil.disk_usage(out).free<64*graph['m']+96*graph['n']+(2<<30):
                    raise RuntimeError('Insufficient author compressed/preparation/output disk reserve')
                domain=sg_clock.clock_domain();start_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                stage=child/'stage';origin_layout(graph,stage);converted_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                with (child/'compression.log').open('wb') as log:
                    compressed=subprocess.run([build['compressor']['path'],str(stage)],stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,**dict(environment(args),CUDA_VISIBLE_DEVICES='')),timeout=args.timeout)
                if compressed.returncode:raise RuntimeError('Author compression returned '+str(compressed.returncode))
                compressed_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                spec=invocation(args,build,graph,stage,child,source);record['gpu_executed']=True
                continuous=execute_native(args,spec,child,start_ns,domain)
                if continuous['values']!=graph['n']:raise ValueError('P output does not cover every canonical vertex')
                rows=[json.loads(line.split('NATIVE_JSON ',1)[1]) for line in (child/'native.log').read_text().splitlines() if 'NATIVE_JSON ' in line]
                if len(rows)!=3 or [row['sequence'] for row in rows]!=[0,1,2]:raise ValueError('Missing three independent P solves')
                compressed_record=validate_compression(stage,graph['n'])
                common.save(child/'invocation.json',dict(spec,compression=compressed_record))
                expected,reference=common.reference_cached(graph,'sssp',source,args.reference_cache)
                calls={}
                for i,label in enumerate(('first','warmup','ready')):
                    path=child/('output.'+str(i)+'.f64');values=np.fromfile(path,dtype='<f8');values.flags.writeable=False
                    calls[label]={'timed':label!='warmup','native':rows[i],
                        'output':common.persist_output(child,values),'quality':common.audit_output(values,expected,graph,'sssp'),
                        'native_output_sha256':common.binding(path)['sha256']}
                    if label=='first':calls[label].update(seconds=continuous['seconds'],continuous_interval=continuous)
                    if label=='ready':
                        seconds=rows[i]['complete_call_seconds']
                        if not 0<seconds<=args.timeout:raise TimeoutError('Complete ready P query exceeded timeout')
                        calls[label]['seconds']=seconds
                    path.unlink()
                passed=all(row['quality']['pass'] for row in calls.values())
                result={'status':'PASS' if passed else 'FAIL_QUALITY','quality_pass':passed,'source':source,'calls':calls,
                    'continuous_first':continuous,'layout_seconds':(converted_ns-start_ns)/1e9,
                    'compression_seconds':(compressed_ns-converted_ns)/1e9,'reference_receipt':reference,
                    'compressed':compressed_record,'outputs_survive_native_teardown':True}
                common.save(child/'result.json',result)
                record['sources'].append({'index':index,'source':source,'status':result['status'],'quality_pass':passed,
                    'result':common.binding(child/'result.json'),'calls':calls})
                shutil.rmtree(stage)
                if not passed:raise ValueError('C2graph-P complete-output quality failed')
            record.update(status='PASS' if args.phase=='formal' else 'PASS_QUALIFICATION',quality_pass=True,performance_eligible=args.phase=='formal')
    except BaseException as error:
        record.update(status='TIMEOUT' if isinstance(error,(TimeoutError,subprocess.TimeoutExpired)) else 'FAILED',error=type(error).__name__+': '+str(error));raise
    finally:common.save(out/'result.json',record)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-gpu',action='store_true')
    p.add_argument('--dataset',choices=common.DATASETS+common.TINY_DATASETS,required=True)
    p.add_argument('--phase',choices=('tiny','target','formal'),required=True);p.add_argument('--repeat',type=int,default=0)
    for key in ('root','input-root','author-build','setup','reference-cache','lock-file','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--tiny-proof',type=Path);p.add_argument('--target-proof',type=Path)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--cpus',required=True)
    p.add_argument('--timeout',type=int,default=3600);p.add_argument('--min-host-free-gib',type=float,default=8.)
    p.add_argument('--budget-mib',type=int,default=0)
    return p


if __name__=='__main__':run(parser().parse_args())
