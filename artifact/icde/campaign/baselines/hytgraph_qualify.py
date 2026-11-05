#!/usr/bin/env python3
"""Prepare tiny complete-output contracts, or run them under the primary GPU lock.

`plan` is CPU-only. `run` executes GPU work and is reserved for the primary agent.
Neither mode creates formal seven-graph performance evidence.
"""
from pathlib import Path
import argparse, fcntl, hashlib, heapq, json, math, os, signal, struct, subprocess, time

N=9
EDGES=[(0,1,2),(1,2,3),(0,3,9),(2,3,1),(3,4,4),(4,5,1),(5,3,2),(6,7,2),(7,6,1)]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,r):Path(p).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')


def distances(n,edges,source,unit=False):
    adj=[[] for _ in range(n)]
    for u,v,w in edges:adj[u].append((v,1 if unit else w))
    d=[math.inf]*n;d[source]=0;q=[(0,source)]
    while q:
        du,u=heapq.heappop(q)
        if du!=d[u]:continue
        for v,w in adj[u]:
            if du+w<d[v]:d[v]=du+w;heapq.heappush(q,(d[v],v))
    return d


def pr_step(x,edges):
    n=len(x);degree=[0]*n
    for u,v,w in edges:degree[u]+=1
    dangling=sum(x[i] for i in range(n) if not degree[i])
    y=[(0.15+0.85*dangling)/n]*n
    for u,v,w in edges:y[v]+=0.85*x[u]/degree[u]
    return y


def pr_reference(n,edges):
    x=[1/n]*n
    for _ in range(10000):
        y=pr_step(x,edges)
        if sum(abs(a-b) for a,b in zip(x,y))<1e-14:return y
        x=y
    raise RuntimeError('Independent PR reference did not converge')


def check_vector(alg,source,values):
    if len(values)!=N:raise ValueError('Incomplete vertex domain')
    if alg in ('bfs','sssp'):
        ref=distances(N,EDGES,source,alg=='bfs')
        bad=sum(x!=y for x,y in zip(values,ref))
        return dict(passed=bad==0,mismatch_count=bad,values_checked=N)
    if alg=='cc':
        # Labels need not equal CPU canonical IDs; equivalence relations must.
        labels=[0,0,0,0,0,0,1,1,2]
        valid=all(math.isfinite(x) and x==int(x) and 0<=x<N for x in values)
        bad=sum((values[i]==values[j])!=(labels[i]==labels[j]) for i in range(N) for j in range(N))
        return dict(passed=valid and bad==0,partition_pair_mismatches=bad,values_checked=N)
    if not all(math.isfinite(x) and x>=0 for x in values):
        return dict(passed=False,reason='Nonfinite or negative PR scores',values_checked=N)
    ref=pr_reference(N,EDGES);step=pr_step(values,EDGES)
    residual=sum(abs(a-b) for a,b in zip(values,step));l1=sum(abs(a-b) for a,b in zip(values,ref));mass=abs(sum(values)-1)
    return dict(passed=residual<=1e-6 and l1<=1e-5 and mass<=1e-6,l1_residual=residual,
                reference_l1=l1,mass_error=mass,values_checked=N,residual_mass_tolerance=1e-6,reference_tolerance=1e-5)


def plan(a):
    r=json.loads(a.build.read_text())
    if r.get('status')!='PASS_BUILD_ONLY' or set(r['binaries'])!={'bfs','sssp','cc','pr'}:
        raise ValueError('All four fixed-source binaries must be built')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);tasks=[]
    directed=out/'directed.edges';directed.write_text(''.join(f'{u} {v} {w}\n' for u,v,w in EDGES))
    weak=out/'weak.edges';arcs=sorted({(u,v) for u,v,w in EDGES}|{(v,u) for u,v,w in EDGES})
    weak.write_text(''.join(f'{u} {v} 1\n' for u,v in arcs))
    for segments,streams in [(1,1),(3,2)]:
        for alg,sources in [('bfs',[0,8]),('sssp',[0,8]),('cc',[None]),('pr',[None])]:
            for source in sources:
                tag=f'{alg}_source{source}_segments{segments}';dest=out/tag;dest.mkdir()
                binary=r['binaries'][alg];inp=weak if alg=='cc' else directed
                cmd=[binary['path'],'--graphfile='+str(inp),'--format=market_big','--weight_num=0',
                     '--SEGMENT='+str(segments),'--n_stream='+str(streams),'--hybrid=2','--max_iteration=10000',
                     '--verbose=false','--check=false']
                if source is not None:cmd+=['--source_node='+str(source)]
                if alg=='pr':cmd+=['--error=1e-7']
                tasks.append(dict(id=tag,algorithm=alg,source=source,argv=cmd,binary_sha256=binary['sha256'],
                    input_path=str(inp),input_sha256=sha(inp),output=str(dest),
                    env={'EGGPU_HYTGRAPH_NODES':str(N),'EGGPU_HYTGRAPH_OUTPUT':str(dest)}))
    record=dict(schema='hytgraph.tiny.plan/v1',status='PREPARED_NOT_RUN',n=N,edges=EDGES,
                build_path=str(a.build.resolve()),build_sha256=sha(a.build),gpu_executed=False,
                formal_performance_eligible=False,tasks=tasks,expected_calls_per_task=3,
                purpose='CPU-independent complete-vector checks, fresh repeated queries, isolated source, non-unit weights, dangling mass and multiple transfer segments.')
    dump(out/'plan.json',record);print(json.dumps({'plan':str(out/'plan.json'),'tasks':len(tasks),'gpu_executed':False}))


def resource_gate(uuid,minimum):
    proc=subprocess.run(['nvidia-smi','-i',uuid,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                        text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    fields=[v.strip() for v in proc.stdout.strip().split(',')]
    if len(fields)!=3 or fields[0].lower()!=uuid.lower() or int(fields[1])>128 or int(fields[2])>10:
        raise RuntimeError('Requested GPU is occupied or UUID mismatch: '+proc.stdout.strip())
    procs=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    if any(x.split(',')[0].strip().lower()==uuid.lower() for x in procs.stdout.splitlines()):
        raise RuntimeError('Requested GPU has a compute process')
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    # Take the smaller of host availability and the active cgroup's headroom.
    cg=Path('/sys/fs/cgroup')
    for relative in [Path('.'),*[Path(x.split(':',2)[2].lstrip('/')) for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')]]:
        limit=cg/relative/'memory.max';used=cg/relative/'memory.current'
        if limit.exists() and used.exists() and limit.read_text().strip()!='max':available=min(available,int(limit.read_text())-int(used.read_text()))
    if available<minimum*1024**3:raise RuntimeError('Insufficient real host-memory headroom')
    return dict(gpu_uuid=fields[0],memory_used_mib=int(fields[1]),gpu_utilization_percent=int(fields[2]),host_available_bytes=available)


def run(a):
    plan_path=a.plan.resolve();p=json.loads(plan_path.read_text());out=plan_path.parent
    if (out/'receipt.json').exists():raise FileExistsError('Use a new plan directory for a new qualification')
    if sha(p['build_path'])!=p['build_sha256']:raise ValueError('Build receipt changed')
    if not a.gpu_uuid.startswith('GPU-') or not a.lock.is_absolute():raise ValueError('Explicit authorized UUID and absolute shared lock required')
    checks=[];overall='PASS_TINY_GPU_ONLY';a.lock.parent.mkdir(parents=True,exist_ok=True)
    with a.lock.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for task in p['tasks']:
            if sha(task['argv'][0])!=task['binary_sha256'] or sha(task['input_path'])!=task['input_sha256']:
                raise ValueError('Frozen binary or tiny input changed')
            gate=resource_gate(a.gpu_uuid,a.minimum_host_memory_gib);dest=Path(task['output'])
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu_uuid,OMP_NUM_THREADS='2',**task['env'])
            env.pop('LD_PRELOAD',None);start=time.monotonic();failure=None
            with (dest/'run.log').open('wb') as log:
                proc=subprocess.Popen(task['argv'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:rc=proc.wait(timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    rc=124;failure='TIMEOUT'
            record=dict(id=task['id'],returncode=rc,gate=gate,process_wall_seconds=time.monotonic()-start,outputs=[])
            if rc:failure=failure or 'PROCESS_ERROR'
            else:
                try:
                    rows=[json.loads(x.split('HYTGRAPH_COMPLETE_JSON ',1)[1]) for x in (dest/'run.log').read_text().splitlines() if x.startswith('HYTGRAPH_COMPLETE_JSON ')]
                    if len(rows)!=3:raise ValueError('Expected three complete independently initialized calls')
                    for i,row in enumerate(rows):
                        if row['round']!=i or row['n']!=N or row['visible_device_count']!=1 or row['gpu_uuid'].lower()!=a.gpu_uuid.lower():raise ValueError('Execution identity mismatch')
                        for key in ('query_complete_host_seconds','backend_prepare_seconds','first_request_seconds'):
                            if not math.isfinite(row[key]) or row[key]<0:raise ValueError('Invalid timing')
                        path=dest/f'output.{i}.f64';values=struct.unpack('<'+'d'*N,path.read_bytes())
                        quality=check_vector(task['algorithm'],task['source'],values)
                        record['outputs'].append(dict(round=i,sha256=sha(path),quality=quality,timing=row))
                        if not quality['passed']:failure='QUALITY_FAILED'
                except (KeyError,ValueError,OSError,struct.error) as exc:failure=type(exc).__name__+': '+str(exc)
            record['status']='PASS' if failure is None else failure;checks.append(record)
            if failure:overall='FAIL_TINY_GPU_QUALIFICATION'
            dump(out/'receipt.json',dict(schema='hytgraph.tiny.qualification/v1',status=overall if len(checks)==len(p['tasks']) else 'IN_PROGRESS',
                 plan_sha256=sha(plan_path),gpu_uuid=a.gpu_uuid,gpu_executed=True,formal_performance_eligible=False,checks=checks))
    print(json.dumps({'status':overall,'receipt':str(out/'receipt.json')}));return int(overall!='PASS_TINY_GPU_ONLY')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('plan');p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--plan',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--lock',type=Path,required=True);p.add_argument('--timeout',type=int,default=120)
    p.add_argument('--minimum-host-memory-gib',type=float,default=4)
    a=parser.parse_args();return plan(a) if a.action=='plan' else run(a)


if __name__=='__main__':raise SystemExit(main())
