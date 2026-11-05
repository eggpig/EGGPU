#!/usr/bin/env python3
"""Fixed-author KGraph SSSP build and tiny CPU correctness qualification.

The adapter preserves partition scheduling and pivot memoization. It fixes two
source-level C++ undefined behaviors and exports complete owning host results.
No GPU work or formal performance campaign is launched by this program.
"""
from pathlib import Path
import argparse, difflib, hashlib, heapq, json, os, shutil, struct, subprocess, time
from source_lock import verify_group

COMMIT = 'e09ca53695cb5a82d1bbb2418caac90b2932af7a'
REPO = 'https://github.com/Gawssin/KGraph'
INF = (2**31 - 1) // 2


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(('Unexpected fixed source', old[:120], text.count(old)))
    return text.replace(old, new)


def build(a):
    src, out = a.source.resolve(), a.output.resolve()
    source_verification=verify_group(src,'kgraph')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    patched = out / 'source'
    shutil.copytree(src, patched, ignore=shutil.ignore_patterns('.git', 'build', 'inputs', '*.pdf'))
    changes = []

    def change(rel, text, reason):
        p = patched / rel
        before = p.read_text()
        p.write_text(text)
        changes.append(dict(path=rel, reason=reason, before_sha256=sha(src/rel), after_sha256=sha(p),
                            patch=''.join(difflib.unified_diff(before.splitlines(True), text.splitlines(True),
                                                             fromfile='a/'+rel, tofile='b/'+rel))))

    header = patched/'ligra/ligra.h'
    text = header.read_text()
    text = '#include <chrono>\n#include <fstream>\n#include <iomanip>\n#include <stdexcept>\n#include <cstdint>\n' + text
    text = replace_once(text, 'int parallel_main(int argc, char* argv[]) {',
        'double eggpu_kgraph_prepare_seconds=0;\nint parallel_main(int argc, char* argv[]) {\n'
        '  const auto eggpu_prepare_begin=std::chrono::steady_clock::now();')
    text = replace_once(text, '  Compute(Ga, Gb, NP, pivots, P);',
        '  eggpu_kgraph_prepare_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-eggpu_prepare_begin).count();\n'
        '  Compute(Ga, Gb, NP, pivots, P);')
    text = replace_once(text, '  Gb.del();\n}', '  Gb.del();\n  return 0;\n}')
    change('ligra/ligra.h', text, 'Expose file/graph/partition/pivot preparation time; return defined success code from parallel_main.')

    for variant in ('sparse', 'dense'):
        rel = f'apps/kgraph-sssp-{variant}.C'
        text = (patched/rel).read_text()
        text = replace_once(text, 'ShortestPathLen_l.reserve(k);', 'ShortestPathLen_l.resize(k);')
        marker = 'void Compute(graph<vertex> &GA, graph<vertex> &GB, int NP, std::vector<int> &pivot_l, commandLine P)\n{'
        text = replace_once(text, marker, marker+'\n  const auto eggpu_compute_begin=std::chrono::steady_clock::now();')
        marker = '  t.print_ms("exuection time");'
        export = r'''
  static_assert(sizeof(intE)==4, "Adapter requires the frozen int32 distance type");
  std::vector<std::int32_t> eggpu_owning(size_t(k)*size_t(n));
  for(long qi=0;qi<k;++qi)
    std::copy(ShortestPathLen_l[qi],ShortestPathLen_l[qi]+n,eggpu_owning.begin()+size_t(qi)*size_t(n));
  const auto eggpu_complete=std::chrono::steady_clock::now();
  const double eggpu_seconds=std::chrono::duration<double>(eggpu_complete-eggpu_compute_begin).count();
  const char* eggpu_destination=std::getenv("EGGPU_KGRAPH_OUTPUT");
  if(!eggpu_destination) throw std::runtime_error("EGGPU_KGRAPH_OUTPUT is required");
  std::ofstream eggpu_file(eggpu_destination,std::ios::binary);
  eggpu_file.write(reinterpret_cast<const char*>(eggpu_owning.data()),eggpu_owning.size()*sizeof(std::int32_t));
  if(!eggpu_file) throw std::runtime_error("Cannot archive complete KGraph output");
  std::cout<<std::setprecision(17)<<"EGGPU_KGRAPH_JSON {\"n\":"<<n
    <<",\"queries\":"<<k<<",\"pivots\":"<<pivotN
    <<",\"complete_host_seconds\":"<<eggpu_seconds
    <<",\"graph_partition_pivot_prepare_seconds\":"<<eggpu_kgraph_prepare_seconds
    <<",\"first_batch_seconds\":"<<(eggpu_kgraph_prepare_seconds+eggpu_seconds)
    <<",\"native_infinity\":"<<(INT_MAX/2)<<"}"<<std::endl;
'''
        text = replace_once(text, marker, marker+'\n'+export)
        change(rel, text, 'resize instead of reserve before indexed writes; complete owning host output and timing only; author scheduling/memoization unchanged.')

    (out/'adapter.patch').write_text(''.join(x['patch'] for x in changes))
    steps = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2')
    for variant in ('sparse','dense'):
        cmd=[a.cxx,'-std=c++14','-O3','-fopenmp','-DWEIGHTED','-DOPENMP','-DBYTERLE','-I'+str(patched/'ligra'),
             str(patched/f'apps/kgraph-sssp-{variant}.C'),'-o',str(out/f'kgraph-sssp-{variant}')]
        start=time.monotonic()
        with (out/f'build_{variant}.log').open('w') as log:
            rc=subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        steps.append(dict(argv=cmd,returncode=rc,seconds=time.monotonic()-start))
        if rc:break
    receipt=dict(schema='kgraph.adapter.build/v1',status='PASS_BUILD_ONLY' if rc==0 else 'FAIL_BUILD',
                 upstream=dict(repository=REPO,commit=COMMIT),gpu_executed=False,source=str(src),patched_source=str(patched),
                 source_verification=source_verification,changes=changes,steps=steps,binaries={v:dict(path=str(out/f'kgraph-sssp-{v}'),sha256=sha(out/f'kgraph-sssp-{v}'))
                 for v in ('sparse','dense') if (out/f'kgraph-sssp-{v}').exists()},
                 timing='First batch includes author graph/partition reading, pivot selection, memoization, query initialization/execution, complete owning host outputs. File output and reference checks excluded. No cross-batch persistent memoization claim.',
                 algorithm_changes=False,formal_performance_eligible=False)
    write(out/'receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'receipt':str(out/'receipt.json')}))
    return rc


def validate_input(n, edges, partitions, sources):
    if not 0<n<2**31 or len(partitions)!=n or not sources:
        raise ValueError('Invalid vertex or query domain')
    nparts=max(partitions)+1
    if set(partitions)!=set(range(nparts)):
        raise ValueError('Partitions must be contiguous and nonempty')
    seen=set();max_weight=0
    for u,v,w in edges:
        if not (0<=u<n and 0<=v<n and u!=v and type(w) is int and 1<=w<=16):
            raise ValueError('Expected canonical non-self positive-integral weighted arcs')
        if (u,v) in seen:raise ValueError('Duplicate arcs are not silently normalized')
        seen.add((u,v));max_weight=max(max_weight,w)
    if (n-1)*max_weight>=INF:raise ValueError('Distances may collide with native INT_MAX/2 sentinel')
    if any(type(s) is not int or not 0<=s<n for s in sources):raise ValueError('Source outside vertex domain')
    return nparts


def ligra_file(path,n,edges):
    edges=sorted(edges)
    offsets=[];j=0
    for u in range(n):
        offsets.append(j)
        while j<len(edges) and edges[j][0]==u:j+=1
    Path(path).write_text('\n'.join(map(str,['WeightedAdjacencyGraph',n,len(edges),*offsets,
                                            *[v for u,v,w in edges],*[w for u,v,w in edges]]))+'\n')


def prepare_fixture(out):
    # Non-unit weights, directed edges, three partitions, disconnected component,
    # isolated source, multiple improving routes, and a return cycle.
    n=9;parts=[0,0,0,1,1,1,2,2,2]
    edges=[(0,1,2),(1,2,3),(0,3,9),(2,3,1),(3,4,4),(4,5,1),(5,3,2),(6,7,2),(7,6,1)]
    sources=[0,3,6,8];np=validate_input(n,edges,parts,sources)
    out.mkdir(parents=True,exist_ok=False)
    ligra_file(out/'inter.adj',n,[e for e in edges if parts[e[0]]!=parts[e[1]]])
    ligra_file(out/'intra.adj',n,[e for e in edges if parts[e[0]]==parts[e[1]]])
    (out/'parts.txt').write_text('\n'.join(map(str,parts))+'\n')
    (out/'sources.txt').write_text('\n'.join(map(str,sources))+'\n')
    fixture=dict(n=n,edges=edges,partitions=parts,sources=sources,num_partitions=np)
    write(out/'fixture.json',fixture)
    return fixture


def reference(n,edges,source):
    adj=[[] for _ in range(n)]
    for u,v,w in edges:adj[u].append((v,w))
    dist=[INF]*n;dist[source]=0;q=[(0,source)]
    while q:
        d,u=heapq.heappop(q)
        if d!=dist[u]:continue
        for v,w in adj[u]:
            if d+w<dist[v]:dist[v]=d+w;heapq.heappush(q,(d+w,v))
    return dist


def qualify(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    fixture=prepare_fixture(out/'input');r=json.loads(a.build.read_text());checks=[]
    if r['status']!='PASS_BUILD_ONLY':raise ValueError('Passing CPU build required')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OMP_DYNAMIC='FALSE')
    for variant,b in r['binaries'].items():
        if sha(b['path'])!=b['sha256']:raise ValueError('Binary changed')
        output=out/f'{variant}.i32';cmd=[b['path'],'-p','3','-ps','5','-pn','2','-qn','4',
             '-qf',str(out/'input/sources.txt'),str(out/'input/inter.adj'),str(out/'input/intra.adj'),str(out/'input/parts.txt')]
        log=subprocess.run(cmd,env=dict(env,EGGPU_KGRAPH_OUTPUT=str(output)),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        (out/f'{variant}.log').write_text(log.stdout)
        if log.returncode:raise RuntimeError((variant,log.returncode,log.stdout[-1500:]))
        data=output.read_bytes();expected=[d for s in fixture['sources'] for d in reference(fixture['n'],fixture['edges'],s)]
        observed=list(struct.unpack('<'+'i'*len(expected),data))
        if observed!=expected:raise AssertionError((variant,observed,expected))
        rows=[json.loads(x.split('EGGPU_KGRAPH_JSON ',1)[1]) for x in log.stdout.splitlines() if x.startswith('EGGPU_KGRAPH_JSON ')]
        if len(rows)!=1 or rows[0]['queries']!=4 or rows[0]['n']!=9 or rows[0]['pivots']!=2:raise ValueError('Metadata mismatch')
        checks.append(dict(variant=variant,argv=cmd,complete_values_checked=len(expected),memoization_pivots=2,output_sha256=sha(output),status='PASS'))
    # Meaningful parser/admission failures, not mirrors of implementation timing.
    invalid_rejected=0
    for n,edges,parts,sources in [(2,[(0,1,0)],[0,0],[0]),(2,[(0,1,2),(0,1,3)],[0,0],[0]),(2,[(0,1,2)],[0,2],[0]),(2,[(0,1,2)],[0,0],[2])]:
        try:validate_input(n,edges,parts,sources)
        except ValueError:invalid_rejected+=1
    assert invalid_rejected==4
    write(out/'receipt.json',dict(schema='kgraph.tiny.qualification/v1',status='PASS_TINY_CPU_ONLY',gpu_executed=False,
          formal_performance_eligible=False,build_sha256=sha(a.build),checks=checks,invalid_inputs_rejected=invalid_rejected,
          no_large_graph_or_formal_runs=True))
    print(json.dumps({'status':'PASS_TINY_CPU_ONLY','receipt':str(out/'receipt.json')}))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    q=s.add_parser('build');q.add_argument('--source',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--cxx',default='/usr/bin/g++')
    q=s.add_parser('qualify-tiny');q.add_argument('--build',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();return build(a) if a.action=='build' else qualify(a)


if __name__=='__main__':raise SystemExit(main())
