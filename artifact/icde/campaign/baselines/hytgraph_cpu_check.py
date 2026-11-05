#!/usr/bin/env python3
"""Exercise the actual adapted author parser on CPU; never load a CUDA library."""
from pathlib import Path
from types import SimpleNamespace
import argparse, json, os, subprocess
from hytgraph_adapter import prepare, sha

DRIVER=r'''
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <exception>
#include <utils/parser.h>
int main(int argc,char**argv){
  try {
    graph_t* g=ReadGraphMarket_bigdata(argv[1],0);
    std::cout<<"DOMAIN "<<g->nvtxs<<" "<<g->nedges<<"\n";
    for(unsigned u=0;u<g->nvtxs;++u)
      for(uint64_t j=g->xadj[u];j<g->xadj[u+1];++j)
        std::cout<<u<<" "<<g->adjncy[j]<<" "<<g->adjwgt[j]<<"\n";
    FreeGraph(&g);return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 3;}
}
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cxx',default='/usr/bin/g++');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    prepared=prepare(SimpleNamespace(source=a.source,output=out/'prepared'))
    src=Path(prepared['patched_source']);driver=out/'parser_driver.cpp';driver.write_text(DRIVER)
    cmd=[a.cxx,'-std=c++17','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I'+str(src/'include'),
         str(src/'src/utils/parser.cpp'),str(driver),'-o',str(out/'parser_driver')]
    built=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (out/'build.log').write_text(built.stdout)
    if built.returncode:raise RuntimeError(built.stdout[-2000:])
    cases=[(9,[(0,1,2),(1,2,3),(0,3,9),(2,3,1),(3,4,4),(4,5,1),(5,3,2),(6,7,2),(7,6,1)]),
           (12,[(0,10,16)]),(4,[])]
    checks=[]
    # Test exact weighted arcs, source/destination-only vertices, trailing isolated
    # vertices and an edgeless explicit domain using the original C++ parser body.
    for i,(n,edges) in enumerate(cases):
        inp=out/f'case_{i}.edges';inp.write_text(''.join(f'{u} {v} {w}\n' for u,v,w in edges))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EGGPU_HYTGRAPH_NODES=str(n),ASAN_OPTIONS='detect_leaks=0')
        run=subprocess.run([str(out/'parser_driver'),str(inp)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/f'case_{i}.log').write_text(run.stdout)
        if run.returncode:raise RuntimeError((i,run.returncode,run.stdout[-2000:]))
        lines=run.stdout.splitlines();assert lines[0]==f'DOMAIN {n} {len(edges)}'
        assert sorted(tuple(map(int,x.split())) for x in lines[1:])==sorted(edges)
        checks.append(dict(case=i,n=n,edges=len(edges),status='PASS',sanitizers=['address','undefined'],input_sha256=sha(inp)))
    # Invalid explicit domain is rejected rather than silently truncating vertices.
    bad=subprocess.run([str(out/'parser_driver'),str(out/'case_1.edges')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EGGPU_HYTGRAPH_NODES='5',ASAN_OPTIONS='detect_leaks=0'),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'invalid_domain.log').write_text(bad.stdout)
    assert bad.returncode==3 and 'Invalid full vertex domain' in bad.stdout
    r=dict(schema='hytgraph.cpu.parser.qualification/v1',status='PASS_TINY_CPU_ONLY',gpu_executed=False,
           formal_performance_eligible=False,checks=checks,invalid_domain_rejected=True,build_argv=cmd,
           leak_sanitizer=False,leak_sanitizer_reason='Container disallows /proc task inspection required by LSan; address/undefined checks remain enabled.',
           preparation_sha256=sha(out/'prepared/preparation.json'),source_verification=prepared['source_verification'])
    (out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'receipt':str(out/'receipt.json')}))


if __name__=='__main__':main()
