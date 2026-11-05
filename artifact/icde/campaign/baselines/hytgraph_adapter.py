#!/usr/bin/env python3
"""Build fixed-author HyTGraph with complete-output instrumentation; never run GPU."""
from pathlib import Path
import argparse, difflib, hashlib, json, os, re, shutil, subprocess, time
from source_lock import verify_group

COMMIT='3e5b0e47035972cd8ba1e0a9c89d99c108041de9'
FUNCTIONS={'bfs':'HybridBFS','sssp':'HybridSSSP','cc':'HybridCC','pr':'HybridPageRank'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def one(t,a,b):
    if t.count(a)!=1:raise ValueError(('Frozen source mismatch',a[:100],t.count(a)))
    return t.replace(a,b)


def application(text,alg):
    if alg=='pr':text=one(text,'if (buffer > 0.01)','if (buffer > m_error)')
    name=FUNCTIONS[alg];pos=text.index('bool '+name+'()');start=text.index('{',pos)
    init=re.search(r'    engine.InitGraph\([^;]+;',text[start:])
    if not init:raise ValueError('No author InitGraph')
    prefix=text[start+1:start+init.start()]
    if alg=='bfs':prefix=one(prefix,'    auto regression = BFSHost(engine.CSRGraph(), source_node);','')
    if alg=='pr':prefix=prefix.replace('sepgraph::engine::EngineOptions','sepgraph::common::EngineOptions')
    init={'bfs':'engine.InitGraph(source_node, FLAGS_non_atomic);','sssp':'engine.InitGraph(source_node);',
          'cc':'engine.InitGraph(source_node);','pr':'engine.InitGraph(FLAGS_error);'}[alg]
    startcall='engine.Start(init_prio);' if alg=='sssp' else 'engine.Start();'
    convert=('double mass=0; for(auto value:host) mass+=double(value); '
             'if(!(mass>0) || !std::isfinite(mass)) throw std::runtime_error("Invalid PR mass"); '
             'for(size_t i=0;i<host.size();++i)owning[i]=double(host[i])/mass;') if alg=='pr' else (
             'double mass=0; for(size_t i=0;i<host.size();++i)owning[i]=(host[i]==UINT32_MAX?std::numeric_limits<double>::infinity():double(host[i]));'
             if alg in ('bfs','sssp') else 'double mass=0; for(size_t i=0;i<host.size();++i)owning[i]=double(host[i]);')
    includes='''#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <sstream>
#include <cmath>
#include <stdexcept>
'''
    body=r'''
    using Clock=std::chrono::steady_clock;
    const auto first_begin=Clock::now();
    int device_count=0;cudaGetDeviceCount(&device_count);
    if(device_count!=1)throw std::runtime_error("Exactly one visible GPU required");
    cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
    std::ostringstream uuid;uuid<<"GPU-"<<std::hex<<std::setfill('0');
    for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)uuid<<'-';uuid<<std::setw(2)<<unsigned(static_cast<unsigned char>(prop.uuid.bytes[i]));}
'''+prefix+r'''
    const char* destination=std::getenv("EGGPU_HYTGRAPH_OUTPUT");
    if(!destination)throw std::runtime_error("EGGPU_HYTGRAPH_OUTPUT required");
    std::vector<std::vector<double>> outputs;
    std::vector<double> times,masses;
    const auto ready=Clock::now();
    for(int call=0;call<3;++call){
        const auto query_begin=Clock::now();
        INIT
        START
        const auto& host=engine.GatherValue();
        std::vector<double> owning(host.size());
        CONVERT
        outputs.push_back(std::move(owning));
        if(cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("CUDA completion failed");
        times.push_back(std::chrono::duration<double>(Clock::now()-query_begin).count());masses.push_back(mass);
    }
    for(size_t i=0;i<outputs.size();++i){
        std::ofstream f(std::string(destination)+"/output."+std::to_string(i)+".f64",std::ios::binary);
        f.write(reinterpret_cast<const char*>(outputs[i].data()),outputs[i].size()*sizeof(double));
        if(!f)throw std::runtime_error("Cannot archive complete output");
        std::cout<<std::setprecision(17)<<"HYTGRAPH_COMPLETE_JSON {\"round\":"<<i
          <<",\"query_complete_host_seconds\":"<<times[i]
          <<",\"backend_prepare_seconds\":"<<std::chrono::duration<double>(ready-first_begin).count()
          <<",\"first_request_seconds\":"<<(std::chrono::duration<double>(ready-first_begin).count()+times[0])
          <<",\"raw_score_sum\":"<<masses[i]<<",\"n\":"<<outputs[i].size()
          <<",\"gpu_uuid\":\""<<uuid.str()<<"\",\"visible_device_count\":"<<device_count<<"}"<<std::endl;
    }
    return true;
}
'''.replace('INIT',init).replace('START',startcall).replace('CONVERT',convert)
    return includes+text[:pos]+'bool '+name+'()\n{'+body


CMAKE='''cmake_minimum_required(VERSION 3.18)
project(hytgraph_adapter LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
foreach(alg bfs sssp cc pr)
  set(host "")
  if(NOT alg STREQUAL "cc")
    set(host ${CMAKE_CURRENT_SOURCE_DIR}/samples/hybrid_${alg}/hybrid_${alg}_host.cpp)
  endif()
  set(main ${CMAKE_CURRENT_SOURCE_DIR}/samples/hybrid_${alg}/main.cpp)
  set_source_files_properties(${main} ${host} PROPERTIES LANGUAGE CUDA)
  add_executable(hytgraph_${alg} src/utils/parser.cpp src/utils/utils.cpp src/utils/to_json.cpp
    samples/hybrid_${alg}/hybrid_${alg}.cu ${host} ${main})
  target_include_directories(hytgraph_${alg} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/deps/json ${HYTGRAPH_NVTX_INCLUDE})
  target_compile_options(hytgraph_${alg} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp> $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
  target_link_libraries(hytgraph_${alg} PRIVATE gflags pthread dl CUDA::cudart OpenMP::OpenMP_CXX)
endforeach()
'''


def prepare(a):
    src,out=a.source.resolve(),a.output.resolve()
    source_verification=verify_group(src,'hytgraph')
    out.mkdir(parents=True,exist_ok=False);dst=out/'source'
    shutil.copytree(src,dst,ignore=shutil.ignore_patterns('.git','build','.kdev4'))
    changes=[]
    def change(rel,text,reason):
        p=dst/rel;old=p.read_text();p.write_text(text)
        changes.append(dict(path=rel,reason=reason,before_sha256=sha(src/rel),after_sha256=sha(p),
            patch=''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))))
    change('CMakeLists.txt',CMAKE,'Use installed CUDA12.8/CUB, gflags, NVTX headers and explicit modern architecture; original execution kernels retained.')
    for alg in FUNCTIONS:
        rel=f'samples/hybrid_{alg}/hybrid_{alg}.cu'
        change(rel,application((dst/rel).read_text(),alg),
            'Three fresh algorithm states on retained author graph, complete owning host output, metadata/timing; PR uses m_error consistently and timed full-score normalization.')
    rel='samples/hybrid_cc/main.cpp';text=(dst/rel).read_text()
    text=one(text,'            HybridCC();','            return HybridCC();')
    change(rel,text,'Return the actual CC result from the declared bool launcher; fixes undefined behavior after complete output, without changing any CC algorithm.')
    rel='src/utils/parser.cpp';text=(dst/rel).read_text()
    text='#include <cstdlib>\n#include <stdexcept>\n'+text
    text=one(text,'    graph->nvtxs++;','''    graph->nvtxs++;
    const char* expected_nodes=std::getenv("EGGPU_HYTGRAPH_NODES");
    if(!expected_nodes) throw std::runtime_error("Explicit full vertex domain required");
    const unsigned long long n=std::stoull(expected_nodes);
    if(n<graph->nvtxs || n>UINT32_MAX || n==0) throw std::runtime_error("Invalid full vertex domain");
    graph->nvtxs=n;
    xadj_pri.resize(n,0);''')
    text=one(text,'    delete[] outDegreeCounter;','    free(outDegreeCounter);')
    change(rel,text,'Preserve trailing isolated vertices and size degree counters to full domain; match calloc with free.')
    rel='include/framework/framework.cuh';text=(dst/rel).read_text()
    text=one(text,'for(index_t seg_idx = 0; node_id < csr_graph.nnodes ; seg_idx++)',
             'for(index_t seg_idx = 0; seg_idx < m_nsegs ; seg_idx++)')
    text=one(text,'while(edge_num < seg_nedges)',
             'while(node_id < csr_graph.nnodes && (edge_num < seg_nedges || seg_idx + 1 == m_nsegs))')
    text=one(text,'if (m_running_info.current_round == 1000 )',
             'if (m_running_info.current_round >= FLAGS_max_iteration )')
    text=one(text,'                // Launch kernel to init value/buffer fields',
             '''                // Every measured query starts from a fresh algorithm state.
                m_running_info.current_round = 0;
                graph_datum.m_current_round.set_val_H2DAsync(0, stream_s.cuda_stream);
                // Launch kernel to init value/buffer fields''')
    change(rel,text,'Guard node/segment bounds, retain trailing isolated vertices, reset each fresh query round counter, and honor author maximum-iteration option.')
    (out/'adapter.patch').write_text(''.join(x['patch'] for x in changes))
    record=dict(schema='hytgraph.adapter.preparation/v1',status='PREPARED_NOT_BUILT',upstream=dict(repository='https://github.com/iDC-NEU/HyTGraph',commit=COMMIT),
                source=str(src),patched_source=str(dst),source_verification=source_verification,changes=changes,gpu_executed=False,
                adapter=sha(__file__),timing='Engine/graph preparation separately charged; repeated InitGraph through complete owning host outputs, actual GPU synchronization and PR normalization charged. Disk output/reference checks excluded; initial CUDA property probe included in first-request preparation.')
    (out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def build(a):
    record=prepare(a);out=a.output.resolve();dst=out/'source';steps=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PATH=str(a.cuda_root/'bin')+':/usr/bin:/bin')
    env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    cmd=[a.cmake,'-S',str(dst),'-B',str(out/'build'),'-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_CUDA_COMPILER='+str(a.cuda_root/'bin/nvcc'),'-DCUDAToolkit_ROOT='+str(a.cuda_root/'targets/x86_64-linux'),
         '-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared','-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
         '-DCMAKE_CUDA_ARCHITECTURES='+str(a.arch),'-DCMAKE_CXX_COMPILER=/usr/bin/g++',
         '-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++','-DHYTGRAPH_NVTX_INCLUDE='+str(a.nvtx_include)]
    for i,command in enumerate([cmd,[a.cmake,'--build',str(out/'build'),'-j','1']]):
        start=time.monotonic()
        with (out/f'build_{i}.log').open('w') as log:
            rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        steps.append(dict(argv=command,returncode=rc,seconds=time.monotonic()-start))
        if rc:break
    record.update(status='PASS_BUILD_ONLY' if rc==0 else 'FAIL_BUILD',steps=steps,
                  binaries={alg:dict(path=str(out/'build'/('hytgraph_'+alg)),sha256=sha(out/'build'/('hytgraph_'+alg)))
                  for alg in FUNCTIONS if (out/'build'/('hytgraph_'+alg)).exists()})
    (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'receipt':str(out/'receipt.json')}));return rc


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cuda-root',type=Path,required=True);p.add_argument('--nvtx-include',type=Path,required=True)
    p.add_argument('--cmake',default='/usr/bin/cmake');p.add_argument('--arch',type=int,default=89)
    return build(p.parse_args())


if __name__=='__main__':raise SystemExit(main())
