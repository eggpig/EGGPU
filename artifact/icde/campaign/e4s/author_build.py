#!/usr/bin/env python3
"""CPU-only observation/FP64 output adaptation of fixed author batch executors.

Does not run either executor. Graph algorithms, pivots, queues and GPU kernels
remain unchanged. Existing author correctness patches retain their identities.
"""
from pathlib import Path
import argparse, difflib, hashlib, json, os, shutil, subprocess, time

CLOCK_HEADER='''#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <time.h>
#include <unistd.h>
static inline std::uint64_t e4s_monotonic_ns(){
    struct timespec value;
    if(clock_gettime(CLOCK_MONOTONIC,&value)!=0)throw std::runtime_error("E4-S CLOCK_MONOTONIC failed");
    return std::uint64_t(value.tv_sec)*1000000000ULL+std::uint64_t(value.tv_nsec);
}
static inline void e4s_report_complete(std::uint64_t ns){
    std::printf("E4S_COMPLETE_JSON {\\"complete_monotonic_ns\\":%llu,\\"pid\\":%lld}\\n",
        static_cast<unsigned long long>(ns),static_cast<long long>(getpid()));
    std::fflush(stdout);
}
'''


def binding(path):
    p=Path(path);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Expected one fixed source anchor: '+old[:100])
    return text.replace(old,new)


def build(args):
    root,out=args.root.resolve(),args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    jobs=[];changes=[];bases={}
    def change(path,after,source):
        before=path.read_text();path.write_text(after)
        changes.append({'path':str(path),'before':binding(source),'after':binding(path),
                        'patch':''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(source),tofile=str(path)))})
    kgbase=root/'experimental/campaign_20261104/baselines/kgraph_build_v1'
    c2base=root/'build/c2_pm_closure_v3'
    for name,base,commit in [('kgraph',kgbase,'e09ca53695cb5a82d1bbb2418caac90b2932af7a'),
                              ('c2graph',c2base,'97af1ebd935a03f5adc180935644e786a624bdbe')]:
        receipt=json.loads((base/'receipt.json').read_text())
        if receipt['status']!='PASS_BUILD_ONLY':raise ValueError('Missing fixed author CPU build')
        observed=receipt.get('source_commit',receipt.get('upstream',{}).get('commit'))
        if observed!=commit:raise ValueError('Author commit differs')
        source=out/name/'source';source.parent.mkdir();shutil.copytree(base/'source',source)
        bases[name]={'receipt':binding(base/'receipt.json'),'commit':commit,
                     'files':{str(p.relative_to(source)):binding(p) for p in sorted(source.rglob('*')) if p.is_file()}}
        if name=='kgraph':
            for variant in ('sparse','dense'):
                path=source/f'apps/kgraph-sssp-{variant}.C';text=CLOCK_HEADER+'#include <limits>\n'+path.read_text()
                text=once(text,'std::vector<std::int32_t> eggpu_owning(size_t(k)*size_t(n));','std::vector<double> eggpu_owning(size_t(k)*size_t(n));')
                text=once(text,'std::copy(ShortestPathLen_l[qi],ShortestPathLen_l[qi]+n,eggpu_owning.begin()+size_t(qi)*size_t(n));',
                    'for(long vi=0;vi<n;++vi){const auto d=ShortestPathLen_l[qi][vi];eggpu_owning[size_t(qi)*size_t(n)+size_t(vi)]=d==INT_MAX/2?std::numeric_limits<double>::infinity():static_cast<double>(d);}')
                text=once(text,'eggpu_owning.size()*sizeof(std::int32_t)','eggpu_owning.size()*sizeof(double)')
                text=once(text,'  const auto eggpu_complete=std::chrono::steady_clock::now();',
                    '  const auto eggpu_complete=std::chrono::steady_clock::now();\n  const auto e4s_complete_ns=e4s_monotonic_ns();\n  e4s_report_complete(e4s_complete_ns);')
                text=once(text,'<<",\\\"native_infinity\\\":"<<(INT_MAX/2)',
                    '<<",\\\"complete_monotonic_ns\\\":"<<e4s_complete_ns<<",\\\"pid\\\":"<<getpid()<<",\\\"native_infinity\\\":"<<(INT_MAX/2)')
                change(path,text,base/'source'/path.relative_to(source))
                binary=out/name/('kgraph-sssp-'+variant)
                command=['/usr/bin/g++','-std=c++14','-O3','-fopenmp','-DWEIGHTED','-DOPENMP','-DBYTERLE',
                         '-I'+str(source/'ligra'),str(path),'-o',str(binary)]
                jobs.append({'name':'kgraph_'+variant,'command':command,'binary':str(binary)})
        else:
            path=source/'src/sssp/purn-perk/Main.cu';text=CLOCK_HEADER+'#include <limits>\n'+path.read_text()
            text=once(text,'    const auto pm_end=std::chrono::steady_clock::now();','''    std::vector<std::vector<double>> complete_fp64;
    complete_fp64.reserve(complete_results.size());
    for(const auto& result:complete_results){
        complete_fp64.emplace_back(result.values.size());
        for(size_t v=0;v<result.values.size();++v){const auto d=result.values[v];complete_fp64.back()[v]=d==1000000000?std::numeric_limits<double>::infinity():static_cast<double>(d);}
    }
    const auto pm_end=std::chrono::steady_clock::now();
    const auto e4s_complete_ns=e4s_monotonic_ns();
    e4s_report_complete(e4s_complete_ns);''')
            text=once(text,'    if(!output)throw std::runtime_error("Explicit output directory required");',
                '    if(!output)throw std::runtime_error("Explicit output directory required");\n'
                '    const char* batch_output=std::getenv("EGGPU_E4S_OUTPUT");\n'
                '    if(!batch_output)throw std::runtime_error("Explicit complete batch output required");\n'
                '    std::ofstream f(batch_output,std::ios::binary);')
            text=once(text,'std::ofstream f(prefix+".i32",std::ios::binary);f.write(reinterpret_cast<const char*>(r.values.data()),r.values.size()*sizeof(ValueType));',
                'f.write(reinterpret_cast<const char*>(complete_fp64[i].data()),complete_fp64[i].size()*sizeof(double));')
            text=once(text,'<<",\\\"batch_complete_host_seconds\\\":"<<std::chrono::duration<double>(pm_end-pm_batch_begin).count()',
                '<<",\\\"complete_monotonic_ns\\\":"<<e4s_complete_ns<<",\\\"pid\\\":"<<getpid()\n'
                '      <<",\\\"batch_complete_host_seconds\\\":"<<std::chrono::duration<double>(pm_end-pm_batch_begin).count()')
            change(path,text,base/'source'/path.relative_to(source))
            binary=out/name/'c2graph-pm';command=[x.replace(str(base/'source'),str(source)) for x in receipt['command']]
            command[command.index('-o')+1]=str(binary)
            jobs.append({'name':'c2graph_pm','command':command,'binary':str(binary)})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2');env.pop('LD_PRELOAD',None)
    os.sched_setaffinity(0,{84,85});steps=[]
    for job in jobs:
        start=time.monotonic()
        with (out/(job['name']+'.build.log')).open('w') as log:
            code=subprocess.run(job['command'],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        steps.append({**job,'returncode':code,'seconds':time.monotonic()-start})
        if code:break
    record={'schema':'e4s.author.fp64.build.v1','status':'PASS_BUILD_ONLY' if code==0 else 'FAIL_BUILD','gpu_executed':False,
            'algorithm_changes':False,'base_builds':bases,'changes':changes,'steps':steps,'builder':binding(__file__),
            'binaries':{j['name']:binding(j['binary']) for j in jobs if Path(j['binary']).exists()},
            'scope':'Only complete owning FP64/+inf output conversion inside existing measured batch interval. Author algorithms/scheduling unchanged; compression/partition generation must separately be included in T_first.'}
    (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');(out/'adapter.patch').write_text(''.join(c['patch'] for c in changes))
    print(json.dumps({'status':record['status'],'receipt':str(out/'receipt.json')}));return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(build(p.parse_args()))
