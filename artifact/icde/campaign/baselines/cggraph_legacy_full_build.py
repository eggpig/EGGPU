#!/usr/bin/env python3
"""Expose canonical outputs for fixed numeric builds of one CGgraph lineage.

The existing global-PR semantic/correctness port is preserved. This adds only
canonical +inf output conversion for its existing BFS/SSSP implementations.
It does not substitute the newer V1.5 traversal implementation.
"""
from pathlib import Path
import argparse, difflib, hashlib, json, os, shutil, subprocess, time


def bind(p):
    p=Path(p);return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--numeric-profile',choices=('fp64_pr','integer_traversal'),default='fp64_pr');a=p.parse_args()
    raw=json.loads(a.base.read_text());base=a.base.parent.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if raw['status']!='PASS_BUILD_ONLY' or bind(raw['binary'])['sha256']!=raw['binary_sha256']:raise ValueError('Legacy build changed')
    for item in raw['source_files']:
        if bind(item['path'])['sha256']!=item['sha256']:raise ValueError('Legacy source changed')
    source=out/'source';shutil.copytree(base/'source',source)
    file=source/'project/main_enginer.hpp';before=file.read_text()
    old='                else if(algorithm==Algorithm_type::WCC) value=coverage_order.at(size_t(value));'
    new=old+'\n                else if(value==double(std::numeric_limits<vertex_data_type>::max())) value=std::numeric_limits<double>::infinity();'
    if before.count(old)!=1:raise ValueError('Unexpected canonical-output site')
    file.write_text(before.replace(old,new));patch=out/'full_output.patch'
    patch.write_text(''.join(difflib.unified_diff(before.splitlines(True),file.read_text().splitlines(True),fromfile='before/main_enginer.hpp',tofile='after/main_enginer.hpp')))
    command=[x.replace(str(base),str(out)) for x in raw['command']]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2');env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    cpus={84,85}&os.sched_getaffinity(0)
    if cpus:os.sched_setaffinity(0,cpus)
    begin=time.monotonic()
    with (out/'build.log').open('w') as log:rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    binary=out/'native';receipt={'status':'PASS_BUILD_ONLY' if rc==0 else 'FAIL_BUILD','gpu_executed':False,
        'variant':'cggraph_legacy_full','source_commit':'3b23a7d7384fab283952c84ae60e491d58e50ccd',
        'base':bind(a.base),'binary':str(binary),'binary_sha256':bind(binary)['sha256'] if binary.exists() else None,
        'command':command,'returncode':rc,'seconds':time.monotonic()-begin,'patch':bind(patch),
        'source_files':[bind(p) for p in sorted(source.rglob('*')) if p.is_file()],
        'functions':['bfs','sssp','wcc'] if a.numeric_profile=='integer_traversal' else ['pr'],
        'numeric_profile':a.numeric_profile,
        'type_admission':'Author checkAlgorithm type gate unchanged. Fixed integer traversal and FP64 PR builds use the same author commit and are selected by function, never by graph/timing.',
        'algorithm_changed':False,'additional_adaptation':'Native unreachable sentinel converted to public +inf in owning full output'}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'receipt':str(out/'receipt.json')}));return rc


if __name__=='__main__':raise SystemExit(main())
