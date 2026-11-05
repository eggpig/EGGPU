#!/usr/bin/env python3
"""Resolve every frozen author binary with ELF tracing only; never enter main."""
from pathlib import Path
import argparse, json
import sg_author_worker as worker
import run_sg_authors as controller

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--clock-build-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();checks=[]
    for variant,(system,functions) in controller.VARIANTS.items():
        for function in functions:
            args=argparse.Namespace(root=a.root,clock_build_root=a.clock_build_root,system=system,algorithm=function,variant=variant,gpu_uuid='GPU-not-used-cpu-trace')
            artifact=worker.artifact(args);runtime=worker.resolved_runtime(artifact['binary']['path'],worker.environment(args))
            checks.append({'variant':variant,'function':function,'artifact':artifact,'runtime_linker':runtime,'passed':True})
    record={'status':'PASS_CPU_ELF_RUNTIME_RESOLUTION','tests':len(checks),'gpu_executed':False,'checks':checks,
            'files':[worker.common.binding(__file__),worker.common.binding(worker.__file__),worker.common.binding(controller.__file__)],
            'scope':'ldd/LD_TRACE_LOADED_OBJECTS under exact native library settings; no main, graph query, CUDA runtime invocation or GPU performance claim'}
    worker.common.save(a.output,record);print(json.dumps({'status':record['status'],'tests':len(checks),'receipt':str(a.output)}))
