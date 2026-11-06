"""CPU-only prepared author input; never takes a GPU lock or starts a binary."""
from pathlib import Path
import argparse
import fcntl
import importlib.util
import json
import os
import sys
import time
from text_cache import Converter,TextCache,atomic


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def memory_gate(common,required,update,poll_seconds,fail_on_resource):
    while True:
        memory=common.cgroup_memory()
        update(required_host_bytes=required,host_memory=memory)
        if memory['available_bytes']>=required:return
        if fail_on_resource:
            raise RuntimeError('CPU_RESOURCE_UNAVAILABLE: insufficient host/cgroup memory; fail without waiting')
        update(status='WAITING_HOST_MEMORY')
        time.sleep(poll_seconds)


def prepare(args):
    campaign=args.campaign.resolve();worker=Path(args.worker).resolve() if args.worker else Path(__file__).resolve().parent.parent/'baselines/sg_author_worker.py'
    sys.path[:0]=[str(Path(__file__).resolve().parent.parent/'timing'),str(worker.parent)]
    import cache_worker
    identity=cache_worker.full_binding(worker)
    if identity['sha256']!=args.worker_sha256:raise ValueError('Original frozen author worker changed')
    selected={int(x) for x in args.cpu_set.split(',')}
    if len(selected)!=16 or not selected<=os.sched_getaffinity(0):raise ValueError('Exactly sixteen available CPU IDs required')
    os.sched_setaffinity(0,selected)
    module=load('_text_preparation_original_author',worker)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cache=cache_worker.BindingCache(args.text_cache.resolve()/'binding_cache.sqlite')
    module.common.binding=cache.binding
    converter=Converter(args.converter_build,cache.binding)
    store=TextCache(args.text_cache,converter,cache,cache_worker,identity,True,nonblocking=args.fail_on_resource)
    record=dict(status='WAITING_HOST_MEMORY',gpu_executed=False,cpu_affinity=sorted(selected),
        original_worker=identity,converter_build=cache.binding(args.converter_build),text_cache=str(args.text_cache.resolve()),layouts={},started_unix=time.time(),
        resource_mode='fail_without_waiting' if args.fail_on_resource else 'wait_for_host_memory')
    def update(**fields):record.update(fields);atomic(out/'state.json',record)
    try:
        canonical=campaign/'inputs/canonical/TW';metadata=json.loads((canonical/'canonical.json').read_text())
        required=max(8<<30,128*(metadata['num_nodes']+metadata['num_entries'])+(2<<30))
        with (args.text_cache.resolve()/'.preparation.lock').open('a+') as private:
            fcntl.flock(private,fcntl.LOCK_EX|fcntl.LOCK_NB)
            memory_gate(module.common,required,update,args.poll_seconds,args.fail_on_resource)
            update(status='LOADING_CANONICAL_CPU_INPUT')
            graph=module.common.load_input(canonical)
            record.update(input_sha256=graph['input_sha256'],canonical_receipt=cache.binding(canonical/'receipt.json'))
            for kind,algorithm in [('canonical','bfs'),('wcc','wcc')]:
                memory_gate(module.common,required,update,args.poll_seconds,args.fail_on_resource)
                update(status='PREPARING_'+kind.upper())
                started=time.monotonic()
                try:
                    layout=module.graph_layout(graph,'grapin',algorithm,False)
                except OverflowError as error:
                    if kind!='wcc' or not str(error).startswith('INDEX_UNSUPPORTED:'):raise
                    record['layouts'][kind]=dict(status='UNSUPPORTED_FROZEN_INT32_LAYOUT',error=str(error),
                        original_graph_layout_executed=True,gpu_executed=False)
                    continue
                entry,receipt=store.ensure(layout)
                record['layouts'][kind]=dict(status='PASS',key=receipt['key'],metadata=cache.binding(entry/'metadata.json'),
                    blob=receipt['blob'],n=layout['n'],m=layout['m'],seconds=time.monotonic()-started)
                del layout
            update(status='PASS_CPU_PREPARED_TEXT_INPUTS',finished_unix=time.time(),cache=cache.report(),events=store.events)
            atomic(out/'receipt.json',record)
            return record
    except BaseException as error:
        update(status='FAILED_CPU_PREPARATION',error=type(error).__name__+': '+str(error),finished_unix=time.time())
        atomic(out/'receipt.json',record);raise
    finally:cache.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','text-cache','converter-build','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--worker-sha256',required=True);p.add_argument('--worker',type=Path,help='Exact original author worker; default public campaign/baselines/sg_author_worker.py')
    p.add_argument('--cpu-set',required=True,help='Exactly sixteen available CPU IDs')
    p.add_argument('--poll-seconds',type=int,choices=range(1,61),default=15)
    p.add_argument('--fail-on-resource',action='store_true',help='Fail immediately on insufficient memory or busy cache locks; use while an external controller lock is held. Disk shortage always fails immediately.')
    print(json.dumps(prepare(p.parse_args())))
