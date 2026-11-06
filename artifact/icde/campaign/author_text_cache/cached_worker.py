"""Run the unchanged author/placement worker with a prepared text-input cache."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time
from text_cache import Converter,TextCache,ShutilProxy,atomic
import disk_admission

PROTOCOL='author-ready-native-call2-cached-input-first-supplement-v1'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def run(args):
    values=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    def option(name):
        if values.count(name)!=1:raise ValueError('Expected one '+name)
        return values[values.index(name)+1]
    if option('--system') not in ('grapin','hytgraph') or option('--dataset')!='TW' or option('--phase')!='target' or option('--repeat')!='0':
        raise ValueError('Cache amendment is restricted to the eight deferred TW first-observation cells')
    worker=Path(args.worker).resolve();output=Path(option('--output')).resolve()
    sys.path.insert(0,str(args.placement_helper.parent))
    placement=load('_cached_text_original_placement',args.placement_helper)
    if placement.cache_worker.full_binding(worker)['sha256']!=args.worker_sha256:raise ValueError('Original author worker changed')
    preparation=json.loads(args.preparation.read_text())
    if preparation['status']!='PASS_CPU_PREPARED_TEXT_INPUTS' or preparation['original_worker']['sha256']!=args.worker_sha256:
        raise ValueError('Explicit completed CPU cache preparation is required')
    if preparation['text_cache']!=str(args.text_cache.resolve()) or preparation['converter_build']!=placement.cache_worker.full_binding(args.converter_build):
        raise ValueError('Cache preparation belongs to another cache/build')
    algorithm=option('--algorithm');kind='wcc' if algorithm=='wcc' else 'canonical'
    if preparation['layouts'][kind]['status']!='PASS':raise ValueError('Frozen author input layout was not prepared')
    storage=disk_admission.prepared_bound(preparation,algorithm,placement.cache_worker.full_binding)
    admission=disk_admission.Admission(storage,output,option('--reference-cache'))
    original_install=placement.PlacementPolicy.install;installed=[];storage_patch={};status='FAILED';started=time.time()
    def install(policy,module):
        original_install(policy,module)
        if module.__file__!=str(worker):raise ValueError('Unexpected imported worker')
        converter=Converter(args.converter_build,policy.cache.binding)
        cache=TextCache(args.text_cache,converter,policy.cache,placement.cache_worker,placement.cache_worker.full_binding(worker),False)
        module.write_layout=cache.write_layout
        module.shutil=ShutilProxy(cache)
        storage_patch.update(disk_admission.install(module,admission))
        installed.append(cache)
    placement.PlacementPolicy.install=install
    try:
        cpus=None if args.cpu_set is None else [int(x) for x in args.cpu_set.split(',')]
        placement.run(worker,args.worker_sha256,args.cache,values,cpus)
        status='COMPLETE'
    finally:
        placement.PlacementPolicy.install=original_install
        if output.is_dir():
            binding=placement.cache_worker.full_binding
            usage=dict(schema='eggpu.author.cached-input.v1',status=status,protocol=PROTOCOL,
                original_worker=binding(worker),placement_helper=binding(args.placement_helper),
                wrapper=binding(__file__),cache_helper=binding(Path(__file__).with_name('text_cache.py')),
                converter_build=binding(args.converter_build),cpu_preparation=binding(args.preparation),
                text_cache=str(args.text_cache.resolve()),events=[e for cache in installed for e in cache.events],
                started_unix=started,finished_unix=time.time(),native_binary_options_unchanged=True,
                numerical_checks_unchanged=True,first_warmup_ready_calls_unchanged=True,
                ready_scope='Original native complete call 2; identical kernel, options, numerical checks and output ownership.',
                first_scope='Supplement only: original continuous interval with prebuilt source-bound input serialization; excludes one-time CPU cache construction.',
                cold_first_comparable=False,raw_author_result_unchanged=True)
            usage['adapter_storage_admission']=dict(bound=storage,checks=admission.events,patch=storage_patch,
                helper=binding(Path(__file__).with_name('disk_admission.py')),native_algorithm_modified=False,
                source_audit=binding(Path(__file__).with_name('disk_source_audit.json')),
                host_memory_and_gpu_guards_unchanged=True)
            atomic(output/'text_cache_usage.json',usage)
            original_result=output/'result.json'
            if original_result.exists():
                result=json.loads(original_result.read_text())
                result.update(measurement_protocol=PROTOCOL,cached_input_first_supplement=True,
                    ready_measurement_unchanged=True,canonical_csr_cold_first_comparable=False,
                    original_author_result=binding(original_result),text_cache_usage=binding(output/'text_cache_usage.json'))
                atomic(output/'result_cached_input.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('worker','cache','placement-helper','text-cache','converter-build','preparation'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--worker-sha256',required=True);p.add_argument('--cpu-set')
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    run(p.parse_args())
