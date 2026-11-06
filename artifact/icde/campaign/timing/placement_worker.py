"""Explicit first-pass P1 placement portability; unchanged numerical workers.

Only physical placement changes: one selected identical GPU, a private GPU lock,
and exactly sixteen CPU IDs. Original tiny proof bytes are still checked by the
original reader. Actual new epochs are never rewritten to an old fingerprint.
"""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import types
import cache_worker

GPU_QUERY = '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version'
PROCESS_QUERY = '--query-compute-apps=gpu_uuid,pid,used_memory'


def option(arguments, name):
    if arguments.count(name) != 1:
        raise ValueError('Exactly one ' + name + ' required')
    return arguments[arguments.index(name)+1]


def placement(epoch):
    if 'gpu_uuid' in epoch and 'affinity' in epoch:
        return epoch['gpu_uuid'], epoch['affinity']
    config = epoch.get('configuration', {})
    return config.get('environment', {}).get('CUDA_VISIBLE_DEVICES'), config.get('cpu_affinity')


def normalize(epoch):
    result = copy.deepcopy(epoch)
    if 'gpu_uuid' in result and 'affinity' in result:
        result.pop('gpu_uuid'); result.pop('affinity')
        environment = result.get('environment', {})
        environment.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        config = result['configuration']
        config.pop('cpu_affinity')
        config['environment'].pop('CUDA_VISIBLE_DEVICES')
    return result


def compare_epochs(old, current, expected_gpu, expected_cpus):
    old_gpu, old_cpus = placement(old)
    new_gpu, new_cpus = placement(current)
    if not isinstance(old_gpu, str) or not old_gpu.startswith('GPU-') or ',' in old_gpu:
        raise ValueError('Original proof is not a single physical GPU placement')
    for cpus in (old_cpus, new_cpus):
        if not isinstance(cpus, list) or len(cpus) != 16 or len(set(cpus)) != 16 or any(type(x) is not int for x in cpus):
            raise ValueError('Both placements must use exactly sixteen CPU IDs')
    if new_gpu != expected_gpu or new_cpus != sorted(expected_cpus):
        raise ValueError('Actual current epoch does not match requested placement')
    for epoch, gpu in ((old,old_gpu),(current,new_gpu)):
        if 'gpu_uuid' in epoch:
            visible = epoch.get('environment', {}).get('CUDA_VISIBLE_DEVICES')
            if visible is not None and visible != gpu:
                raise ValueError('GPU environment is not the same explicit single-device placement')
    if normalize(old) != normalize(current):
        raise ValueError('Qualification differs beyond explicit physical GPU/CPU placement')
    return dict(old_gpu=old_gpu, new_gpu=new_gpu, old_cpus=old_cpus, new_cpus=new_cpus)


def filter_selected_csv(text, command, gpu):
    if GPU_QUERY in command:
        index = 1
    elif PROCESS_QUERY in command:
        index = 0
    else:
        return text
    return '\n'.join(line for line in text.splitlines()
                     if len(next(csv.reader([line]))) > index
                     and next(csv.reader([line]))[index].strip() == gpu)


class SelectedSubprocess:
    def __init__(self, original, gpu):
        self.original, self.gpu = original, gpu
    def __getattr__(self, name):
        return getattr(self.original, name)
    def check_output(self, command, *args, **kwargs):
        value = self.original.check_output(command, *args, **kwargs)
        if isinstance(value, str):
            return filter_selected_csv(value, command, self.gpu)
        return value


def cloned(function, changes):
    namespace = dict(function.__globals__); namespace.update(changes)
    result = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class PlacementPolicy:
    def __init__(self, gpu, cpus, setup, helper, cache):
        self.gpu, self.cpus, self.setup = gpu, sorted(cpus), Path(setup)
        self.helper, self.cache = helper, cache
        self.epochs, self.accepted, self.devices = {}, [], {}
    def observe_devices(self, old_gpu):
        allowed_text = json.loads(self.setup.read_text())['gpus']['output']
        allowed = {row[1].strip() for row in list(csv.reader(io.StringIO(allowed_text)))[1:] if len(row) > 1}
        if not {old_gpu,self.gpu} <= allowed:
            raise ValueError('Placement outside recorded dedicated server')
        text = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True)
        rows = {row[0].strip(): tuple(x.strip() for x in row[1:]) for row in csv.reader(io.StringIO(text)) if len(row)==4}
        if old_gpu not in rows or self.gpu not in rows or rows[old_gpu] != rows[self.gpu]:
            raise ValueError('Physical GPUs differ in model, memory or driver')
        if '4090d' not in rows[self.gpu][0].lower().replace(' ',''):
            raise ValueError('Expected the authorized RTX 4090 D server')
        result = dict(old_gpu=old_gpu,new_gpu=self.gpu,model=rows[self.gpu][0],memory_mib=rows[self.gpu][1],driver=rows[self.gpu][2])
        self.devices[old_gpu] = result
        return result
    def install(self, module):
        common = getattr(module, 'common', module)
        original_epoch = module.source_epoch if hasattr(module,'source_epoch') else module.epoch
        def capture(*args, **kwargs):
            value = original_epoch(*args, **kwargs)
            epoch, digest = value if isinstance(value, tuple) else (value, common.digest_json(value))
            self.epochs[digest] = copy.deepcopy(epoch)
            return value
        if hasattr(module,'source_epoch'): module.source_epoch = capture
        else: module.epoch = capture
        original_validate = common.validate_proof
        def validate(path, *, kind, system, algorithm, epoch_sha, graph):
            proof = json.loads(Path(path).read_text())
            old_sha = proof.get('epoch_sha256')
            if old_sha == epoch_sha:
                return original_validate(path,kind=kind,system=system,algorithm=algorithm,epoch_sha=epoch_sha,graph=graph)
            if kind != 'tiny' or epoch_sha not in self.epochs:
                raise ValueError('Only existing tiny proof can be portable for a first-pass target')
            old_binding = proof.get('epoch', {})
            if not old_binding or self.cache.binding(old_binding['path']) != old_binding:
                raise ValueError('Original tiny epoch file changed')
            old = json.loads(Path(old_binding['path']).read_text())
            current = self.epochs[epoch_sha]
            placement_info = compare_epochs(old,current,self.gpu,self.cpus)
            hardware = self.observe_devices(placement_info['old_gpu'])
            # The unchanged original reader still validates every actual tiny
            # result, source block and output. No receipt/hash is fabricated.
            result = original_validate(path,kind=kind,system=system,algorithm=algorithm,epoch_sha=old_sha,graph=graph)
            self.accepted.append(dict(proof=result,old_epoch_sha256=old_sha,actual_current_epoch_sha256=epoch_sha,
                placement=placement_info,hardware=hardware,
                reason='Exact same source/binary/precision/quality/thread count on another recorded identical GPU and sixteen CPU IDs. Only physical placement fields are normalized.'))
            return result
        common.validate_proof = validate
        common.resource_check = cloned(common.resource_check, {'subprocess': SelectedSubprocess(subprocess,self.gpu)})


def run(worker, expected, cache_path, arguments, cpu_set=None):
    worker = Path(worker).resolve()
    if worker.name not in ('sg_worker.py','sg_author_worker.py'):
        raise ValueError('Placement wrapper is restricted to SG or author P1 workers')
    if option(arguments,'--phase') not in ('tiny','target'):
        raise ValueError('Placement wrapper is only for the preliminary first pass')
    gpu = option(arguments,'--gpu-uuid'); lock = Path(option(arguments,'--lock-file')).resolve()
    if not gpu.startswith('GPU-') or ',' in gpu or lock.name != gpu+'.lock':
        raise ValueError('Use one explicit GPU UUID and its own GPU-UUID.lock')
    if not lock.parent.is_dir():raise ValueError('Per-GPU lock directory must already exist')
    if cpu_set is not None:os.sched_setaffinity(0,set(cpu_set))
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) != 16:raise ValueError('Exactly sixteen CPU IDs required')
    if '--cpus' in arguments and sorted(int(x) for x in option(arguments,'--cpus').split(',')) != cpus:
        raise ValueError('Author CPU placement argument differs from actual affinity')
    identity = cache_worker.full_binding(worker)
    if identity['sha256'] != expected:raise ValueError('Original worker changed')
    wrapper = Path(__file__).resolve(); cache = cache_worker.BindingCache(cache_path)
    old_argv, old_path, original_popen = sys.argv, list(sys.path), subprocess.Popen
    output = Path(option(arguments,'--output')).resolve(); status='FAILED'; policy=None
    try:
        sys.argv=[str(worker),*arguments];sys.path.insert(0,str(worker.parent))
        spec=importlib.util.spec_from_file_location('_first_pass_placement_original',worker)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        common=getattr(module,'common',module)
        for value in {module,common}:
            if callable(getattr(value,'binding',None)):value.binding=cache.binding
        policy=PlacementPolicy(gpu,cpus,option(arguments,'--setup'),common,cache)
        policy.install(module)
        def child(command,*args,**kwargs):
            return original_popen(cache_worker.rewrite_child(command,worker,wrapper,cache.path,expected),*args,**kwargs)
        subprocess.Popen=child
        if worker.name=='sg_author_worker.py':module.run(module.parser().parse_args())
        else:module.main()
        status='COMPLETE'
    finally:
        subprocess.Popen=original_popen;sys.argv=old_argv;sys.path[:]=old_path
        record=dict(schema='eggpu.first_pass.placement.v1',status=status,original_worker=identity,
            wrapper=cache_worker.full_binding(wrapper),cache_helper=cache_worker.full_binding(cache_worker.__file__),
            gpu_uuid=gpu,cpu_affinity=cpus,per_gpu_lock=str(lock),cache=cache.report(),
            portable_tiny_proofs=[] if policy is None else policy.accepted,
            actual_epoch_hashes=[] if policy is None else list(policy.epochs),
            actual_epoch_preserved=True,algorithm_modified=False,numerical_checks_modified=False,
            concurrency_policy='Root schedules disjoint GPU/CPU placements and aggregate host-memory headroom; this worker excludes activity only on its reserved GPU.',
            formal_performance_claim=False)
        if output.is_dir():
            with (output/'placement_usage.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
        cache.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path,required=True);p.add_argument('--worker-sha256',required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--cpu-set')
    p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    cpus=None if a.cpu_set is None else [int(x) for x in a.cpu_set.split(',')]
    run(a.worker,a.worker_sha256,a.cache,args,cpus)


if __name__=='__main__':main()
