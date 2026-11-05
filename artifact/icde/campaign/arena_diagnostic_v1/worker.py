"""Independent diagnostic protocol; never usable as a formal worker.

Imports frozen runners unchanged, instruments only this diagnostic process,
and retains their complete input/proof/output/lease/resource checks.
"""
from pathlib import Path
import hashlib
import importlib
import json
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'sg_current'
PROTOCOL = 'e1-arena-existing-metrics-diagnostic-v1'
EXPECTED = {
    'sg_worker.py':'225bbd84957278295c200ef90f3b6b0136b44d8085d6ffd91e7f2a864a965867',
    'sequence16_worker.py':'b4f33528ba83585e2dfb9fc567bc07dde857f3f71d24eb29008b02a2c7cd54ed',
    'sequence16_adapter.py':'de759c61a3f1fcee9c89fc34e44322210c24fdb1b38b4f14d1cb7cc5c1bfbac6',
    'cugraph_csr_adapter.py':'a91460d8ff6a4402d8e4e7619fef295aa36fd9b1e4aa92d56eed762b723f89e3',
}


def load_base():
    for name, expected in EXPECTED.items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen diagnostic dependency changed: '+name)
    sys.path.insert(0,str(BASE))
    worker = importlib.import_module('sequence16_worker')
    for name in ('sg_worker','sequence16_adapter','sequence16_worker'):
        if Path(sys.modules[name].__file__).resolve()!=(BASE/(name+'.py')).resolve():
            raise ValueError('Wrong diagnostic runner import: '+name)
    return worker


def install(worker):
    from capture import diagnostic_execute
    original_epoch = worker.epoch
    original_command = worker.child_command
    original_execute = worker.execute_sequence
    original_configure = worker.configure
    original_job = worker.job
    original_child = worker.source_child
    worker.PROTOCOL = PROTOCOL

    def configure(args):
        if args.phase not in ('tiny','target') or args.repeat!=0 or args.system!='eggpu':
            raise ValueError('Only one EGGPU diagnostic qualification; no formal or alternate algorithm route')
        return original_configure(args)

    def epoch(args):
        data,_ = original_epoch(args)
        data['diagnostic_only'] = True
        data['formal_repetitions'] = 0
        data['files'].update(arena_diagnostic_worker=worker.shared.binding(__file__),
                             arena_diagnostic_capture=worker.shared.binding(HERE/'capture.py'))
        return data,worker.shared.digest_json(data)

    def child_command(args,folder,digest,lease_fd):
        command = original_command(args,folder,digest,lease_fd)
        if command[2]!=str(Path(worker.__file__).resolve()):
            raise ValueError('Unexpected frozen child command')
        command[2] = str(Path(__file__).resolve())
        return command

    def execute(args,graph):
        return diagnostic_execute(original_execute,worker.adapter.EGGPUSequence,args,graph)

    def stamp(args, action):
        try:
            return action(args)
        finally:
            path=args.output/'result.json'
            if path.is_file():
                record=json.loads(path.read_text())
                record.update(diagnostic_only=True,performance_eligible=False,formal_measurement=False,
                              protocol=PROTOCOL)
                worker.shared.save(path,record)

    worker.configure = configure
    worker.epoch = epoch
    worker.child_command = child_command
    worker.execute_sequence = execute
    worker.job = lambda args: stamp(args,original_job)
    worker.source_child = lambda args: stamp(args,original_child)
    return worker


if __name__=='__main__':
    install(load_base()).main()
