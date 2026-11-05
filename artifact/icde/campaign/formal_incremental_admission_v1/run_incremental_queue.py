"""Preserve frozen v8 execution; expose actual completed receipts to old maintenance.

The process hook refreshes completion before workers and CPU maintenance, while
the original v8 controller still holds its global lock. Importing this file runs no
worker. Only the primary agent may explicitly invoke main for GPU execution.
"""
from pathlib import Path
import argparse
import copy
import fcntl
import importlib.util
import json

import admit_incremental as admission


def project_completed(gate):
    manifest = admission.load(admission.checked(gate['parent_manifest']))
    with Path(manifest['controller_lock']).open('a+') as probe:
        try: fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise ValueError('All completed projection updates require the original controller lock')
    folder = Path(gate['registry']).parent
    registry = admission.registry_read(folder, gate['owner'])
    target = Path(gate['completed_projection'])
    admission.require(admission.load(target.parent / 'incremental_fence.json') == gate['legacy_fence'],
                      'Completed projection destination is not exclusively fenced')
    completed, states, seen = [], [], set()
    for batch in registry['batches']:
        if batch['status'] != 'COMMITTED': continue
        plan_path = admission.checked(batch['plan']); plan = admission.load(plan_path)
        state_path = plan_path.parent / 'queue/state.json'
        if not state_path.exists(): continue
        state = admission.load(state_path)
        admission.require(state.get('schema') in ('campaign.queue.v5','campaign.queue.v8'), 'Unrecognized real controller state')
        admission.require(state.get('plan') == batch['plan'] and state.get('jobs') == plan['jobs'], 'Controller state/plan identity differs')
        jobs = {j['id']:j for j in plan['jobs']}
        rows = state.get('completed', [])
        admission.require([r['id'] for r in rows] == [j['id'] for j in plan['jobs'][:len(rows)]], 'Completed jobs must be the actual original prefix')
        states.append(admission.bind(state_path))
        for row in rows:
            job = jobs[row['id']]
            if job['phase'] != 'formal': continue
            admission.require(row.get('phase') == 'formal' and row.get('outcome') in ('PASS','FAIL','SKIPPED_DEPENDENCY'),
                              'Unrecognized completed job record')
            if not row.get('receipt'): continue
            admission.require(row['receipt']['path'] == job['receipt'], 'Completed receipt is not the exact original job output')
            admission.checked(row['receipt'])
            admission.require(row['id'] not in seen, 'Duplicate completed formal job across batches')
            seen.add(row['id']); completed.append(copy.deepcopy(row))
    projection = dict(schema='eggpu.incremental.completed.projection.v1', owner=gate['owner'],
        status='COMPLETED_RECEIPT_PROJECTION_ONLY_NOT_PERFORMANCE_SUMMARY', completed=completed,
        source_queue_states=states, gpu_executed_by_projection=False, projection_only=True,
        physical_worker_runs=0, performance_eligible=False,
        meaning='Verbatim completed formal records from bound real v8 queues for unchanged lossless CPU maintenance; no fabricated PASS, timing or aggregate performance count.')
    admission.replace_json(target, projection)
    return projection


def run(spec_path, plan_path, output, controller_lock):
    gate = admission.load(spec_path); plan = admission.load(plan_path)
    with Path(controller_lock).open('a+') as startup_lock:
        fcntl.flock(startup_lock, fcntl.LOCK_EX)
        admission.require(str(controller_lock) == admission.load(admission.checked(gate['parent_manifest']))['controller_lock'], 'Global controller lock changed')
        registry = admission.registry_read(Path(gate['registry']).parent, gate['owner'])
        batches = [b for b in registry['batches'] if b['batch_id'] == gate['batch_id']]
        admission.require(len(batches) == 1 and batches[0]['status'] == 'COMMITTED', 'Batch is not uniquely committed')
        batch = batches[0]
        admission.require(admission.bind(plan_path) == batch['plan'] and admission.bind(spec_path) == batch['gate'],
                          'Runtime plan/spec path or bytes differ from committed batch')
        admission.require(Path(output).resolve() == Path(batch['plan']['path']).parent / 'queue',
                          'Runtime output must be the single reserved batch queue')
        admission.require(plan['gate'] == batch['gate'], 'Wrong batch gate')
        admission.require(plan['required_controller'] == admission.bind(__file__), 'Incremental controller changed')
    admission.checked(gate['supervisor']); base_path = admission.checked(gate['base_controller'])
    module_spec = importlib.util.spec_from_file_location('_frozen_incremental_base_v8', base_path)
    base = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(base)
    original = base.run_process
    def preserve_process(command, env, log, timeout, on_start):
        # Refresh completed records before either the next worker or maintenance;
        # the current active process is never projected as a completed job.
        with Path(controller_lock).open('a+') as probe:
            try: fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: pass
            else:
                fcntl.flock(probe, fcntl.LOCK_UN)
                raise ValueError('Maintenance projection was called without the original controller lock')
        project_completed(gate)
        return original(command, env, log, timeout, on_start)

    base.run_process = preserve_process
    try:
        return base.run(plan_path, output, controller_lock)
    finally:
        # v8 has now saved the final successful or failed maintenance fields and
        # released its lock. Refresh the last job too, without hiding an error.
        with Path(controller_lock).open('a+') as final_lock:
            fcntl.flock(final_lock, fcntl.LOCK_EX)
            project_completed(gate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec','plan','output','controller-lock'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run(args.spec,args.plan,args.output,args.controller_lock)))
