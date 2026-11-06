"""Prepare a new plan document; never submit GPU work or overwrite a plan."""
from pathlib import Path
import argparse
import copy
import hashlib
import json

HERE = Path(__file__).resolve().parent
SOURCE = '71ea757d372f8c1b969ea074caaa27d3a77d4cea6d48cd9c6ae9c4b02936d7ce'


def bind(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def prepare(plan_path, output):
    original = json.loads(Path(plan_path).read_text())
    if original.get('source_tree_sha256') != SOURCE:
        raise ValueError('This amendment requires the frozen unified 71ea source')
    freeze = json.loads((HERE / 'freeze.json').read_text())
    for name, identity in freeze['files'].items():
        actual = bind(HERE / name)
        if any(actual[key] != identity[key] for key in ('bytes', 'sha256')):
            raise ValueError('Frozen derivative changed: ' + name)
    plan = copy.deepcopy(original)
    changed = []
    for job in plan['jobs']:
        if Path(job['receipt']).exists():
            raise ValueError('Plan already has execution evidence; preserve it: ' + job['receipt'])
        command = job['command']
        indices = [i for i, value in enumerate(command) if Path(value).name in ('native_worker.py', 'qualify_static_native_domains.py')]
        if not indices:
            continue
        if len(indices) != 1:
            raise ValueError('Ambiguous worker command')
        index = indices[0]
        old = Path(command[index])
        expected = freeze['parent_files'].get(old.name)
        current = bind(old)
        if not expected or any(current[key] != expected[key] for key in ('bytes', 'sha256')):
            raise ValueError('Parent worker differs from the reviewed derivative base')
        replacement = HERE / 'runner' / old.name
        command[index] = str(replacement)
        if '--worker-sha256' in command:
            command[command.index('--worker-sha256') + 1] = bind(replacement)['sha256']
        job['bindings'] = [item for item in job['bindings'] if Path(item['path']).parent != old.parent]
        job['bindings'] += [bind(HERE / name) for name in freeze['runtime_files']]
        job['bindings'] += [bind(HERE / name) for name in ('freeze.json', 'protocol_amendment.json', 'prepare_plan.py')]
        job['measurement_protocol'] = freeze['protocol']
        job['historical_parent_worker'] = current
        changed.append(job['id'])
    if not changed:
        raise ValueError('No native jobs found')
    plan['native_measurement_amendment'] = dict(protocol=freeze['protocol'], changed_jobs=changed,
        parent_plan=bind(plan_path), amendment=bind(HERE / 'protocol_amendment.json'),
        gpu_executed=False, external_baseline_changes=False, runtime_source_changes=False,
        historical_timings_must_not_be_pooled=True)
    with Path(output).open('x') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
    return dict(status='PREPARED_NOT_EXECUTED', changed_native_jobs=len(changed), total_jobs=len(plan['jobs']), output=bind(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.plan, args.output)))
