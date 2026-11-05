"""CPU-only complete-cell ledger. No admission, derived queue, or worker launch.

A terminal disposition is an explicit primary-agent review of one exact cell;
it is never inferred from a failed receipt. Existing d84 v2 pure proof readers
remain the authority for successful execution and recoverable E4S outputs.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import collections
import hashlib
import importlib.util
import json
import math

PARENT_SHA256 = '467c9b4cddd1c21d43c848b74364c6f817b51da14fd0bc6aba26531030ae54cc'
TERMINAL = {'OOM', 'TO', 'INDEX', 'PLATFORM'}
RESOLVED = {'QUALIFIED'} | TERMINAL
PASS = {'PASS_QUALIFICATION', 'PASS_E4S_QUALIFICATION',
        'PASS_STATIC_NATIVE_NUMERIC_QUALIFICATION'}
PENDING = {'ABSENT', 'RUNNING', 'PENDING', 'WAITING', 'NOT_STARTED'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def parent_helper():
    path = Path(__file__).resolve().parents[1] / 'formal_admission_d84_v2/admit.py'
    require(hashlib.sha256(path.read_bytes()).hexdigest() == PARENT_SHA256,
            'Frozen parent helper changed; review a new candidate instead')
    spec = importlib.util.spec_from_file_location('_disposition_frozen_parent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def same_path(left, right):
    return Path(left).resolve() == Path(right).resolve()


def stage_identity(manifest, group, target, formal_jobs):
    """Includes every original command, binding, guard, timeout and repeat."""
    return digest(dict(source_sha256=manifest['source_sha256'], group=group,
                       target=target, formal_jobs=formal_jobs))


def disposition_index(bindings, binder, cell_ids):
    result = {}
    for binding in bindings:
        record = load(binder.checked(binding))
        require(record.get('schema') == 'eggpu.formal.cell.disposition.v1',
                'Unknown disposition schema')
        cell = record.get('cell_id')
        require(cell in cell_ids, 'Disposition refers to a cell outside fixed matrix')
        require(cell not in result, 'Duplicate disposition for one fixed cell')
        require(record.get('classification') in TERMINAL,
                'Disposition cannot manufacture PASS or waive generic failure')
        review = record.get('review', {})
        require(review.get('authority') == 'primary_agent_explicit' and
                review.get('decision') == 'accept_capability_terminal' and
                isinstance(review.get('rationale'), str) and review['rationale'].strip(),
                'Explicit per-cell primary-agent review is required')
        result[cell] = (record, binding)
    return result


class Auditor:
    def __init__(self, parent, binder, manifest, group):
        self.a = parent
        self.b = binder
        self.manifest = manifest
        self.group = group
        self.cache = {}
        self.freeze = load(binder.checked(manifest['source_freeze']))
        require(self.freeze['source_tree_sha256'] == manifest['source_sha256'],
                'Current source freeze differs from manifest')
        for rel, item in self.freeze['source_files'].items():
            require(binder.content(Path(manifest['source_root']) / rel) == item,
                    'Current source changed: ' + rel)
        binder.checked(manifest['native_dso'])
        self.readers = parent.workers(manifest['proof_readers'], binder)

    def common(self, job):
        a, b = self.a, self.b
        for item in job.get('bindings', []):
            b.checked(item)
        opt = a.options(job['command'])
        proof = load(job['receipt'])
        b.bind(job['receipt'])
        ep = a.current_epoch(proof, job, self.manifest, self.freeze, b)
        worker = next(x for x in job['command'] if x.endswith('.py'))
        expected = [x for x in list(self.manifest['proof_readers'].values()) +
                    job.get('bindings', []) if same_path(x['path'], worker)]
        require(expected, 'Worker is not bound by frozen reader/job bindings')
        for item in expected:
            b.checked(item)
        if 'setup' in opt:
            b.bind(opt['setup'])
        protocol = {'sg': 'e1-host-csr-complete-v1',
                    'native': 'native-host-csr-complete-static-full-profile-v1',
                    'sequence': 'e1-continuous-16-public-csr-v1',
                    'e4s': 'e4s.epoch.v1'}[self.group]
        require(ep.get('schema' if self.group == 'e4s' else 'protocol') == protocol,
                'Qualification timing/protocol epoch differs')
        # Numeric qualification shares the frozen native protocol but has no
        # canonical graph. The protocol check must precede this early return.
        if proof.get('status') == 'PASS_STATIC_NATIVE_NUMERIC_QUALIFICATION':
            return opt, proof, ep, None
        folder = Path(opt['input-root']) / 'canonical' / opt['dataset']
        graph = a.canonical(folder, b)
        # The old reader binds metadata; this inspector also rehashes every input.
        receipt = load(folder / 'receipt.json')
        for item in receipt['files'].values():
            b.checked(item)
        require(proof.get('input_sha256') == graph['input_sha256'],
                'Receipt does not bind exact full canonical input')
        require(proof.get('phase') == opt['phase'] and
                proof.get('dataset') == opt['dataset'], 'Receipt job identity differs')
        return opt, proof, ep, graph

    def audit(self, job):
        key = job['receipt']
        if key in self.cache:
            # Recheck the receipt even when expensive full-vector audit is reused.
            self.b.checked(self.cache[key]['receipt'])
            return self.cache[key]
        a, b = self.a, self.b
        opt, proof, ep, graph = self.common(job)
        require(len(job['expected_statuses']) == 1 and
                proof.get('status') == job['expected_statuses'][0] and
                proof['status'] in PASS, 'Success requires exact frozen PASS status')
        if proof['status'] == 'PASS_STATIC_NATIVE_NUMERIC_QUALIFICATION':
            require(self.group == 'native', 'Numeric proof outside native group')
            audit = a.numeric_artifacts(proof, job, b, self.readers['native_observe'])
        else:
            args = SimpleNamespace(system=opt.get('system', 'eggpu'),
                                   algorithm=opt.get('algorithm'),
                                   count=int(opt.get('count', 1)),
                                   sssp_encoding=opt.get('sssp-encoding', 'u8_exact_u32_exact'),
                                   variant=opt.get('variant'), batch=int(opt.get('batch', 0)))
            if self.group == 'sg':
                self.readers['sg'](key, kind=opt['phase'], system=args.system,
                                   algorithm=args.algorithm, epoch_sha=proof['epoch_sha256'],
                                   graph=graph)
            elif self.group == 'e4s':
                self.readers['e4s'](Path(key), opt['phase'], args,
                                    proof['epoch_sha256'], graph, proof['source_sequence'])
            else:
                self.readers[self.group](key, opt['phase'], args,
                                         proof['epoch_sha256'], graph)
            audit = a.call_artifacts(proof, self.group, graph, b)
            if self.group == 'e4s':
                require(not self.manifest.get('policy', {}).get(
                    'e4s_formal_archive_repair_pending', True),
                    'E4S recoverable archive contract is not ready')
                sidecar = Path(key).parent / 'recoverable_outputs.json'
                require(sidecar.is_file(), 'E4S recoverable output sidecar is missing')
                recovery, _ = a.recovery_modules(self.manifest['archive_contract'], b)
                audit['recoverable_outputs'] = recovery.validate_sidecar(
                    key, sidecar, self.manifest['archive_contract']['helper'])
                b.bind(sidecar)
                for vector in load(sidecar)['vectors']:
                    for kind in ['object', 'reference_metadata']:
                        b.checked(vector[kind])
        result = dict(receipt=b.bind(key), epoch=proof['epoch'],
                      epoch_sha256=proof['epoch_sha256'], audit=audit)
        self.cache[key] = result
        return result

    def cell_success(self, target, tiny, formal_jobs):
        tiny_audit = self.audit(tiny)
        target_audit = self.audit(target)
        t = load(tiny['receipt'])
        p = load(target['receipt'])
        require(t['epoch_sha256'] == p['epoch_sha256'], 'Tiny/target old or different epoch')
        require(t['directed'] == p['directed'], 'Tiny directedness differs')
        require(p.get('tiny_proof', p.get('qualification', {}).get('tiny')) ==
                tiny_audit['receipt'], 'Target did not bind exact intended tiny proof')
        return dict(tiny=tiny_audit, target=target_audit)

    def formal_contract(self, target, formal_jobs):
        """All branches retain formal file bindings and exact E4S input sequence."""
        p = load(target['receipt']) if Path(target['receipt']).is_file() else None
        for job in formal_jobs:
            for binding in job.get('bindings', []):
                self.b.checked(binding)
            if self.group == 'e4s' and p is not None and p.get('status') not in PENDING:
                contract = job['target_contract']
                require(p['source_sequence'] == contract['source_sequence'] and
                        p['input_sha256'] == contract['input_sha256'],
                        'E4S formal source sequence/input differs')


def marker_checked(marker, logs, binder):
    require(isinstance(marker, dict) and isinstance(marker.get('text'), str) and
            marker['text'].strip(), 'Exact original-log error marker is required')
    matches = [x for x in logs if same_path(x['path'], marker.get('path', ''))]
    require(len(matches) == 1, 'Error marker must refer to one bound raw log')
    path = binder.checked(matches[0])
    needle = marker['text'].encode()
    require(len(needle) <= 65536, 'Error marker is unexpectedly large')
    tail = b''
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            data = tail + chunk
            if needle in data:
                return
            tail = data[-len(needle):]
    raise ValueError('Reviewed error marker is absent from bound original log')


def correctness_failure(proof, binder):
    """A default quality_pass=False on exceptions is not a measured mismatch."""
    def walk(value):
        if isinstance(value, dict):
            if value.get('status') in {'FAIL_QUALITY', 'FAIL_CORRECTNESS', 'FAILED_CORRECTNESS'}:
                return True
            quality = value.get('quality')
            if isinstance(quality, dict) and (
                    quality.get('pass') is False or quality.get('passed') is False):
                return True
            return any(walk(x) for x in value.values())
        return isinstance(value, list) and any(walk(x) for x in value)
    if walk(proof):
        return True
    for source in proof.get('sources', []):
        item = source.get('result')
        if isinstance(item, dict) and {'path', 'bytes', 'sha256'} <= set(item):
            if walk(load(binder.checked(item))):
                return True
    return False


def terminal_cell(record, binding, auditor, target, tiny, cell_identity):
    """Validate an explicit review; do not infer classifications from log text."""
    b, a = auditor.b, auditor.a
    require(record['cell_id'] == target['id'] and
            record.get('cell_identity_sha256') == cell_identity,
            'Disposition is for old/different command, repeat, resource or cell identity')
    require(record.get('source_sha256') == auditor.manifest['source_sha256'],
            'Disposition source epoch is old/different')
    evidence = record.get('evidence', {})
    for key, job in [('target_receipt', target), ('tiny_receipt', tiny)]:
        require(key in evidence and same_path(evidence[key]['path'], job['receipt']),
                'Disposition must bind exact original ' + key)
        b.checked(evidence[key])
    target_proof = load(target['receipt'])
    require(target_proof.get('status') not in PASS | PENDING,
            'Success, missing and running receipts cannot be capability terminals')
    require(not correctness_failure(target_proof, b),
            'Observed numerical correctness failure cannot be waived as capability')
    tiny_audit = auditor.audit(tiny)
    opt, proof, ep, graph = auditor.common(target)
    require(proof['epoch_sha256'] == tiny_audit['epoch_sha256'],
            'Capability terminal used old/different tiny epoch')
    require(proof.get('directed') == load(tiny['receipt']).get('directed') == graph['directed'],
            'Capability terminal tiny/target/canonical directedness differs')
    require(proof.get('tiny_proof', proof.get('qualification', {}).get('tiny')) ==
            tiny_audit['receipt'], 'Failed target did not bind intended passing tiny')
    require(record.get('epoch_sha256') == proof['epoch_sha256'] and
            record.get('input_sha256') == graph['input_sha256'],
            'Disposition epoch or full input identity differs')
    require(isinstance(record.get('failure_stage'), str) and record['failure_stage'].strip(),
            'Precise failure stage is required')
    logs = evidence.get('raw_logs', [])
    require(logs and len({str(Path(x['path']).resolve()) for x in logs}) == len(logs),
            'Unique original failure logs are required')
    for item in logs:
        b.checked(item)
    marker_checked(record.get('error_marker'), logs, b)
    resource = load(b.checked(evidence['resource_observation']))
    require(resource.get('schema') == 'eggpu.reviewed.failure.resources.v1' and
            resource.get('cell_id') == target['id'] and
            resource.get('source_sha256') == record['source_sha256'] and
            resource.get('epoch_sha256') == record['epoch_sha256'] and
            resource.get('input_sha256') == record['input_sha256'] and
            resource.get('failure_stage') == record['failure_stage'],
            'Reviewed runtime resource observation differs from failed cell')
    require(resource.get('execution_attempted') is True and
            resource.get('unrelated_gpu_contention') is False and
            resource.get('disk_preflight_passed') is True and
            resource.get('host_preflight_passed') is True,
            'Missing resources/infra/contention cannot be waived as capability')
    raw_resources = resource.get('evidence')
    require(isinstance(raw_resources, dict) and raw_resources and
            all(isinstance(x, dict) and {'path', 'bytes', 'sha256'} <= set(x)
                for x in raw_resources.values()),
            'Runtime resource facts require actual raw file bindings')
    a.epoch_bound_artifacts(raw_resources, b)
    require(same_path(evidence['setup']['path'], opt.get('setup', '')),
            'Disposition must bind frozen setup command path')
    b.checked(evidence['setup'])
    expected_gpus = str(opt.get('gpu-uuids', opt.get('gpu-uuid', ''))).split(',')
    require(expected_gpus != [''] and resource.get('gpu_uuids') == expected_gpus,
            'Failure resource GPU identity differs from frozen command')
    facts = record.get('facts', {})
    kind = record['classification']
    if kind == 'OOM':
        require(facts.get('memory_domain') == 'device' and
                facts.get('error_class') in {'cudaErrorMemoryAllocation',
                                             'CUDA_ERROR_OUT_OF_MEMORY',
                                             'rmm::bad_alloc', 'std::bad_alloc_device'},
                'OOM requires an explicit device-allocation failure, not host/disk failure')
        require(facts['error_class'] in record['error_marker']['text'],
                'Device allocation error is not present in reviewed marker')
    elif kind == 'TO':
        require(any(x in record['error_marker']['text'].lower()
                    for x in ['timeout', 'timed out']),
                'TO requires an original timeout marker, not another failure class')
        limit = facts.get('frozen_timeout_seconds')
        elapsed = facts.get('observed_elapsed_seconds')
        require(isinstance(limit, (int, float)) and not isinstance(limit, bool) and
                math.isfinite(limit) and limit > 0 and limit == target.get('timeout'),
                'TO currently supports only the exact frozen controller timeout')
        require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and
                math.isfinite(elapsed) and elapsed >= limit and
                facts.get('owned_processes_terminated') is True,
                'Timeout requires elapsed deadline and owned-process cleanup evidence')
    elif kind == 'INDEX':
        require('INDEX_UNSUPPORTED' in record['error_marker']['text'],
                'INDEX requires explicit original INDEX_UNSUPPORTED evidence')
        required, maximum = facts.get('required_entries'), facts.get('supported_max_entries')
        require(type(required) is int and type(maximum) is int and
                required > maximum >= 0 and facts.get('index_bits') in (32, 64) and
                facts.get('index_domain') in {'global_offsets', 'local_offsets', 'local_vertices'},
                'INDEX requires actual domain overflow, not a generic input rejection')
        geometry = load(b.checked(evidence['index_geometry']))
        require(geometry.get('input_sha256') == graph['input_sha256'] and
                geometry.get('all_arcs_retained') is True and
                geometry.get('required_entries') == required and
                geometry.get('supported_max_entries') == maximum and
                geometry.get('index_domain') == facts['index_domain'],
                'Index geometry is not bound to complete graph and exact limit')
    elif kind == 'PLATFORM':
        require(facts.get('required_feature') == 'peer_access' and
                facts.get('native_error_code') == 217 and
                facts.get('author_feature_enabled') is True,
                'PLATFORM v1 supports only reviewed author peer-access error 217')
        require('217' in record['error_marker']['text'],
                'Peer access error code absent from bound raw marker')
    return dict(disposition=binding, classification=kind, failure_stage=record['failure_stage'],
                tiny=tiny_audit, failed_target=b.bind(target['receipt']),
                epoch=proof['epoch'], input_sha256=graph['input_sha256'],
                evidence=evidence, facts=facts, review=record['review'])


def inspect(request_path):
    a, helper_path = parent_helper()
    b = a.Binder()
    request_binding = b.bind(request_path)
    request = load(request_path)
    require(request.get('schema') == 'eggpu.formal.disposition.inspect.v1', 'Unknown request schema')
    group = request.get('group')
    require(group in {'sg', 'native', 'sequence', 'e4s'}, 'Unsupported group')
    manifest = load(b.checked(request['manifest']))
    require(manifest.get('schema') == 'eggpu.d84.formal.admission.v1', 'Unknown parent manifest')
    spec = manifest['groups'][group]
    formal = load(b.checked(spec['pending']))
    plans = [load(b.checked(p)) for p in spec['qualifications']]
    qualifications, mapping = a.structural_gate(formal, plans)
    require(len(formal['jobs']) == spec['expected_formal_jobs'], 'Original formal cardinality differs')
    require(formal['source_tree_sha256'] == manifest['source_sha256'] and
            all(p['source_tree_sha256'] == manifest['source_sha256'] for p in plans),
            'Qualification/pending source is old/different')
    require(all(p.get('environment', {}) == formal.get('environment', {}) for p in plans),
            'Qualification/pending environment differs')
    target_by_id = {j['id']: j for j in qualifications.values()
                    if a.options(j['command']).get('phase') == 'target'}
    formal_by_target = collections.defaultdict(list)
    for j in formal['jobs']:
        formal_by_target[mapping[j['id']]['id']].append(j)
    require(set(formal_by_target) == set(target_by_id), 'Omitted or extra original target cell')
    if 'expected_cell_ids' in request:
        claimed = request['expected_cell_ids']
        require(len(claimed) == len(set(claimed)) and set(claimed) == set(target_by_id),
                'Requested cell list omits/duplicates a frozen target')
    dispositions = disposition_index(request.get('dispositions', []), b, target_by_id)
    receipts = {j['receipt']: j for j in qualifications.values()}
    inventory = []
    for job in qualifications.values():
        path = Path(job['receipt'])
        item = dict(id=job['id'], intended_receipt=str(path),
                    expected_statuses=job['expected_statuses'], job_identity_sha256=digest(job))
        if path.is_file():
            item['receipt'] = b.bind(path)
            try:
                item['observed_status'] = load(path).get('status', 'MALFORMED')
            except (OSError, ValueError, TypeError):
                item['observed_status'] = 'MALFORMED'
        else:
            item['observed_status'] = 'ABSENT'
        inventory.append(item)
    observed = {x['id']: x['observed_status'] for x in inventory}
    global_errors = []
    auditor = None
    try:
        auditor = Auditor(a, b, manifest, group)
    except (ValueError, OSError, KeyError, TypeError, StopIteration) as error:
        global_errors.append(dict(stage='source_runner_epoch_initialization', reason=str(error)))
    cells = []
    covered_qualifications = set()
    for cell_id, target in target_by_id.items():
        jobs = formal_by_target[cell_id]
        identity = stage_identity(manifest, group, target, jobs)
        opt = a.options(target['command'])
        tiny = receipts[opt['tiny-proof']]
        covered_qualifications.update([target['id'], tiny['id']])
        # Keep original complete jobs rather than a lossy command summary.
        row = dict(cell_id=cell_id, cell_identity_sha256=identity,
                   source_sha256=manifest['source_sha256'], target_job=target, tiny_job=tiny,
                   formal_slots=[dict(original_job=j, time_seconds=None,
                                      performance_measured=False, counts_as_win=False,
                                      execution_scheduled=False) for j in jobs],
                   status='PENDING', observed_status=observed[cell_id],
                   time_seconds=None, counts_as_win=False, formal_admitted=False)
        try:
            if auditor is None:
                row.update(status='INFRA', reason='Current source/runner initialization failed')
            else:
                auditor.formal_contract(target, jobs)
                if cell_id in dispositions:
                    rec, bound = dispositions[cell_id]
                    row['proof_audit'] = terminal_cell(rec, bound, auditor, target, tiny, identity)
                    row['status'] = rec['classification']
                elif observed[cell_id] in PASS:
                    row['proof_audit'] = auditor.cell_success(target, tiny, jobs)
                    row['status'] = 'QUALIFIED'
                elif observed[cell_id] in PENDING:
                    row['reason'] = 'Qualification is missing or not terminal'
                else:
                    failed = load(target['receipt'])
                    row['status'] = 'CORRECTNESS' if correctness_failure(failed, b) else 'INFRA'
                    row['reason'] = 'Unresolved failure; no explicit reviewed capability disposition'
        except (ValueError, OSError, KeyError, TypeError, StopIteration) as error:
            row.update(status='INFRA', reason=str(error))
        cells.append(row)
    # Numeric and any other global qualifiers cannot be hidden by cell dispositions.
    extra_audits = []
    for job in qualifications.values():
        if job['id'] in covered_qualifications:
            continue
        try:
            require(auditor is not None, 'Source/runner initialization failed')
            extra_audits.append(dict(id=job['id'], audit=auditor.audit(job)))
        except (ValueError, OSError, KeyError, TypeError, StopIteration) as error:
            global_errors.append(dict(stage='global_qualification', job=job['id'], reason=str(error)))
    cpu = None
    try:
        cpu = a.cpu_gate(manifest, b)
    except (ValueError, OSError, KeyError, TypeError) as error:
        global_errors.append(dict(stage='unchanged_parent_cpu_gate', reason=str(error)))
    counts = dict(collections.Counter(x['status'] for x in cells))
    all_resolved = all(x['status'] in RESOLVED for x in cells) and not global_errors
    require(len(cells) == len(target_by_id) and
            sum(len(x['formal_slots']) for x in cells) == len(formal['jobs']),
            'Ledger lost an original cell or formal repeat')
    # No code path writes or returns an executable plan. All derived times null.
    return dict(schema='eggpu.formal.disposition.ledger.v1',
                status='INSPECTED_NOT_ADMITTED', group=group,
                all_cells_resolved_and_proofs_checked=all_resolved,
                formal_admitted=False, workers_launched=0, gpu_executed=False,
                source_sha256=manifest['source_sha256'], original_cell_count=len(cells),
                original_formal_slots=len(formal['jobs']), repetitions=list(range(10)),
                counts=counts, cells=cells, qualification_inventory=inventory,
                extra_qualification_audits=extra_audits, global_blockers=global_errors,
                parent_cpu_gate=cpu, request=request_binding, parent_manifest=request['manifest'],
                parent_helper=b.bind(helper_path), input_bindings=list(b.seen.values()),
                hashed_bytes=b.hashed_bytes,
                limitation='Inspect/ledger only: no admitted plan or controller-lock commit; '
                           'even an all-resolved ledger requires a separately reviewed admission implementation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite any existing report')
    report = inspect(args.request)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], counts=report['counts'],
                          all_cells_resolved_and_proofs_checked=report['all_cells_resolved_and_proofs_checked'],
                          formal_admitted=False, workers_launched=0, output=str(args.output))))


if __name__ == '__main__':
    main()
