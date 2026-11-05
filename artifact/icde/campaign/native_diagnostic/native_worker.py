"""Fixed-native E1-MG / E2 worker. No GPU work without --execute-gpu.

T_first starts from qualified host CSR and includes weak projection, actual
partition construction, manifest import, runtime startup and complete output.
T_ready measures a fresh complete query after one explicit warmup. Native P1
is always native, never the static single-GPU arena route.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import importlib
import json
import os
import shutil
import subprocess
import sys
import time

import sg_worker as shared
import native_input
import native_profile

E2_DATASETS = ('US', 'OR', 'IN', 'UK', 'TW')
PROTOCOL = 'native-host-csr-complete-static-full-profile-v1'
NATIVE_PR_MAX_ITER = 1000  # Actual _native.py API, not the campaign's allowed upper cap.


def make_tiny(root):
    """The native fixture has 16 vertices so all P=1/2/4/8 are executable."""
    import numpy as np
    shared.make_tiny(root)
    for name in shared.TINY_DATASETS:
        directory = Path(root) / 'canonical' / name
        offsets = np.fromfile(directory / 'offsets.bin', dtype='<i4')
        np.r_[offsets, np.full(9, offsets[-1], dtype='<i4')].astype('<i4').tofile(directory / 'offsets.bin')
        metadata = json.loads((directory / 'canonical.json').read_text())
        metadata['num_nodes'] = 16
        shared.save(directory / 'canonical.json', metadata)
        receipt = json.loads((directory / 'receipt.json').read_text())
        receipt['native_partition_fixture'] = {'vertices': 16, 'gpu_counts': [1, 2, 4, 8]}
        for role, item in receipt['files'].items():
            receipt['files'][role] = shared.binding(directory / Path(item['path']).name)
        shared.save(directory / 'receipt.json', receipt)


def configure(args):
    if args.sssp_encoding not in ('f64', 'u8_exact_u32_exact'):
        raise ValueError('SSSP encoding must be explicit and fixed across all GPU counts')
    args.communication = args.communication or ('nccl' if args.algorithm == 'pr' else 'host_staging')
    if args.communication not in ('host_staging', 'nccl', 'peer_to_peer'):
        raise ValueError('Native communication must be explicit and fixed across GPU counts')
    uuids = args.gpu_uuids.split(',')
    if args.count not in (1, 2, 4, 8) or len(uuids) != args.count or len(set(uuids)) != args.count:
        raise ValueError('Supply exactly P distinct GPU UUIDs, with P=1/2/4/8')
    if any(not value.startswith('GPU-') or value.strip() != value for value in uuids):
        raise ValueError('Use explicit GPU UUIDs without whitespace')
    if args.experiment == 'E1-MG' and args.count != 8:
        raise ValueError('The frozen E1-MG main panel uses eight GPUs')
    if (args.dataset not in shared.TINY_DATASETS and args.experiment == 'E2' and args.dataset not in E2_DATASETS
            and not (args.phase == 'diagnostic' and args.count == 1)):
        raise ValueError('Frozen E2 datasets are US/OR/IN/UK/TW')
    if (args.dataset in shared.TINY_DATASETS) != (args.phase == 'tiny'):
        raise ValueError('Synthetic fixture and main-graph phases must remain separate')
    if args.repeat < 0 or (args.phase == 'formal' and args.repeat not in range(10)):
        raise ValueError('Formal repeat is one of 0..9')
    args.uuids = uuids
    native_profile.install(args)
    return args


def runtime_dependencies(path):
    """Bind resolved native runtime libraries; ASLR addresses are not identity."""
    output = subprocess.check_output(['ldd', str(Path(path).resolve())], text=True)
    dependencies = {}
    for line in output.splitlines():
        line = line.strip()
        if '=> not found' in line:
            raise RuntimeError('Native runtime dependency unresolved: ' + line.split('=>')[0].strip())
        if '=>' in line:
            name, value = line.split('=>', 1)
            target = value.strip().split()[0]
        else:
            target = line.split()[0] if line else ''
            name = Path(target).name
        if target.startswith('/'):
            dependencies[name.strip()] = shared.binding(target)
    if not dependencies:
        raise RuntimeError('Could not bind the native DSO runtime dependencies')
    return dependencies


def epoch(args):
    # Reuse the same actual-package and public-source hashing, without claiming
    # the arena binary is executed by this forced-native worker.
    proxy = SimpleNamespace(**vars(args))
    proxy.system, proxy.gpu_uuid = 'native', args.uuids[0]
    data, _ = shared.source_epoch(proxy)
    data.update(protocol=PROTOCOL, system='eggpu', implementation='forced_native',
                gpu_count=args.count, gpu_uuids=args.uuids,
                native_route='static_immutable' if args.algorithm=='sssp' else 'persistent_full',
                communication=args.communication,
                sssp_encoding=args.sssp_encoding, sssp_session='Session.native_static',
                sssp_delta_rule=args.sssp_delta_rule, native_profile=native_profile.identity(args),
                sssp_single_owner_execution='off',
                inventory_mode='recheck', retained_budget_mib=args.retained_budget_mib,
                pr_max_iter=NATIVE_PR_MAX_ITER, pr_tolerance_l1=shared.PR_TOLERANCE,
                preparation='actual weak projection and equal-arc contiguous partitions inside T_first')
    data.pop('arena_budget_mib', None)
    data['gpu_admission_minimum_free_mib'] = args.budget_mib + 512
    data['environment'].update({key: value for key, value in os.environ.items()
        if key.startswith(('EGMGPU_', 'NCCL_', 'UCX_')) or key == 'LD_PRELOAD'})
    data['files'].update(native_worker=shared.binding(__file__),
        native_input=shared.binding(Path(__file__).with_name('native_input.py')),
        native_profile=shared.binding(Path(__file__).with_name('native_profile.py')),
        historical_profile=shared.binding(Path(__file__).with_name('historical_profile.json')),
        native_dso=shared.binding(args.native_dso))
    data['runtime_dependencies'] = runtime_dependencies(args.native_dso)
    data['cgroup_resource_limits'] = []
    for row in shared.cgroup_memory()['cgroup_limits']:
        cpu = Path(row['path']) / 'cpu.max'
        data['cgroup_resource_limits'].append({'path': row['path'], 'memory_max': row['max'],
                                              'cpu_max': cpu.read_text().strip() if cpu.exists() else None})
    for base in (args.release / 'gpu_easygraph',):
        if base.is_dir():
            for path in sorted(base.rglob('*')):
                if path.is_file() and path.suffix in ('.py', '.cu', '.cpp', '.c', '.h', '.hpp', '.cuh'):
                    data['release_sources'][str(path.relative_to(args.release))] = shared.content_binding(path)
    return data, shared.digest_json(data)


def admission(args, graph):
    rows = []
    for uuid in args.uuids:
        proxy = SimpleNamespace(**vars(args)); proxy.gpu_uuid = uuid
        rows.append(shared.resource_check(proxy, graph))
    return rows


def validate_proof(path, kind, args, digest, graph):
    item = shared.validate_proof(path, kind=kind, system='eggpu',
                               algorithm=args.algorithm, epoch_sha=digest, graph=graph)
    proof = json.loads(Path(path).read_text())
    if proof.get('implementation') != 'forced_native' or proof.get('gpu_count') != args.count:
        raise ValueError('Qualification did not execute this native GPU count')
    for source in proof['sources']:
        child = json.loads(Path(source['result']['path']).read_text())
        if child.get('partition_audit', {}).get('status') != 'PASS_FULL_PARTITION_ROUNDTRIP':
            raise ValueError('Native partition round-trip evidence is incomplete')
        manifest = child['preparation']['manifest']
        if shared.binding(manifest['path']) != {key: manifest[key] for key in ('path', 'bytes', 'sha256')}:
            raise ValueError('Qualified native manifest changed')
        for label in ('first', 'warmup', 'ready'):
            metrics=child['calls'][label]['native_metrics']
            native_profile.observe(metrics, args.algorithm, args.count)
            if metrics.get('devices_used') != list(range(args.count)):
                raise ValueError('Qualified native call did not use the requested devices')
            if args.algorithm=='sssp':
                expected=('u8_exact','u32_exact') if args.sssp_encoding!='f64' else ('f64','f64')
                if ((metrics.get('weight_encoding'),metrics.get('wire_encoding'))!=expected
                        or metrics.get('static_sssp_encoding')!=args.sssp_encoding
                        or metrics.get('single_owner_execution_used') is not False):
                    raise ValueError('Native proof lacks the actual frozen static encoding observations')
                if args.sssp_encoding!='f64':
                    certificate=metrics.get('static_sssp_input_certificate',{})
                    if (metrics.get('wire_integer_bound_certified') is not True
                            or certificate.get('finite_distance_upper_bound')!=metrics.get('wire_distance_bound')
                            or not 0<=certificate.get('finite_distance_upper_bound',2**32)<2**32-1):
                        raise ValueError('Native proof lacks complete compact numeric certification')
    return item


def json_metric_values(value):
    """Represent documented nonfinite metric sentinels without changing answers."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return {'numeric_class': 'nan' if math.isnan(value) else 'positive_infinity' if value > 0 else 'negative_infinity'}
    if isinstance(value, dict): return {k: json_metric_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_metric_values(v) for v in value]
    return value


class NativeBackend:
    def __init__(self, args, graph, source):
        begin = time.perf_counter()
        if os.environ.get('EGGPU_COMM_BACKEND') != args.communication:
            raise RuntimeError('Native communication environment differs from the explicit job option')
        sys.path.insert(0, str(args.release.resolve()))
        sys.path.insert(1, str(args.native_dso.resolve().parent))
        from easygraph.gpu import Session
        cpp = importlib.import_module('cpp_easygraph')
        if Path(cpp.__file__).resolve() != args.native_dso.resolve():
            raise RuntimeError('Imported DSO differs from the source-bound native extension')
        if Path(sys.modules[Session.__module__].__file__).resolve() != (args.release / 'easygraph/gpu/session.py').resolve():
            raise RuntimeError('Imported Session differs from the public source')
        prepared = native_input.original_graph(graph, args.dataset)
        projection = None
        projection_started = time.perf_counter()
        if args.algorithm == 'wcc' and graph['directed']:
            prepared, projection = native_input.weak_projection(prepared, args.output / 'weak')
        projection_seconds = time.perf_counter() - projection_started
        partition_started = time.perf_counter()
        manifest = native_input.native_partitions(prepared, args.output / 'partitions', args.count,
            native_input.load_signer(args.release), weighted=args.algorithm == 'sssp')
        partition_seconds = time.perf_counter() - partition_started
        self.preparation = dict(manifest=manifest, projection=projection,
            projection_seconds=projection_seconds, partition_seconds=partition_seconds,
            adapter_including_import_seconds=time.perf_counter()-begin,
            timing='inside T_first; full independent partition audit is outside timing')
        self.audit_graph = prepared
        budgets = None if args.retained_budget_mib == 0 else {
            device: args.retained_budget_mib << 20 for device in range(args.count)}
        if args.algorithm == 'sssp':
            self.session = Session.native_static({'g': manifest['path']}, devices=tuple(range(args.count)), cpp=cpp,
                sssp_encoding=args.sssp_encoding, sssp_delta_rule=args.sssp_delta_rule,
                inventory_mode='recheck', budgets=budgets)
        else:
            self.session = Session.native({'g': manifest['path']}, devices=tuple(range(args.count)), cpp=cpp,
                                          route='persistent_full', inventory_mode='recheck', budgets=budgets)
        self.sssp_encoding = args.sssp_encoding
        self.algorithm, self.source, self.count = args.algorithm, source, args.count
        self.last_metrics = {}

    def query(self):
        parameters = {'source': self.source} if self.algorithm in ('bfs', 'sssp') else {}
        result = self.session.query('g', self.algorithm, parameters)
        if result.backend_metrics.get('devices_used') != list(range(self.count)):
            raise ValueError('Actual native devices_used differs from requested P')
        observed = native_profile.observe(result.backend_metrics, self.algorithm, self.count,
                                          previous=self.last_metrics or None)
        self.last_metrics = json_metric_values(dict(result.backend_metrics, frozen_native_profile_observation=observed))
        if self.algorithm == 'sssp':
            expected = ('u8_exact', 'u32_exact') if self.sssp_encoding != 'f64' else ('f64', 'f64')
            if (self.last_metrics.get('weight_encoding'), self.last_metrics.get('wire_encoding')) != expected:
                raise ValueError('Measured SSSP encoding differs from the frozen runner policy')
            if self.last_metrics.get('single_owner_execution_used') is not False:
                raise ValueError('Fixed-native strong scaling cannot use the separate P1 engine')
        if self.algorithm == 'pr':
            if (result.contract.max_iterations != NATIVE_PR_MAX_ITER
                    or result.contract.damping != shared.PR_ALPHA
                    or result.contract.tolerance_per_vertex != shared.PR_TOLERANCE / len(result.values)):
                raise ValueError('Native PR contract changed; explicitly rebind and qualify the new API')
        return result.values

    def close(self):
        self.session.close()


def source_child(args):
    import numpy as np
    if args.lease_fd is None:
        raise RuntimeError('Native child requires inherited exclusive global lease')
    actual, inherited = args.lock_file.stat(), os.fstat(args.lease_fd)
    if (actual.st_dev, actual.st_ino) != (inherited.st_dev, inherited.st_ino):
        raise RuntimeError('Lease descriptor does not identify the global GPU lock')
    with args.lock_file.open('a') as independent:
        try:
            fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Parent does not hold the exclusive GPU lease')
    graph = shared.load_input(args.input_root / 'canonical' / args.dataset)
    source = graph['sources'][args.source_index] if args.algorithm in ('bfs', 'sssp') else None
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    record = dict(status='RUNNING', source=source, source_index=args.source_index, gpu_count=args.count,
                  implementation='forced_native', input_sha256=graph['input_sha256'], calls={},
                  timing_start='common immutable host CSR ready; fresh Python process for every source')
    shared.save(out / 'result.json', record)
    backend = None
    try:
        record['host_ready_checksum'] = [int(graph['offsets'].sum(dtype=np.int64)),
            int(graph['indices'].sum(dtype=np.int64)), float(graph['weights'].sum(dtype=np.float64))]
        outputs = {}
        for label in ('first', 'warmup', 'ready'):
            shared.save(out / 'stage.json', {'active': True, 'label': label, 'started_monotonic': time.monotonic()})
            with shared.complete_timeout(1800):
                started = time.perf_counter()
                if label == 'first':
                    backend = NativeBackend(args, graph, source)
                outputs[label] = backend.query()
                seconds = time.perf_counter() - started
            shared.save(out / 'stage.json', {'active': False, 'label': label})
            record['calls'][label] = ({'seconds': seconds} if label != 'warmup' else {'timed': False, 'count': 1})
            record['calls'][label]['native_metrics'] = backend.last_metrics
        record['preparation'] = backend.preparation
        prepared_graph = backend.audit_graph
        backend.close(); backend = None
        record['partition_audit'] = native_input.audit_partitions(prepared_graph,
                                                record['preparation']['manifest']['path'])
        expected, reference_receipt = shared.reference_cached(graph, args.algorithm, source, args.reference_cache)
        record['reference_receipt'] = reference_receipt
        for label, values in outputs.items():
            if any(np.shares_memory(values, other) for key, other in outputs.items() if key != label):
                raise ValueError('Native output arrays alias across complete calls')
            record['calls'][label]['output'] = shared.persist_output(out, values)
            record['calls'][label]['quality'] = shared.audit_output(values, expected, graph, args.algorithm)
        record['quality_pass'] = all(call['quality']['pass'] for call in record['calls'].values())
        record['status'] = 'PASS' if record['quality_pass'] else 'FAIL_QUALITY'
        record['outputs_survive_close'] = True
        if record['quality_pass'] and not args.keep_prepared:
            record['generated_storage'] = native_input.remove_generated_payloads(
                record['preparation']['manifest']['path'], record['preparation']['projection'])
        else:
            record['generated_storage'] = {'retained': 'all generated partition/projection payloads'}
    except BaseException as error:
        status = 'INDEX_UNSUPPORTED' if isinstance(error, native_input.IndexUnsupported) else (
                 'TIMEOUT' if isinstance(error, TimeoutError) else 'FAILED')
        record.update(status=status, error=type(error).__name__ + ': ' + str(error), quality_pass=False)
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as error:
                record.update(status='FAILED_CLOSE', quality_pass=False, close_error=type(error).__name__+': '+str(error))
        shared.save(out / 'result.json', record)
    return record


def forward_options(args):
    values = ['--execute-gpu', '--experiment', args.experiment, '--algorithm', args.algorithm,
              '--dataset', args.dataset, '--phase', args.phase, '--repeat', str(args.repeat),
              '--count', str(args.count), '--gpu-uuids', args.gpu_uuids]
    values += ['--communication', args.communication]
    values += ['--sssp-encoding', args.sssp_encoding]
    values += ['--sssp-delta-rule', args.sssp_delta_rule, '--native-profile', args.native_profile]
    for name in ('input_root', 'release', 'native_dso', 'setup', 'lock_file', 'reference_cache',
                 'budget_mib', 'retained_budget_mib', 'min_host_free_gib'):
        values += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    if args.keep_prepared:
        values.append('--keep-prepared')
    return values


def job(args):
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(schema='eggpu.native.job.v1', protocol=PROTOCOL, experiment=args.experiment,
        phase=args.phase, system='eggpu', implementation='forced_native', dataset=args.dataset,
        algorithm=args.algorithm, repeat=args.repeat, gpu_count=args.count, status='RUNNING',
        quality_pass=False, performance_eligible=False, sources=[], started_unix=time.time(),
        complete_task_timeout_seconds=1800,
        native_route='static_immutable' if args.algorithm=='sssp' else 'persistent_full',
        communication=args.communication,
        sssp_encoding=args.sssp_encoding,
        sssp_delta_rule=args.sssp_delta_rule, native_profile=native_profile.identity(args),
        pr_max_iter_actual=NATIVE_PR_MAX_ITER, pr_max_iter_campaign_upper_bound=shared.PR_MAX_ITER,
        retained_budget_mib=args.retained_budget_mib,
        retained_budget_meaning='per-device retained state only; 0 means no extra software cap; actual device capacity still applies')
    shared.save(args.output / 'result.json', record)
    try:
        if not args.lock_file.parent.is_dir():
            raise ValueError('Use the existing machine-wide GPU lock path')
        with args.lock_file.open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            graph = shared.load_input(args.input_root / 'canonical' / args.dataset)
            if graph['n'] < args.count:
                raise ValueError('Native tiny fixture must contain at least P vertices; use --make-tiny-native')
            # At most one source's generated partition payload is live by default.
            # Full raw outputs and all content-bound manifests remain retained.
            source_count = 3 if args.algorithm in ('bfs', 'sssp') else 1
            generated = (12 if args.algorithm == 'sssp' else 16 if args.algorithm == 'wcc' else 4) * graph['m']
            generated += 16 * args.count * graph['n'] + (4 << 30)
            if args.keep_prepared:
                generated *= source_count
            if shutil.disk_usage(args.output).free < generated + 24 * graph['n'] * source_count:
                raise RuntimeError('Insufficient disk for actual partition staging and retained full outputs')
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.uuids)
            os.environ['EGGPU_COMM_BACKEND'] = args.communication
            data, digest = epoch(args)
            shared.save(args.output / 'epoch.json', data)
            record.update(epoch=shared.binding(args.output / 'epoch.json'), epoch_sha256=digest,
                          input=graph['identity'], input_sha256=graph['input_sha256'], directed=graph['directed'])
            if args.phase in ('target', 'formal'):
                if args.tiny_proof is None:
                    raise ValueError('A current GPU tiny proof for the same native P is required')
                record['tiny_proof'] = validate_proof(args.tiny_proof, 'tiny', args, digest, graph)
            if args.phase == 'formal':
                if args.target_proof is None:
                    raise ValueError('A current target-input proof for the same native P is required')
                record['target_proof'] = validate_proof(args.target_proof, 'target', args, digest, graph)
            for index in range(source_count):
                resources = admission(args, graph)
                child = args.output / f'source_{index}'
                command = [sys.executable, str(Path(__file__).resolve()), *forward_options(args),
                    '--source-index', str(index), '--lease-fd', str(lease.fileno()), '--output', str(child)]
                with (args.output / f'source_{index}.log').open('wb') as log:
                    code = shared.run_child(command, log, lease.fileno(), child, 1800)
                path = child / 'result.json'
                row = json.loads(path.read_text()) if path.exists() else {'status': 'FAILED_PROCESS', 'quality_pass': False}
                if (child / 'watchdog.json').exists():
                    row.update(status='TIMEOUT', quality_pass=False)
                record['sources'].append(dict(index=index, source=graph['sources'][index] if source_count == 3 else None,
                    result=shared.binding(path) if path.exists() else None, returncode=code, status=row['status'],
                    admission=resources, quality_pass=row.get('quality_pass', False), calls=row.get('calls', {})))
                shared.save(args.output / 'result.json', record)
                if code or row['status'] != 'PASS':
                    raise RuntimeError('Native source failed; inspect its retained receipt/log')
            if epoch(args)[1] != digest:
                raise RuntimeError('Native source, binary or environment changed during the job')
            record.update(quality_pass=True, performance_eligible=args.phase == 'formal',
                status='PASS_QUALIFICATION' if args.phase in ('tiny', 'target') else 'PASS',
                T_first_seconds=sum(row['calls']['first']['seconds'] for row in record['sources']) / source_count,
                T_ready_seconds=sum(row['calls']['ready']['seconds'] for row in record['sources']) / source_count,
                source_aggregation='equal arithmetic mean of three fresh-source processes; one independent repeat block')
    except BaseException as error:
        record.update(status='FAILED', error=type(error).__name__+': '+str(error), quality_pass=False)
        raise
    finally:
        record['finished_unix'] = time.time()
        shared.save(args.output / 'result.json', record)
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--make-tiny-native', type=Path)
    p.add_argument('--execute-gpu', action='store_true')
    p.add_argument('--experiment', choices=('E1-MG', 'E2'), default='E2')
    p.add_argument('--algorithm', choices=shared.ALGORITHMS)
    p.add_argument('--dataset', choices=shared.DATASETS + shared.TINY_DATASETS)
    p.add_argument('--phase', choices=('tiny', 'diagnostic', 'target', 'formal'), default='diagnostic')
    p.add_argument('--count', type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument('--gpu-uuids')
    p.add_argument('--communication', choices=('host_staging', 'nccl'), default=None,
                   help='Frozen function rule: PR nccl; BFS/SSSP/WCC host_staging at every P')
    p.add_argument('--sssp-encoding', choices=('f64', 'u8_exact_u32_exact'), default='u8_exact_u32_exact',
                   help='Fixed static SSSP policy for all graphs and P; full FP64 input certification is timed')
    p.add_argument('--sssp-delta-rule', choices=('fixed16', 'certified_bound_plus_one'), default='certified_bound_plus_one')
    p.add_argument('--native-profile', choices=(native_profile.NAME,), default=native_profile.NAME)
    p.add_argument('--repeat', type=int, default=0)
    for name in ('input-root', 'release', 'native-dso', 'setup', 'lock-file', 'reference-cache', 'output', 'tiny-proof', 'target-proof'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--budget-mib', type=int, default=22000, help='GPU admission reserve parameter; not an arena allocation')
    p.add_argument('--retained-budget-mib', type=int, default=0, help='Native per-device retained-state cap; 0 leaves capacity to device')
    p.add_argument('--min-host-free-gib', type=float, default=8)
    p.add_argument('--keep-prepared', action='store_true', help='Retain all generated partition files; otherwise unlink after full audit')
    p.add_argument('--source-index', type=int, help=argparse.SUPPRESS)
    p.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    return p


def main():
    p = parser(); args = p.parse_args()
    if args.make_tiny_native:
        if args.execute_gpu:
            p.error('--make-tiny-native is CPU only')
        make_tiny(args.make_tiny_native)
        print(json.dumps({'status': 'PASS_NATIVE_TINY_INPUTS_CREATED', 'gpu_executed': False}))
        return
    required = ('algorithm', 'dataset', 'gpu_uuids', 'input_root', 'release', 'native_dso',
                'setup', 'lock_file', 'reference_cache', 'output')
    if not args.execute_gpu or any(getattr(args, key) is None for key in required):
        p.error('Explicit GPU execution and complete resource/input/source bindings are required')
    if args.budget_mib <= 0 or args.retained_budget_mib < 0 or args.min_host_free_gib <= 0:
        p.error('Invalid native resource budget')
    configure(args)
    record = source_child(args) if args.source_index is not None else job(args)
    print(json.dumps({key: record[key] for key in ('status', 'quality_pass')}))


if __name__ == '__main__':
    main()
