"""Existing GUM algorithm/build/runtime adapters, mechanically extracted.
No author algorithm, work-stealing flag or native query timer is changed.
"""
from pathlib import Path
import json,os
import sg_worker as shared
HERE=Path(__file__).resolve().parent
TARGETS=dict(bfs='gum_bfs_wcc',wcc='gum_bfs_wcc',pr='gum_pr_fp64',sssp='gum_sssp_standard')
U32_MAX=(1<<32)-1
def read(path):return json.loads(Path(path).read_text())
def binding(path):return shared.binding(path)
def check_binding(item):
 actual=shared.binding(item['path'])
 if actual['sha256']!=item['sha256'] or actual['bytes']!=item.get('bytes',item.get('byte_size')):raise ValueError('Changed GUM source/build binding')


def weight_domain(graph):
    import numpy as np
    maximum = 0
    for start in range(0, len(graph['weights']), 1 << 20):
        block = graph['weights'][start:start + (1 << 20)]
        if (not np.isfinite(block).all() or np.any(block <= 0) or np.any(block >= U32_MAX)
                or np.any(block != np.floor(block))):
            raise ValueError('GUM author uint32 SSSP requires positive exact integral canonical weights')
        if len(block): maximum = max(maximum, int(block.max()))
    bound = (graph['n'] - 1) * maximum
    if bound >= U32_MAX: raise ValueError('GUM uint32 shortest-path bound reaches unreachable sentinel')
    return dict(canonical_storage='float64', native_storage='uint32', all_weights_exactly_integral=True,
                max_weight=maximum, simple_path_distance_bound=bound, unreachable_sentinel=U32_MAX,
                conversion='Original canonical values converted exactly by native loader; no regenerated weights')

def normalise_native(path, function, n):
    import numpy as np
    path = Path(path)
    if path.stat().st_size != n * 8: raise ValueError('Native output does not cover the complete vertex domain')
    values = np.fromfile(path, dtype='<f8' if function == 'pr' else '<i8')
    if function in ('bfs', 'sssp'):
        missing = (values == U32_MAX) | (values == -1)
        if np.any((values < 0) & ~missing) or np.any(values > U32_MAX):
            raise ValueError('Invalid native uint32 distance')
        values = values.astype(np.float64); values[missing] = np.inf
    # Raw WCC labels are deliberately preserved. Canonicalization is audit-only.
    return np.array(values, copy=True, order='C')

def artifact_identity(build_path, function):
    build = read(build_path)
    if build['status'] != 'BUILT_GPU_UNTESTED': raise ValueError('GUM CSR build is incomplete')
    check_binding(build['plan']); plan = read(build['plan']['path'])
    if plan['schema'] != 'gum_law_csr_cpu_build_v1': raise ValueError('Use isolated CSR-loader build')
    check_binding(plan['source_manifest']); manifest = read(plan['source_manifest']['path'])
    if build['source_manifest_sha256'] != plan['source_manifest']['sha256']:
        raise ValueError('Compiled source binding changed')
    if not plan['preservation']['algorithm_kernels_and_output_adapters_unchanged']:
        raise ValueError('CSR loader must preserve author algorithms and complete-output adapters')
    source = Path(plan['source'])
    for name, item in manifest['files'].items():
        path = (source / name).resolve(); path.relative_to(source.resolve())
        check_binding(dict(path=str(path), bytes=item['bytes'], sha256=item['sha256']))
    selected = build['binaries'][TARGETS[function]]; check_binding(selected)
    runtime = source / 'runtime.py'
    if 'runtime.py' not in manifest['files']: raise ValueError('Native timing parser is not source bound')
    return dict(build=binding(build_path), plan=build['plan'], source_manifest=plan['source_manifest'],
                binary=selected, runtime=binding(runtime), source=str(source))

def native_command(args, binary, descriptor, n, source, count, native):
    application = dict(bfs='bfs', pr='pagerank', sssp='sssp', wcc='wcc_opt')[args.function]
    argv = [str(args.mpiexec), '--bind-to', 'none', '--mca', 'btl', 'self,vader,tcp', '--tag-output', '-n', str(count),
            str(args.core_python), '-B', str(HERE / 'gum_rank_exec.py'), str(binary),
            '--application=' + application, '--efile=' + str(descriptor), '--vfile=' + str(descriptor),
            '--out_prefix=' + str(native), '--directed=true', '--rebalance=false', '--ws=false', '--partitioner=seg',
            '--lb=cm', '--bfs_source=' + str(source), '--stage2_vertices=' + str(n), '--stage2_rounds=3',
            '--logtostderr=true', '--serialization_prefix=']
    if args.function == 'sssp': argv += ['--sssp_source=' + str(source)]
    if args.function == 'pr': argv += ['--pr_d=0.85', '--pr_mr=117']
    return argv

def native_proofs(folder, record):
    count = len(record['physical_devices']); cpus = set(record['cpu_affinity'])
    launch = []; cpu = []
    for rank in range(count):
        path = folder / f'launch-rank-{rank}.json'; row = read(path)
        if row['rank'] != rank or row['size'] != count or set(row['cpu_affinity']) != cpus:
            raise ValueError('Native rank launch/CPU placement differs')
        if row['binary'] != record['artifact']['binary'] or row['cuda_visible_devices'] != ','.join(record['gpu_uuids']):
            raise ValueError('Launched GUM binary/GPU mask differs')
        if row['pid'] <= 0: raise ValueError('Missing native rank PID')
        launch.append(dict(binding=binding(path), evidence=row))
        if record['function'] == 'sssp':
            path = folder / f'cpu-rank-{rank}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if [(x['round'], x['bind']) for x in rows] != [(r, b) for r in range(3) for b in (True, False)]:
                raise ValueError('Incomplete standard SSSP thread CPU observations')
            for row in rows:
                if row['rank'] != rank or row['pid'] != launch[-1]['evidence']['pid'] or not row['threads']:
                    raise ValueError('Invalid native thread CPU record')
                if any(not t['cpus'] or not set(t['cpus']) <= cpus for t in row['threads']):
                    raise ValueError('Native thread exceeded CPU budget')
            cpu.append(dict(binding=binding(path), observations=len(rows)))
    return dict(rank_launch=launch, standard_sssp_thread_observations=cpu)

def execution_environment(args, uuids, cpus, native, scratch):
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=','.join(uuids),
        OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        EGGPU_GUM_CPUSET=','.join(map(str, cpus)), EGMGPU_BASELINE_OUTPUT_NUMA='gpu_local',
        HWLOC_COMPONENTS='-gl', OMPI_ALLOW_RUN_AS_ROOT='1', OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1',
        NCCL_DEBUG='WARN', PMIX_MCA_gds='hash', TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
        OMPI_MCA_orte_tmpdir_base=str(scratch), GUM_RANK_EVIDENCE=str(native), PYTHONDONTWRITEBYTECODE='1',
        LD_PRELOAD=args.ld_preload, LD_LIBRARY_PATH=args.ld_library_path)
    controls = {k: v for k, v in sorted(env.items()) if k.startswith(('EGGPU_', 'EGMGPU_', 'OMP_', 'MKL_', 'OPENBLAS_', 'NCCL_', 'CUDA_'))
                or k in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'HWLOC_COMPONENTS', 'PMIX_MCA_gds')}
    controls.update(mpiexec=binding(args.mpiexec), python_executable=str(args.core_python), pr_iterations=117,
        partitioner='seg', load_balance='cm', work_stealing=False, wcc_application='wcc_opt', rounds=3)
    return env, controls
