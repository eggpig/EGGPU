"""Explicit, source-bound restoration of the historical native execution profile.

No graph name, output value or measured duration selects this configuration.
The only graph-derived scheduling value is the public compact B+1 certificate.
"""
from pathlib import Path
import copy
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / 'historical_profile.json'
NAME = 'static_full_profile_v1'
CPUS = tuple(int(x) for x in os.environ.get('EGGPU_BENCH_CPU_SET', '0,1,2,3,4,5,6,7,48,49,50,51,52,53,54,55').split(','))
if len(CPUS) != 16 or len(set(CPUS)) != 16:
    raise ValueError('EGGPU_BENCH_CPU_SET must name exactly 16 distinct CPUs')


def environment(function, count):
    if function not in ('bfs', 'sssp', 'wcc', 'pr') or count not in (1, 2, 4, 8):
        raise ValueError('Unsupported fixed native function or GPU count')
    record = json.loads(REFERENCE.read_text())['records'][f'{function}_p{count}']
    env = copy.deepcopy(record['environment'])
    # Preserve the explicitly selected CPU map through fresh worker execs.
    env['EGGPU_BENCH_CPU_SET'] = ','.join(map(str, CPUS))
    if 'EGMGPU_CORE_OUTPUT_CPUS' in env:
        teams = [CPUS[i * (16 // count):(i + 1) * (16 // count)] for i in range(count)]
        env['EGMGPU_CORE_OUTPUT_CPUS'] = ';'.join(','.join(map(str, team)) for team in teams)
    # These are explicit protocol amendments, not an inheritance of historic
    # calibration or the separate eligible P1 single-owner engine.
    env.update(EGGPU_SSSP_WEIGHT_CERTIFICATION='full', EGGPU_SSSP_DELTA_MAX_DENSITY_PPM='0',
               EGGPU_SSSP_DELTA_CALIBRATION_SHA256='', EGGPU_SINGLE_OWNER_SSSP='0',
               EGGPU_DYNAMIC_WEIGHT_ROUTE='persistent_full',
               NCCL_IB_DISABLE='1', NCCL_P2P_DISABLE='1', NCCL_SOCKET_IFNAME='lo')
    return env


def install(args):
    if args.native_profile != NAME:
        raise ValueError('This runner accepts exactly its frozen full native profile')
    if args.sssp_encoding != 'u8_exact_u32_exact' or args.sssp_delta_rule != 'certified_bound_plus_one':
        raise ValueError('The full static profile requires explicit compact B+1 scheduling')
    expected = environment(args.algorithm, args.count)
    if args.communication != expected['EGGPU_COMM_BACKEND']:
        raise ValueError('The frozen profile requires PR NCCL and other functions host_staging for every P')
    for key in list(os.environ):
        if key.startswith(('EGGPU_', 'EGMGPU_', 'NCCL_')):
            os.environ.pop(key)
    os.environ.update(expected)
    return expected


def identity(args):
    return dict(name=NAME, function=args.algorithm, gpu_count=args.count,
                environment=environment(args.algorithm, args.count),
                reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                cpu_allocation=list(CPUS), sssp_delta_rule=args.sssp_delta_rule,
                exact_scope='Source-bound current execution; historical receipts motivate configuration only, not inherited GPU qualification.')


def require_fields(actual, expected, label):
    for key, value in expected.items():
        if actual.get(key) != value or (type(value) in (bool, int) and type(actual.get(key)) is not type(value)):
            raise ValueError('Actual native profile differs: ' + label + '.' + key)


def observe(metrics, function, count, *, previous=None):
    """Check actual C++ route/ownership evidence, never just environment names."""
    requested_communication = 'nccl' if function == 'pr' else 'host_staging'
    communication = requested_communication if count > 1 else 'none'
    require_fields(metrics, dict(communication_backend=communication), 'transport')
    if function in ('bfs', 'pr'):
        require_fields(metrics, dict(runtime_communication_backend=communication), 'runtime_transport')
    if function != 'pr':
        require_fields(metrics.get('algorithm_control_reduction', {}),
            dict(version='cooperative_control_completion_v1', requested=True, eligible=count > 1,
                 used=count > 1), 'control_fusion')
    if function == 'bfs':
        require_fields(metrics, dict(edge_schedule='warp', sparse_schedule='cta4' if count == 1 else 'cta1',
            dense_direction='push', output_mode='move', result_assembly='direct', host_assembly='workers',
            core_output_team_policy='team16', output_representation='owning_fp64_dense'), 'bfs')
        require_fields(metrics.get('bfs_worker_public_output', {}),
            dict(version='native_bfs_worker_public_fp64_v1', requested='1', eligible=True, used=True), 'bfs_output')
        require_fields(metrics.get('bfs_output_ownership', {}),
            dict(adopted_private_owned=True, route='adopt_private_worker_fp64_output'), 'bfs_public')
    elif function == 'sssp':
        require_fields(metrics, dict(weight_encoding='u8_exact', wire_encoding='u32_exact',
            host_assembly_policy='worker', host_output_team_policy='worker1',
            single_owner_execution_used=False, static_sssp_delta_rule='certified_bound_plus_one', bucket_rounds=1), 'sssp')
        require_fields(metrics.get('sssp_empty_heavy', {}), dict(requested=True, eligible=True, used=True,
            all_edges_light=True, heavy_rounds_executed=0, heavy_rounds_skipped=1), 'sssp_empty_heavy')
        require_fields(metrics.get('sssp_frontier_queue', {}), dict(version='sssp_light_epoch_queue_v1',
            requested=True, used=True), 'sssp_frontier')
        require_fields(metrics.get('sssp_output_ownership', {}), dict(requested='1', eligible=True, used=True), 'sssp_output')
        require_fields(metrics.get('sssp_public_output_ownership', {}),
            dict(route='adopt_private_numpy', bridge_copy_bytes=0), 'sssp_public')
    elif function == 'wcc':
        require_fields(metrics, dict(wcc_algorithm='forest', boundary_representation='coo',
            host_output_mode='pinned', result_merge_mode='canonical_direct',
            host_assembly_strategy='tiled' if count == 1 else 'worker', core_output_team_policy='team16',
            output_representation='owning_fp64_dense'), 'wcc')
        require_fields(metrics.get('wcc_worker_public_output', {}),
            dict(version='native_wcc_worker_public_fp64_v1', requested='1', eligible=True, used=True), 'wcc_output')
        require_fields(metrics.get('wcc_output_finalization', {}),
            dict(adopted_private_owned=True, route='adopt_private_worker_fp64_output'), 'wcc_public')
        pool = metrics.get('wcc_pinned_routes', {})
        require_fields(pool, dict(version='wcc_runtime_pinned_routes_v2', requested=True,
            eligible=count > 1, used=count > 1, query_completed_before_reuse=count > 1), 'wcc_pool')
        if previous is not None and count > 1:
            require_fields(pool, dict(cache_hit=True, new_allocation_count=0, capacity_growth_host_bytes=0), 'wcc_warm_pool')
    elif function == 'pr':
        require_fields(metrics, dict(preparation_method='transpose', host_weight_storage='implicit',
            device_metadata_policy='pull_only', execution_schedule='streamed', result_assembly='direct',
            requested_halo_exchange='nccl_sparse_halo', halo_exchange='nccl_sparse_halo' if count > 1 else 'local_skip',
            nccl_halo_active=count > 1, nccl_sparse_state_policy='retained',
            core_output_team_policy='original'), 'pr')
        require_fields(metrics.get('pr_output_ownership', {}), dict(requested='1', eligible=True, used=True), 'pr_output')
        require_fields(metrics.get('pr_public_output_ownership', {}),
            dict(route='adopt_private_numpy', bridge_copy_bytes=0), 'pr_public')
    else:
        raise ValueError('Unknown function')
    return dict(profile=NAME, function=function, gpu_count=count,
                requested_communication=requested_communication, observed_communication=communication,
                actual_native_route_checks_passed=True)
