"""CPU-only profile provenance, prerequisite and actual-observation guards."""
import copy
import json
import os
from types import SimpleNamespace
import pytest
import native_profile as profile
import native_worker as worker


@pytest.mark.parametrize('function', ['bfs', 'sssp', 'wcc', 'pr'])
@pytest.mark.parametrize('count', [1, 2, 4, 8])
def test_every_frozen_function_count_preserves_historical_optimizations(function, count):
    env = profile.environment(function, count)
    old = json.loads(profile.REFERENCE.read_text())['records'][f'{function}_p{count}']['environment']
    overrides = {'EGGPU_SSSP_WEIGHT_CERTIFICATION', 'EGGPU_SSSP_DELTA_MAX_DENSITY_PPM',
                 'EGGPU_SSSP_DELTA_CALIBRATION_SHA256', 'EGGPU_SINGLE_OWNER_SSSP'}
    assert all(env[key] == value for key, value in old.items() if key not in overrides)
    assert env['EGGPU_COMM_BACKEND'] == ('nccl' if function == 'pr' else 'host_staging')
    assert env['EGGPU_SINGLE_OWNER_SSSP'] == '0' and env['EGGPU_SSSP_WEIGHT_CERTIFICATION'] == 'full'
    for key in ('EGGPU_NATIVE_BFS_OWNED_OUTPUT', 'EGGPU_NATIVE_BFS_FUSED_OUTPUT',
                'EGGPU_NATIVE_BFS_WORKER_FP64_OUTPUT', 'EGGPU_WCC_CANONICAL_CERTIFICATE',
                'EGGPU_WCC_OUTPUT_FINALIZER', 'EGGPU_NATIVE_WCC_WORKER_FP64_OUTPUT',
                'EGGPU_NATIVE_SSSP_OWNED_OUTPUT', 'EGGPU_NATIVE_PR_OWNED_OUTPUT',
                'EGGPU_ALGORITHM_CONTROL_FUSION', 'EGGPU_SSSP_SKIP_EMPTY_HEAVY',
                'EGGPU_WCC_PINNED_ROUTE_REUSE', 'EGGPU_SSSP_DIRECT_FRONTIER'):
        assert env[key] == '1'
    if function in ('bfs', 'wcc'):
        cpus = [int(value) for group in env['EGMGPU_CORE_OUTPUT_CPUS'].split(';') for value in group.split(',')]
        assert cpus == list(profile.CPUS)
        assert len(env['EGMGPU_CORE_OUTPUT_CPUS'].split(';')) == count


@pytest.mark.parametrize('function', ['bfs', 'sssp', 'wcc', 'pr'])
@pytest.mark.parametrize('count', [1, 2, 4, 8])
def test_actual_historical_observations_satisfy_route_guards(function, count):
    row = json.loads(profile.REFERENCE.read_text())['records'][f'{function}_p{count}']
    measured = copy.deepcopy(row['observed_example'])
    # This field is new public API metadata; all route/ownership fields below
    # are original source-bound historical observations, not GPU execution here.
    if function == 'sssp':
        measured['static_sssp_delta_rule'] = 'certified_bound_plus_one'
    assert profile.observe(measured, function, count)['actual_native_route_checks_passed']
    measured['communication_backend'] = 'unqualified_transport'
    with pytest.raises(ValueError, match='transport'):
        profile.observe(measured, function, count)


def test_runner_profile_and_transport_propagate_and_foreign_flags_are_removed(monkeypatch):
    monkeypatch.setenv('EGGPU_UNREVIEWED_FLAG', '1')
    args = worker.parser().parse_args(['--algorithm','pr','--dataset','TINY_D','--phase','tiny',
                                      '--count','2','--gpu-uuids','GPU-a,GPU-b'])
    worker.configure(args)
    assert args.communication == 'nccl' and 'EGGPU_UNREVIEWED_FLAG' not in os.environ
    forwarded = worker.forward_options(args)
    for flag, value in [('--native-profile',profile.NAME), ('--sssp-delta-rule','certified_bound_plus_one'),
                        ('--communication','nccl')]:
        assert forwarded[forwarded.index(flag)+1] == value


def test_wrong_transport_and_delta_are_rejected_before_native_import():
    args = SimpleNamespace(native_profile=profile.NAME, algorithm='pr', count=8,
        communication='host_staging', sssp_encoding='u8_exact_u32_exact', sssp_delta_rule='certified_bound_plus_one')
    with pytest.raises(ValueError, match='requires PR NCCL'):
        profile.install(args)
    args.communication = 'nccl'; args.sssp_delta_rule = 'fixed16'
    with pytest.raises(ValueError, match='compact B\\+1'):
        profile.install(args)
