"""A non-P2P deployment qualifies only the paths it can actually execute."""
from types import SimpleNamespace
from copy import deepcopy

import pytest

from easygraph.egmgpu.partitioned_bindings import (
    NativeBindingEvidenceError, NativeBindingResultError,
    _requirements_for_group, _require_qualified_transport,
    default_partitioned_native_specs,
)


def topology(count, fully_peer):
    return {'peer_access': [[i == j or fully_peer for j in range(count)]
                            for i in range(count)]}


@pytest.mark.parametrize('count', [2, 4, 8])
def test_fully_connected_deployment_preserves_a100_requirements(count):
    for spec in default_partitioned_native_specs():
        assert _requirements_for_group(spec, count, topology(count, True)) == _requirements_for_group(spec, count)


@pytest.mark.parametrize('count', [2, 4, 8])
def test_non_peer_group_retains_case_coverage_on_real_transports(count):
    for spec in default_partitioned_native_specs():
        requirements = _requirements_for_group(spec, count, topology(count, False))
        assert tuple(x.name for x in requirements) == (('host_staging', 'nccl') if spec.name == 'pagerank' else ('host_staging',))
        host = next(x for x in spec.backend_requirements if x.name == 'host_staging')
        assert requirements[0].cases == (host.cases if count == 2 else (host.cases[0], host.cases[1], host.cases[3] if spec.name == 'pagerank' else host.cases[2]))


def test_incomplete_topology_is_not_evidence_of_no_peer_support():
    spec = default_partitioned_native_specs()[0]
    for identity in ({}, {'peer_access': [[True]]}, {'peer_access': [[1, 0], [0, 1]]}):
        with pytest.raises(NativeBindingEvidenceError):
            _requirements_for_group(spec, 2, identity)


@pytest.mark.parametrize('transport', ['peer_to_peer', 'nccl', '', 'auto'])
def test_host_only_qualification_rejects_unqualified_result_transport(transport):
    evidence = SimpleNamespace(runs=[SimpleNamespace(observed_backend='host_staging', extension_sha256='a'*64, correctness_passed=True)])
    _require_qualified_transport({'communication_backend': 'host_staging'}, evidence, 'a'*64)
    with pytest.raises(NativeBindingResultError):
        _require_qualified_transport({'communication_backend': transport}, evidence, 'a'*64)


def test_transport_from_other_binary_or_failed_probe_is_not_qualified():
    for digest, passed in [('b'*64, True), ('a'*64, False)]:
        evidence = SimpleNamespace(runs=[SimpleNamespace(observed_backend='host_staging', extension_sha256=digest, correctness_passed=passed)])
        with pytest.raises(NativeBindingResultError):
            _require_qualified_transport({'communication_backend': 'host_staging'}, evidence, 'a'*64)


@pytest.mark.parametrize('route', ['single', 'replicated', 'partitioned'])
def test_nominal_memory_reserve_preserves_capacity_class_without_admitting_other_card(route):
    from easygraph.egmgpu import single_bindings, native_bindings, partitioned_bindings
    native = {'device_id': 0, 'uuid': 'GPU-test', 'name': 'RTX 4090D',
              'pci_bus_id': '01:00.0', 'compute_capability': '8.9',
              'total_global_memory': 24540 << 20}
    measured = dict(native, total_global_memory=24564 << 20)
    group = {'devices': [native], 'peer_access': [[True]]}
    measurement = {'devices': [measured], 'peer_access': [[True]]}
    if route == 'partitioned':
        def validate(candidate):
            scope = partitioned_bindings.TopologyEvidence('not-used-by-this-check', (0,), candidate)
            partitioned_bindings._validate_group_within_deployment(scope, {0: measured}, {})
    else:
        module = single_bindings if route == 'single' else native_bindings
        def validate(candidate):
            module._validate_group_against_measurement(candidate, measurement)
    validate(group)
    wrong = deepcopy(group); wrong['devices'][0]['total_global_memory'] = 32700 << 20
    with pytest.raises(NativeBindingEvidenceError):
        validate(wrong)
