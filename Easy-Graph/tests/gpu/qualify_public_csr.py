"""Explicit GPU semantic qualification of the independent CSR candidate.

The experiment owner supplies the global GPU lock/resource gate. Importing this
file executes no GPU work. This is not a performance measurement.
"""
from pathlib import Path
import argparse
import hashlib
import json


def run(device):
    import numpy as np
    from easygraph.gpu import Session, StaleRequestError
    from qualify_capability_route import references
    nodes = np.array([-9, 4, 10, 11, 13, 99], dtype=np.int64)
    edges = np.array([[-9, 4], [4, 10], [10, -9], [10, 11], [13, 99]], dtype=np.int64)
    weights = np.array([1.25, 2.5, 4.75, 0., 3.125], dtype=np.float64)
    payload = dict(nodes=nodes.copy(), offsets=np.array([0, 1, 2, 4, 4, 5, 5], np.int32),
                   indices=np.array([1, 2, 0, 3, 5], np.int32), weights=weights.copy())
    expected = references(nodes, edges, weights)
    outputs, checks = [], []
    with Session.create({'g': payload}, devices=(device,), state_model='journal',
                        algorithms=tuple(expected), input_format='csr', budget_bytes=256 << 20,
                        pr_tolerance_l1=1e-11, pr_max_iter=2000) as session:
        assert session.dispatch_plan.input_kind == 'owned_csr'
        owned = session._backend.backend.session.journal._records['g'].base_edges
        assert owned._expanded_pairs is None
        profile = dict(session._backend.producer_profiles['g'])
        assert profile['dense_endpoint_temporary_bytes'] == 0
        assert profile['canonical_dense_endpoint_bytes_avoided'] == 16 * len(edges)
        # Caller mutations must not alter the registered snapshot or source map.
        payload['nodes'][:] = 777; payload['offsets'][:] = 0
        payload['indices'][:] = 0; payload['weights'][:] = 99
        def check_all(stage):
            expected_now = references(nodes, edges, weights)
            for algorithm in expected_now:
                result = session.query('g', algorithm, {'source': -9} if algorithm in ('bfs', 'sssp') else {})
                if algorithm in ('pr', 'weighted_pr'):
                    np.testing.assert_allclose(result.values, expected_now[algorithm], rtol=1e-9, atol=1e-9)
                else:
                    np.testing.assert_array_equal(result.values, expected_now[algorithm])
                assert result.values.flags.owndata and not result.values.flags.writeable
                assert not any(np.shares_memory(result.values, old.values) for old, _ in outputs)
                outputs.append((result, result.values.copy()))
                checks.append(dict(stage=stage, algorithm=algorithm, complete=True, independent=True))
        check_all('initial_after_caller_mutation')
        check_all('repeat')
        stale = session.request('g', 'sssp', {'source': -9})
        session.update('g', {'set': np.array([[-9., 4., 5.5]])})
        weights[0] = 5.5
        try:
            session.execute(stale)
        except StaleRequestError:
            pass
        else:
            raise AssertionError('CSR pre-update request did not become stale')
        check_all('weight_set')
        session.update('g', {'insert': np.array([[99., -9., 2.]])})
        edges = np.vstack([edges, [99, -9]]); weights = np.r_[weights, 2.]
        check_all('arc_insert')
        session.update('g', {'delete': np.array([[10, 11]], dtype=np.int64)})
        keep = ~((edges[:, 0] == 10) & (edges[:, 1] == 11))
        edges, weights = edges[keep], weights[keep]
        check_all('arc_delete')
    for output, saved in outputs:
        np.testing.assert_array_equal(output.values, saved)
    # Reciprocal CSR and a zero-arc graph retain all explicit isolated nodes.
    for zero_arcs in (False, True):
        labels = np.arange(4, dtype=np.int64)
        arcs = np.empty((0, 2), np.int64) if zero_arcs else np.array([[0, 1], [1, 0]], np.int64)
        row = np.zeros(5, np.int32) if zero_arcs else np.array([0, 1, 2, 2, 2], np.int32)
        col = np.empty(0, np.int32) if zero_arcs else np.array([1, 0], np.int32)
        expected_now = references(labels, arcs, np.ones(len(arcs)))
        with Session.create({'g': dict(nodes=labels, offsets=row, indices=col)}, devices=(device,),
                            input_format='csr', budget_bytes=256 << 20,
                            pr_tolerance_l1=1e-11, pr_max_iter=2000) as session:
            for algorithm in ('bfs', 'sssp', 'wcc', 'pr'):
                result = session.query('g', algorithm, {'source': 0} if algorithm in ('bfs', 'sssp') else {})
                np.testing.assert_allclose(result.values, expected_now[algorithm], rtol=1e-9, atol=1e-9)
                checks.append(dict(stage='zero_arc' if zero_arcs else 'reciprocal', algorithm=algorithm,
                                   complete=True, explicit_isolates=2 if len(arcs) else 4))
    source = Path(__file__).resolve().parents[2]
    bindings = {}
    for path in sorted((source/'easygraph').rglob('*.py')):
        bindings[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    bindings[str(Path(__file__).resolve().relative_to(source))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return dict(status='PASS_GPU_CSR_SEMANTICS', performance_eligible=False,
                gpu_device=device, checks=checks, initial_profile=profile,
                caller_arrays_independent=True, old_outputs_survive_updates_and_close=True,
                source_bindings=bindings, throughput_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-gpu', action='store_true', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh qualification output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'FAILED', 'performance_eligible': False}
    try:
        result = run(args.device)
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'performance_eligible': False}))


if __name__ == '__main__':
    main()
