"""Explicit GPU qualification; importing this file never starts CUDA.

Run only through the experiment owner's resource-controlled job launcher:
  python tests/gpu/qualify_capability_route.py --execute-gpu --output receipt.json
This tiny semantic check is not a performance measurement or a paper result.
"""
import argparse
import heapq
import json
from pathlib import Path
from time import perf_counter


def references(nodes, edges, weights):
    import numpy as np
    n = len(nodes)
    labels = {int(node): i for i, node in enumerate(nodes)}
    adjacency = [[] for _ in nodes]
    matrix = np.zeros((n, n), dtype=np.float64)
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for (u, v), weight in zip(edges, weights):
        a, b = labels[int(u)], labels[int(v)]
        adjacency[a].append((b, float(weight)))
        matrix[a, b] = float(weight)
        x, y = root(a), root(b)
        parent[max(x, y)] = min(x, y)
    def distance(weighted):
        d = np.full(n, np.inf)
        d[0] = 0
        queue = [(0., 0)]
        while queue:
            value, u = heapq.heappop(queue)
            if value != d[u]:
                continue
            for v, w in adjacency[u]:
                candidate = value + (w if weighted else 1.)
                if candidate < d[v]:
                    d[v] = candidate
                    heapq.heappush(queue, (candidate, v))
        return d
    ranks = {}
    for name, graph in [('pr', (matrix != 0).astype(float)), ('weighted_pr', matrix.copy())]:
        # Zero-weight edges remain topology edges for the unweighted projection.
        if name == 'pr':
            graph.fill(0.)
            for u, v in edges:
                graph[labels[int(u)], labels[int(v)]] = 1.
        sums = graph.sum(axis=1)
        transition = np.divide(graph, sums[:, None], out=np.zeros_like(graph), where=sums[:, None] != 0)
        transition[sums == 0] = 1. / n
        ranks[name] = np.linalg.solve(np.eye(n) - .85 * transition.T, np.full(n, .15 / n))
    return dict(bfs=distance(False), sssp=distance(True),
                wcc=np.array([root(i) for i in range(n)], dtype=float), **ranks)


def run(device):
    import numpy as np
    from easygraph.gpu import Session, StaleRequestError
    nodes = np.array([-9, 4, 10, 11, 13, 99], dtype=np.int64)
    edges = np.array([[-9, 4], [4, 10], [10, -9], [10, 11], [13, 99]], dtype=np.int64)
    weights = np.array([1.25, 2.5, 4.75, 0., 3.125], dtype=np.float64)
    expected = references(nodes, edges, weights)
    outputs, checks = [], []
    started = perf_counter()
    with Session.create({'g': dict(nodes=nodes, edges=edges, weights=weights)},
                        devices=(device,), state_model='journal',
                        algorithms=('bfs', 'sssp', 'wcc', 'pr', 'weighted_pr'),
                        budget_bytes=256 << 20, pr_tolerance_l1=1e-11, pr_max_iter=2000) as session:
        plan = session.dispatch_plan.to_dict()
        assert plan['backend'] == 'arena'
        assert set(session.capabilities().algorithms) == set(expected)
        for repetition in range(2):
            for algorithm in expected:
                result = session.query('g', algorithm, {'source': -9} if algorithm in ('bfs', 'sssp') else {})
                if algorithm in ('pr', 'weighted_pr'):
                    np.testing.assert_allclose(result.values, expected[algorithm], atol=1e-9, rtol=1e-9)
                else:
                    np.testing.assert_array_equal(result.values, expected[algorithm])
                assert result.values.flags.owndata and not result.values.flags.writeable
                assert not any(np.shares_memory(result.values, old.values) for old, _ in outputs)
                outputs.append((result, result.values.copy()))
                checks.append(dict(algorithm=algorithm, repetition=repetition, complete=True,
                                   immutable=True, independent=True))
        stale = session.request('g', 'sssp', {'source': -9})
        session.update('g', {'set': np.array([[-9., 4., 5.5]])})
        try:
            session.execute(stale)
        except StaleRequestError:
            pass
        else:
            raise AssertionError('Pre-update request executed against a different version')
        changed = weights.copy()
        changed[0] = 5.5
        result = session.query('g', 'sssp', {'source': -9})
        np.testing.assert_array_equal(result.values, references(nodes, edges, changed)['sssp'])
        outputs.append((result, result.values.copy()))
    for result, original in outputs:
        np.testing.assert_array_equal(result.values, original)
    return dict(status='PASS_GPU_SEMANTICS', performance_eligible=False,
                plan=plan, checks=checks, update_and_stale_request=True,
                all_outputs_survive_close=True, full_outputs=len(outputs),
                qualification_seconds=perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-gpu', action='store_true', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = {'status': 'FAIL', 'performance_eligible': False}
    try:
        receipt = run(args.device)
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
