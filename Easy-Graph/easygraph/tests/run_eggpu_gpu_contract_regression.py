"""Focused CUDA regressions for EGGPU's state and result contracts.

Run in an isolated process with two visible GPUs, for example:

CUDA_VISIBLE_DEVICES=1,2 EASYGRAPH_ENABLE_GPU=TRUE \
  EASYGRAPH_GPU_STRICT_ERRORS=TRUE EASYGRAPH_GPU_RESULT_CACHE=FALSE \
  python easygraph/tests/run_eggpu_gpu_contract_regression.py
"""

import contextlib
import ctypes
import json
import math
import os

import easygraph as eg

from easygraph.functions.centrality.pagerank import _pagerank_power_iteration
from easygraph.utils import gpu_eggpu_backend as eggpu_backend


def assert_mapping_close(actual, expected, atol=5.0e-5):
    assert set(actual) == set(expected)
    for node, expected_value in expected.items():
        actual_value = actual[node]
        assert math.isclose(
            float(actual_value),
            float(expected_value),
            rel_tol=atol,
            abs_tol=atol,
        ), (node, actual_value, expected_value)


@contextlib.contextmanager
def gpu_runtime(enabled):
    previous = os.environ.get("EASYGRAPH_ENABLE_GPU")
    if enabled:
        os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
    else:
        os.environ.pop("EASYGRAPH_ENABLE_GPU", None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("EASYGRAPH_ENABLE_GPU", None)
        else:
            os.environ["EASYGRAPH_ENABLE_GPU"] = previous


def cpu_call(function, *args, **kwargs):
    with gpu_runtime(False):
        return function(*args, **kwargs)


def edge_weight_sum(tree, weight="weight"):
    return sum(float(data.get(weight, 1.0)) for _, _, data in tree.edges)


def load_cudart():
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("CUDA runtime library not found")


def set_device(device_id):
    cudart = load_cudart()
    rc = int(cudart.cudaSetDevice(ctypes.c_int(device_id)))
    if rc != 0:
        raise RuntimeError(f"cudaSetDevice({device_id}) failed with code {rc}")


def expect_runtime_error(function):
    try:
        function()
    except RuntimeError:
        return
    raise AssertionError("expected fixed-device affinity RuntimeError")


def test_weight_mutation_rebuilds_pagerank_and_dijkstra():
    graph = eg.DiGraph()
    graph.add_edge(0, 1, weight=8.0)
    graph.add_edge(0, 2, weight=1.0)
    graph.add_edge(1, 2, weight=2.0)
    graph.add_edge(2, 0, weight=1.0)
    first_generation = graph._mutation_generation

    first_pr = eg.pagerank(graph, weight="weight")
    first_dist = eg.single_source_dijkstra(graph, 0, weight="weight")

    graph[0][1]["weight"] = 0.25
    assert graph._mutation_generation == first_generation + 1
    second_pr = eg.pagerank(graph, weight="weight")
    second_dist = eg.single_source_dijkstra(graph, 0, weight="weight")

    fresh = eg.DiGraph()
    fresh.add_edge(0, 1, weight=0.25)
    fresh.add_edge(0, 2, weight=1.0)
    fresh.add_edge(1, 2, weight=2.0)
    fresh.add_edge(2, 0, weight=1.0)
    reference_pr = _pagerank_power_iteration(fresh, weight="weight")
    reference_dist = cpu_call(
        eg.single_source_dijkstra,
        fresh,
        0,
        weight="weight",
    )

    assert_mapping_close(second_pr, reference_pr)
    assert_mapping_close(second_dist, reference_dist)
    assert any(
        not math.isclose(float(first_pr[node]), float(second_pr[node]), abs_tol=1.0e-4)
        for node in second_pr
    )
    assert first_dist != second_dist


def test_weighted_and_directed_lcc_use_semantic_fallback():
    weighted = eg.Graph()
    weighted.add_edge(0, 1, weight=1.0)
    weighted.add_edge(1, 2, weight=2.0)
    weighted.add_edge(2, 0, weight=4.0)
    weighted.add_edge(2, 3, weight=8.0)
    gpu_enabled = eg.clustering(weighted, weight="weight")
    reference = cpu_call(eg.clustering, weighted, weight="weight")
    assert_mapping_close(gpu_enabled, reference)

    directed = eg.DiGraph()
    directed.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 1)])
    gpu_enabled = eg.clustering(directed)
    reference = cpu_call(eg.clustering, directed)
    assert_mapping_close(gpu_enabled, reference)


def test_specialized_layouts_are_isolated_by_graph_identity():
    rank_a = eg.DiGraph()
    rank_a.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
    rank_b = eg.DiGraph()
    rank_b.add_edges_from([(0, 1), (1, 0), (1, 2), (3, 2)])
    for graph in (rank_a, rank_b):
        actual = eg.pagerank(graph)
        expected = _pagerank_power_iteration(graph)
        assert_mapping_close(actual, expected)

    lcc_a = eg.Graph()
    lcc_a.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
    lcc_b = eg.Graph()
    lcc_b.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    for graph in (lcc_a, lcc_b):
        actual = eg.clustering(graph)
        expected = cpu_call(eg.clustering, graph)
        assert_mapping_close(actual, expected)

    mst_a = eg.Graph()
    mst_a.add_weighted_edges_from(
        [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0), (0, 3, 9.0)]
    )
    mst_b = eg.Graph()
    mst_b.add_weighted_edges_from(
        [(0, 1, 9.0), (1, 2, 2.0), (2, 3, 1.0), (0, 3, 3.0)]
    )
    for graph in (mst_a, mst_b):
        actual = eg.minimum_spanning_tree(graph, weight="weight")
        expected = cpu_call(eg.minimum_spanning_tree, graph, weight="weight")
        assert math.isclose(edge_weight_sum(actual), edge_weight_sum(expected))


def test_specialized_workspaces_enforce_fixed_device_affinity():
    graph = eg.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 1.0), (1, 2, 2.0), (2, 0, 3.0), (2, 3, 4.0)]
    )
    directed = graph.to_directed()

    set_device(0)
    eg.pagerank(directed, weight="weight")
    set_device(1)
    expect_runtime_error(lambda: eg.pagerank(directed, weight="weight"))
    set_device(0)

    eg.clustering(graph)
    set_device(1)
    expect_runtime_error(lambda: eg.clustering(graph))
    set_device(0)

    eg.minimum_spanning_tree(graph, weight="weight")
    set_device(1)
    expect_runtime_error(
        lambda: eg.minimum_spanning_tree(graph, weight="weight")
    )
    set_device(0)


def main():
    assert eggpu_backend._RESULT_CACHE_ENABLED is False
    tests = [
        test_weight_mutation_rebuilds_pagerank_and_dijkstra,
        test_weighted_and_directed_lcc_use_semantic_fallback,
        test_specialized_layouts_are_isolated_by_graph_identity,
        test_specialized_workspaces_enforce_fixed_device_affinity,
    ]
    passed = []
    for test in tests:
        test()
        passed.append(test.__name__)
        print(f"PASS {test.__name__}", flush=True)
    print(json.dumps({"status": "PASS", "tests": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
