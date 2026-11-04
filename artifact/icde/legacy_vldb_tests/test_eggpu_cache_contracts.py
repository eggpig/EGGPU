import json
import os
import pickle
import sys
from collections.abc import Mapping
from importlib import import_module
from types import SimpleNamespace

import easygraph as eg
import numpy as np
import pytest

from easygraph.utils import gpu_eggpu_backend as eggpu_backend


def test_direct_edge_attribute_mutation_invalidates_graph_state():
    graph = eg.Graph()
    graph.add_edge("u", "v", weight=1.0)
    graph.cache["__eggpu_prepared_graph__"] = {"ctx_id": 1}
    graph.cache["__eggpu_cpp_graph__"] = object()

    generation = graph._mutation_generation
    graph["u"]["v"]["weight"] = 7.0

    assert graph.cache == {}
    assert graph._mutation_generation == generation + 1
    assert graph["u"]["v"]["weight"] == 7.0


def test_set_edge_attributes_invalidates_graph_state():
    graph = eg.DiGraph()
    graph.add_edge(1, 2, weight=1.0)
    graph.cache["__eggpu_prepared_graph__"] = {"ctx_id": 1}

    eg.set_edge_attributes(graph, {(1, 2): 3.0}, name="weight")

    assert graph.cache == {}
    assert graph[1][2]["weight"] == 3.0


def test_graph_context_binds_monotonic_mutation_generation():
    graph = eg.Graph()
    graph.add_edge(0, 1, weight=1.0)
    first = eggpu_backend._graph_context(graph)

    graph[0][1]["weight"] = 2.0
    second = eggpu_backend._graph_context(graph)

    assert second is not first
    assert second["ctx_id"] != first["ctx_id"]
    assert second["mutation_generation"] == graph._mutation_generation


def test_multigraph_edge_attribute_mutation_invalidates_graph_state():
    graph = eg.MultiGraph()
    key = graph.add_edge(1, 2, weight=1.0)
    graph.cache["__eggpu_prepared_graph__"] = {"ctx_id": 1}

    graph[1][2][key].update(weight=5.0)

    assert graph.cache == {}
    assert graph[1][2][key]["weight"] == 5.0


def test_remove_nodes_from_invalidates_graph_state():
    graph = eg.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    graph.cache["__eggpu_prepared_graph__"] = {"ctx_id": 1}

    graph.remove_nodes_from([1])

    assert graph.cache == {}
    assert 1 not in graph


def test_dense_result_is_an_honest_read_only_mapping():
    values = eggpu_backend._DenseValueDict(range(3), [0.25, 0.5, 0.25])
    expected = {0: 0.25, 1: 0.5, 2: 0.25}

    assert isinstance(values, Mapping)
    assert not isinstance(values, dict)
    assert values == expected
    assert dict(values) == expected
    assert pickle.loads(pickle.dumps(values)) == expected
    assert values.to_numpy(copy=False).flags.writeable is False

    try:
        values[0] = 1.0
    except TypeError:
        pass
    else:
        raise AssertionError("dense result mapping unexpectedly accepted mutation")

    # The standard JSON encoder does not support arbitrary Mapping objects.
    # Raising is preferable to the old dict-subclass behavior, which silently
    # emitted an empty object despite nonempty logical contents.
    try:
        encoded = json.dumps(values)
    except TypeError:
        pass
    else:
        assert json.loads(encoded) == {"0": 0.25, "1": 0.5, "2": 0.25}


def test_dense_distance_and_multisource_mapping_equality():
    row = eggpu_backend._DenseDistanceDict(range(3), [0.0, 2.0, np.inf])
    rows = eggpu_backend._DenseMultiSourceSSSPDict(
        [0],
        range(3),
        [[0.0, 2.0, np.inf]],
    )

    assert row == {0: 0.0, 1: 2.0}
    assert dict(row) == {0: 0.0, 1: 2.0}
    assert rows == {0: {0: 0.0, 1: 2.0}}
    assert dict(rows[0]) == {0: 0.0, 1: 2.0}


def test_dense_shape_contract_accepts_only_exact_or_explicit_sentinel():
    exact = eggpu_backend._normalize_dense_array(
        [1.0, 2.0, 3.0],
        3,
        result_name="test vector",
    )
    sentinel = eggpu_backend._normalize_dense_array(
        [-1.0, 1.0, 2.0, 3.0],
        3,
        allow_leading_sentinel=True,
        result_name="test vector",
    )
    np.testing.assert_array_equal(exact, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(sentinel, [1.0, 2.0, 3.0])

    with pytest.raises(eggpu_backend._DenseResultShapeError, match="expected \\(3,\\)"):
        eggpu_backend._normalize_dense_array(
            [-1.0, 1.0, 2.0, 3.0],
            3,
            result_name="test vector",
        )
    with pytest.raises(
        eggpu_backend._DenseResultShapeError,
        match="dense node-value mapping.*expected \\(3,\\)",
    ):
        eggpu_backend._DenseValueDict(range(3), [1.0, 2.0])
    with pytest.raises(
        eggpu_backend._DenseResultShapeError,
        match="dense distance mapping.*expected \\(3,\\)",
    ):
        eggpu_backend._DenseDistanceDict(range(3), [0.0, 1.0])

    rows = eggpu_backend._DenseMultiSourceSSSPDict(
        [0, 1],
        range(3),
        [
            [-1.0, 0.0, 1.0, 2.0],
            [-1.0, 2.0, 1.0, 0.0],
        ],
        allow_leading_sentinel=True,
        result_name="test SSSP matrix",
    )
    np.testing.assert_array_equal(
        rows.to_numpy(),
        [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
    )

    with pytest.raises(
        eggpu_backend._DenseResultShapeError,
        match="expected \\(2, 3\\) or \\(2, 4\\)",
    ):
        eggpu_backend._DenseMultiSourceSSSPDict(
            [0, 1],
            range(3),
            [[0.0, 1.0, 2.0]],
            allow_leading_sentinel=True,
            result_name="test SSSP matrix",
        )
    with pytest.raises(
        eggpu_backend._DenseResultShapeError,
        match="expected \\(2, 3\\) or \\(2, 4\\)",
    ):
        eggpu_backend._DenseMultiSourceSSSPDict(
            [0, 1],
            range(3),
            [[0.0, 1.0], [1.0, 0.0]],
            allow_leading_sentinel=True,
            result_name="test SSSP matrix",
        )


def _install_faulty_cpp_dense_backend(monkeypatch, **functions):
    monkeypatch.setattr(
        eggpu_backend,
        "_get_cached_cpp_graph",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "cpp_easygraph", SimpleNamespace(**functions))


def test_pagerank_wrong_dense_size_fails_closed(monkeypatch):
    _install_faulty_cpp_dense_backend(
        monkeypatch,
        cpp_gpu_pagerank_dense=lambda _graph, **_kwargs: {
            "values_dense": np.asarray([0.5, 0.5]),
            "kernel_seconds": 0.001,
        },
    )
    prepared = {"nodes": [0, 1, 2], "node_to_idx": {0: 0, 1: 1, 2: 2}}

    with pytest.raises(
        eggpu_backend._DenseResultShapeError,
        match="PageRank values_dense.*expected \\(3,\\) or \\(4,\\)",
    ):
        eggpu_backend._cpp_gpu_pagerank(
            prepared,
            object(),
            alpha=0.85,
            max_iter=100,
            eps=1e-6,
            weight=None,
        )


def test_lcc_wrong_dense_size_fails_closed(monkeypatch):
    _install_faulty_cpp_dense_backend(
        monkeypatch,
        cpp_gpu_clustering_dense=lambda _graph: {
            "values_dense": np.asarray([1.0, 0.0]),
            "kernel_seconds": 0.001,
        },
    )
    prepared = {"nodes": [0, 1, 2], "node_to_idx": {0: 0, 1: 1, 2: 2}}

    with pytest.raises(
        eggpu_backend._DenseResultShapeError,
        match="LCC values_dense.*expected \\(3,\\) or \\(4,\\)",
    ):
        eggpu_backend._cpp_gpu_clustering(prepared, object())


def test_connected_components_wrong_dense_size_fails_closed(monkeypatch):
    _install_faulty_cpp_dense_backend(
        monkeypatch,
        cpp_gpu_connected_component_labels_dense=lambda _graph, directed: {
            "labels_dense": np.asarray([0, 0]),
            "kernel_seconds": 0.001,
        },
    )
    prepared = {"nodes": [0, 1, 2], "node_to_idx": {0: 0, 1: 1, 2: 2}}

    with pytest.raises(
        eggpu_backend._DenseResultShapeError,
        match="connected-component labels_dense.*expected \\(3,\\) or \\(4,\\)",
    ):
        eggpu_backend._cpp_gpu_connected_component_labels_dense(
            prepared,
            object(),
            directed=False,
        )


def test_sssp_wrong_dense_matrix_shape_fails_closed(monkeypatch):
    _install_faulty_cpp_dense_backend(
        monkeypatch,
        cpp_gpu_dijkstra_multisource_dense=lambda _graph, _sources, **_kwargs: {
            "values_dense": np.zeros((1, 3), dtype=np.float64),
            "kernel_seconds": 0.001,
        },
    )
    prepared = {"nodes": [0, 1, 2], "node_to_idx": {0: 0, 1: 1, 2: 2}}

    with pytest.raises(
        eggpu_backend._DenseResultShapeError,
        match="Dijkstra multi-source values_dense.*expected \\(2, 3\\) or \\(2, 4\\)",
    ):
        eggpu_backend._cpp_gpu_multi_source_dijkstra(
            prepared,
            object(),
            sources=[0, 1],
            weight="weight",
            target=None,
        )


def test_opt_in_result_cache_owns_independent_snapshots():
    old_enabled = eggpu_backend._RESULT_CACHE_ENABLED
    old_copy = eggpu_backend._RESULT_CACHE_RETURN_COPY
    eggpu_backend._RESULT_CACHE_ENABLED = True
    eggpu_backend._RESULT_CACHE_RETURN_COPY = True
    try:
        prepared = {"nodes": [0], "result_cache": {}, "result_cache_order": []}
        original = {0: [1]}
        key = ("test",)
        eggpu_backend._result_cache_put(prepared, key, original, 0.25)
        original[0].append(2)

        first, kernel = eggpu_backend._result_cache_get(prepared, key)
        first[0].append(3)
        second, _ = eggpu_backend._result_cache_get(prepared, key)

        assert kernel == 0.25
        assert first == {0: [1, 3]}
        assert second == {0: [1]}
    finally:
        eggpu_backend._RESULT_CACHE_ENABLED = old_enabled
        eggpu_backend._RESULT_CACHE_RETURN_COPY = old_copy


def test_lcc_gpu_dispatch_rejects_unimplemented_semantics():
    cluster = import_module("easygraph.functions.basic.cluster")
    old_value = os.environ.get("EASYGRAPH_ENABLE_GPU")
    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
    try:
        directed = eg.DiGraph()
        directed.add_edges_from([(0, 1), (1, 2), (2, 0)])
        assert cluster._clustering_gpu_runtime_dispatch(directed) is None

        weighted = eg.Graph()
        weighted.add_edge(0, 1, weight=2.0)
        assert (
            cluster._clustering_gpu_runtime_dispatch(
                weighted,
                weight="weight",
            )
            is None
        )
    finally:
        if old_value is None:
            os.environ.pop("EASYGRAPH_ENABLE_GPU", None)
        else:
            os.environ["EASYGRAPH_ENABLE_GPU"] = old_value


def test_lcc_gpu_dispatch_accepts_directed_bulk_projection(monkeypatch):
    cluster = import_module("easygraph.functions.basic.cluster")

    class DirectedBulkProjection:
        _eggpu_bulk_csr = True
        undirected_projection = {"format": "logical-undirected"}

        def is_directed(self):
            return True

        def __contains__(self, node):
            return isinstance(node, int) and node in {0, 1, 2}

    graph = DirectedBulkProjection()
    expected = {0: 1.0, 1: 0.5, 2: 0.0}
    monkeypatch.setenv("EASYGRAPH_ENABLE_GPU", "TRUE")
    monkeypatch.setattr(eggpu_backend, "eggpu_backend_enabled", lambda: True)
    monkeypatch.setattr(eggpu_backend, "clustering", lambda current: expected)

    assert cluster._clustering_gpu_runtime_dispatch(graph) == expected
    assert cluster._clustering_gpu_runtime_dispatch(graph, nodes=[1, 2]) == {
        1: 0.5,
        2: 0.0,
    }
