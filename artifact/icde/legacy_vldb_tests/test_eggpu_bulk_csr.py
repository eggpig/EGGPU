import json
import importlib
from array import array

import numpy as np
import pytest

import easygraph as eg
from easygraph.utils import gpu_eggpu_backend


EDGES = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]


def _write_bulk_graph(tmp_path, directed=False, weighted=False):
    rows = [[] for _ in range(6)]
    for u, v in EDGES:
        rows[u].append(v)
        if not directed:
            rows[v].append(u)
    offsets = [0]
    indices = []
    for row in rows:
        indices.extend(row)
        offsets.append(len(indices))

    offsets_path = tmp_path / "tiny.offsets.i32"
    indices_path = tmp_path / "tiny.indices.i32"
    array("i", offsets).tofile(offsets_path.open("wb"))
    array("i", indices).tofile(indices_path.open("wb"))
    metadata = {
        "format": "eggpu-csr-v1",
        "generation": 1,
        "name": "tiny",
        "directed": directed,
        "num_nodes": 6,
        "num_edges": len(EDGES),
        "num_entries": len(indices),
        "node_labels": "zero_based_contiguous",
        "offset_dtype": "int32",
        "index_dtype": "int32",
        "offsets_path": offsets_path.name,
        "indices_path": indices_path.name,
    }
    if weighted:
        weights_path = tmp_path / "tiny.weights.f64"
        weights = []
        for source, row in enumerate(rows):
            weights.extend(1.0 + ((source * target) % 6) for target in row)
        array("d", weights).tofile(weights_path.open("wb"))
        metadata.update(
            {
                "weights_path": weights_path.name,
                "weight_dtype": "float64",
                "weight_key": "weight",
                "weight_semantics": "1 + (src * dst) % num_nodes",
            }
        )
    metadata_path = tmp_path / "tiny.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return eg.read_eggpu_csr(metadata_path)


def _regular_graph():
    graph = eg.Graph()
    graph.add_nodes(range(6))
    graph.add_edges(EDGES)
    return graph


def _write_cycle_bulk_graph(tmp_path, num_nodes):
    offsets = [2 * node for node in range(num_nodes + 1)]
    indices = []
    for node in range(num_nodes):
        indices.extend(((node - 1) % num_nodes, (node + 1) % num_nodes))

    offsets_path = tmp_path / "cycle.offsets.i32"
    indices_path = tmp_path / "cycle.indices.i32"
    array("i", offsets).tofile(offsets_path.open("wb"))
    array("i", indices).tofile(indices_path.open("wb"))
    metadata_path = tmp_path / "cycle.json"
    metadata_path.write_text(
        json.dumps(
            {
                "format": "eggpu-csr-v1",
                "generation": 1,
                "name": "cycle",
                "directed": False,
                "num_nodes": num_nodes,
                "num_edges": num_nodes,
                "num_entries": len(indices),
                "node_labels": "zero_based_contiguous",
                "offset_dtype": "int32",
                "index_dtype": "int32",
                "offsets_path": offsets_path.name,
                "indices_path": indices_path.name,
            }
        ),
        encoding="utf-8",
    )
    return eg.read_eggpu_csr(metadata_path)


def _write_projection_source(tmp_path):
    arcs = [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 3),
        (4, 3),
        (3, 4),
        (0, 1),
    ]
    rows = [[] for _ in range(5)]
    for source, target in arcs:
        rows[source].append(target)
    offsets = [0]
    indices = []
    for row in rows:
        indices.extend(row)
        offsets.append(len(indices))

    offsets_path = tmp_path / "projection-source.offsets.i32"
    indices_path = tmp_path / "projection-source.indices.i32"
    array("i", offsets).tofile(offsets_path.open("wb"))
    array("i", indices).tofile(indices_path.open("wb"))
    manifest_path = tmp_path / "projection-source.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "eggpu-csr-v1",
                "generation": 1,
                "name": "projection-source",
                "directed": True,
                "num_nodes": 5,
                "num_edges": len(arcs),
                "num_entries": len(indices),
                "node_labels": "zero_based_contiguous",
                "offset_dtype": "int32",
                "index_dtype": "int32",
                "offsets_path": offsets_path.name,
                "indices_path": indices_path.name,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_undirected_projection(tmp_path, *, degree=None):
    projection_dir = tmp_path / "projection"
    projection_dir.mkdir()
    arrays = {
        "lower_V": [0, 2, 3, 4, 5, 5],
        "lower_E": [1, 2, 2, 3, 4],
        "upper_V": [0, 0, 1, 3, 4, 5],
        "upper_E": [0, 0, 1, 2, 3],
        "degree": degree or [2, 2, 3, 2, 1],
        "forward_V": [0, 2, 3, 3, 4, 5],
        "forward_E": [1, 2, 2, 2, 3],
    }
    metadata = {
        "format": "eggpu-logical-undirected-projection-v1",
        "generation": 1,
        "num_nodes": 5,
        "unique_edge_count": 5,
        "offset_dtype": "int32",
        "index_dtype": "int32",
        "degree_dtype": "int32",
    }
    for role, values in arrays.items():
        path = projection_dir / f"{role}.i32"
        array("i", values).tofile(path.open("wb"))
        metadata[f"{role}_path"] = path.name
    projection_manifest = projection_dir / "projection.json"
    projection_manifest.write_text(json.dumps(metadata), encoding="utf-8")
    return projection_manifest


def _canonical_components(components):
    return sorted(tuple(sorted(component)) for component in components)


def test_bulk_graph_contract(tmp_path):
    graph = _write_bulk_graph(tmp_path)
    assert len(graph) == 6
    assert graph.number_of_edges() == 7
    assert list(graph.nodes) == list(range(6))
    assert graph.memory_estimate()["host_total_bytes"] == (7 + 14) * 4
    assert "materialize" not in repr(graph)


def test_logical_undirected_projection_loader(tmp_path):
    source_manifest = _write_projection_source(tmp_path)
    projection_manifest = _write_undirected_projection(tmp_path)

    graph = eg.read_eggpu_csr(
        source_manifest,
        undirected_projection_path=projection_manifest,
    )
    assert graph.undirected_projection_info() == {
        "loaded": True,
        "unique_edge_count": 5,
        "lower_V_size": 6,
        "lower_E_size": 5,
        "upper_V_size": 6,
        "upper_E_size": 5,
        "degree_size": 5,
        "forward_V_size": 6,
        "forward_E_size": 5,
        "lower_W_size": 0,
        "weight_key": "",
    }
    estimate = graph.memory_estimate()
    assert estimate["host_undirected_projection_bytes"] == (6 * 3 + 5 * 4) * 4


def test_undirected_projection_rejects_inconsistent_degree(tmp_path):
    source_manifest = _write_projection_source(tmp_path)
    projection_manifest = _write_undirected_projection(
        tmp_path,
        degree=[2, 2, 99, 2, 1],
    )

    with pytest.raises(
        ValueError, match="degree differs from half views"
    ):
        eg.read_eggpu_csr(
            source_manifest,
            undirected_projection_path=projection_manifest,
            validate=True,
        )


def test_bulk_structural_subset_uses_contiguous_node_ids():
    prepared = {
        "bulk_csr": True,
        "nodes": range(6),
        "node_count": 6,
        "node_to_idx": None,
    }
    result = gpu_eggpu_backend._slice_dense_for_nodes(
        prepared,
        np.asarray([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
        [1, 4],
        {1: 0, 4: 1},
    )

    assert dict(result.items()) == {1: 1.5, 4: 4.5}


def test_weighted_bulk_graph_contract(tmp_path):
    graph = _write_bulk_graph(tmp_path, weighted=True)
    estimate = graph.memory_estimate()
    assert estimate["implicit_unit_weights"] is False
    assert estimate["host_weights_bytes"] == graph.num_entries * 8
    assert estimate["host_total_bytes"] == (7 + 14) * 4 + 14 * 8
    assert graph.weight_key == "weight"


def test_bulk_degree_values_match_graph_semantics(tmp_path):
    undirected_path = tmp_path / "undirected"
    directed_path = tmp_path / "directed"
    undirected_path.mkdir()
    directed_path.mkdir()
    undirected = _write_bulk_graph(undirected_path)
    assert np.array_equal(
        undirected.degree_values(),
        np.asarray([2, 2, 3, 3, 2, 2], dtype=np.float64),
    )
    assert np.array_equal(undirected.nonisolated_mask(), np.ones(6, dtype=bool))

    directed = _write_bulk_graph(directed_path, directed=True)
    # EasyGraph directed degree is in-degree plus out-degree.
    assert np.array_equal(
        directed.degree_values(),
        np.asarray([2, 2, 3, 3, 2, 2], dtype=np.float64),
    )
    assert np.array_equal(directed.nonisolated_mask(), np.ones(6, dtype=bool))


def test_weighted_bulk_degree_values(tmp_path):
    graph = _write_bulk_graph(tmp_path, directed=True, weighted=True)
    expected = np.zeros(6, dtype=np.float64)
    for source, target in EDGES:
        edge_weight = 1.0 + ((source * target) % 6)
        expected[source] += edge_weight
        expected[target] += edge_weight
    assert np.array_equal(graph.degree_values(weight="weight"), expected)


def test_bulk_graph_gpu_matches_regular_graph(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYGRAPH_ENABLE_GPU", "TRUE")
    monkeypatch.setenv("EASYGRAPH_GPU_STRICT_ERRORS", "TRUE")
    monkeypatch.setenv("EASYGRAPH_GPU_ADAPTIVE_HOST", "FALSE")
    regular = _regular_graph()
    bulk = _write_bulk_graph(tmp_path)

    regular_pr = eg.pagerank(regular, alpha=0.75, max_iter=200, tol=1.0e-6)
    bulk_pr = eg.pagerank(bulk, alpha=0.75, max_iter=200, tol=1.0e-6)
    assert np.allclose(
        [regular_pr[node] for node in range(6)],
        [bulk_pr[node] for node in range(6)],
        rtol=1.0e-6,
        atol=1.0e-8,
    )

    assert _canonical_components(eg.connected_components(regular)) == _canonical_components(
        eg.connected_components(bulk)
    )

    regular_bfs = eg.multi_source_bfs(regular, [0, 4])
    bulk_bfs = eg.multi_source_bfs(bulk, [0, 4])
    for source in (0, 4):
        assert dict(regular_bfs[source].items()) == dict(bulk_bfs[source].items())

    regular_core = np.asarray(eg.k_core(regular), dtype=np.int32)
    bulk_core = np.asarray(eg.k_core(bulk), dtype=np.int32)
    assert np.array_equal(regular_core, bulk_core)

    labels = gpu_eggpu_backend.connected_component_labels(bulk, directed=False)
    assert len(labels) == 6
    assert set(labels.keys()) == set(range(6))

    deferred_components = gpu_eggpu_backend.connected_components(
        bulk, directed=False
    )
    assert "component_sets_materialized=False" in repr(deferred_components)
    assert len(deferred_components.labels_numpy()) == len(bulk)
    assert _canonical_components(deferred_components) == _canonical_components(
        eg.connected_components(regular)
    )


def test_pagerank_cache_invalidates_when_alpha_changes(monkeypatch):
    monkeypatch.setenv("EASYGRAPH_ENABLE_GPU", "TRUE")
    monkeypatch.setenv("EASYGRAPH_GPU_STRICT_ERRORS", "TRUE")
    graph = eg.DiGraph()
    graph.add_nodes(range(7))
    graph.add_edges(
        [
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 0),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 5),
            (5, 4),
            (5, 6),
        ]
    )
    pagerank_module = importlib.import_module(
        "easygraph.functions.centrality.pagerank"
    )

    # The second call must rebuild alpha-dependent inbound weights rather than
    # reuse the first call's cached values.
    eg.pagerank(graph, alpha=0.75, max_iter=200, tol=1.0e-7)
    actual = eg.pagerank(graph, alpha=0.90, max_iter=200, tol=1.0e-7)
    expected = pagerank_module._pagerank_power_iteration(
        graph,
        alpha=0.90,
        max_iter=200,
        tol=1.0e-7,
    )

    assert np.allclose(
        [actual[node] for node in range(7)],
        [expected[node] for node in range(7)],
        rtol=2.0e-5,
        atol=1.0e-7,
    )


def test_bulk_wcc_resolves_complete_union_find_parent_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYGRAPH_ENABLE_GPU", "TRUE")
    monkeypatch.setenv("EASYGRAPH_GPU_STRICT_ERRORS", "TRUE")
    bulk = _write_cycle_bulk_graph(tmp_path, 1024)

    deferred_components = gpu_eggpu_backend.connected_components(
        bulk, directed=False
    )

    assert np.unique(deferred_components.labels_numpy()).size == 1
    assert _canonical_components(deferred_components) == [tuple(range(1024))]


def test_weighted_bulk_mst_preserves_graph_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYGRAPH_ENABLE_GPU", "TRUE")
    monkeypatch.setenv("EASYGRAPH_GPU_STRICT_ERRORS", "TRUE")
    regular = _regular_graph()
    for source, target in EDGES:
        regular[source][target]["weight"] = 1.0 + ((source * target) % 6)
    bulk = _write_bulk_graph(tmp_path, weighted=True)

    expected = eg.minimum_spanning_tree(regular, weight="weight")
    actual = eg.minimum_spanning_tree(bulk, weight="weight")

    assert len(actual) == len(bulk) == 6
    assert list(actual) == list(range(6))
    assert list(actual.nodes) == list(range(6))
    assert actual.number_of_edges() == expected.number_of_edges()
    assert actual.node_index[4] == 4
    assert actual.index2node[4] == 4

    expected_weight = sum(data["weight"] for _, _, data in expected.edges)
    actual_weight = sum(data["weight"] for _, _, data in actual.edges)
    assert actual_weight == expected_weight


def test_unweighted_bulk_bc_does_not_require_unit_weight_array(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYGRAPH_ENABLE_GPU", "TRUE")
    monkeypatch.setenv("EASYGRAPH_GPU_STRICT_ERRORS", "TRUE")
    regular = _regular_graph()
    bulk = _write_bulk_graph(tmp_path)

    expected = eg.betweenness_centrality(
        regular,
        sources=[0, 4],
        normalized=False,
        endpoints=False,
    )
    actual = eg.betweenness_centrality(
        bulk,
        sources=[0, 4],
        normalized=False,
        endpoints=False,
    )

    assert np.allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_bulk_graph_never_falls_into_python_adjacency_path(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYGRAPH_ENABLE_GPU", "TRUE")
    monkeypatch.delenv("EASYGRAPH_GPU_STRICT_ERRORS", raising=False)
    bulk = _write_bulk_graph(tmp_path)
    try:
        eg.pagerank(bulk, weight="weight")
    except NotImplementedError as exc:
        assert "weighted PageRank requires an EGGPU CSR artifact" in str(exc)
    else:
        raise AssertionError("topology-only bulk graph accepted weighted PageRank")


def test_directed_bulk_kcore_requires_projection(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYGRAPH_ENABLE_GPU", "TRUE")
    monkeypatch.delenv("EASYGRAPH_GPU_STRICT_ERRORS", raising=False)
    bulk = _write_bulk_graph(tmp_path, directed=True)
    try:
        eg.k_core(bulk)
    except NotImplementedError as exc:
        assert "undirected-projection artifact" in str(exc)
    else:
        raise AssertionError("directed outgoing CSR was accepted as a k-core projection")


def test_gpu_kcore_empty_graph_returns_empty_array(monkeypatch):
    monkeypatch.setenv("EASYGRAPH_ENABLE_GPU", "TRUE")
    monkeypatch.setenv("EASYGRAPH_GPU_STRICT_ERRORS", "TRUE")
    graph = eg.Graph()
    result = np.asarray(eg.k_core(graph), dtype=np.int32)
    assert result.shape == (0,)
