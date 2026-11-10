import json

import easygraph as eg
import numpy as np
import pytest

from easygraph.gpu.io import write_partitioned_graph
from easygraph.gpu._dynamic import stage_weights, verify_manifest


def test_partitioned_input_preserves_edges_and_old_weights(tmp_path):
    edges = [(3, 0), (0, 2), (1, 3), (0, 1)]
    weights = [4.0, 2.0, 3.0, 1.0]
    path = write_partitioned_graph(4, edges, tmp_path / "original",
                                   weights=weights, devices=(2, 5))
    document = json.loads(path.read_text())
    verify_manifest(path, document)
    assert [p["device_id"] for p in document["partitions"]] == [2, 5]
    restored = []
    for p in document["partitions"]:
        arrays = {k: np.fromfile(path.parent / spec["path"], dtype=spec["dtype"])
                  for k, spec in p["files"].items()}
        mapping = arrays["local_to_global"]
        for u in range(p["owned_vertex_count"]):
            for index in range(arrays["offsets"][u], arrays["offsets"][u + 1]):
                restored.append((int(mapping[u]), int(mapping[arrays["indices"][index]]),
                                 float(arrays["weights"][index])))
    assert sorted(restored) == sorted((u, v, w) for (u, v), w in zip(edges, weights))
    changed, _ = stage_weights(path, "updated", [(0, 0, 7.0)], tmp_path / "versions")
    verify_manifest(changed, json.loads(changed.read_text()))
    verify_manifest(path, document)
    old = np.fromfile(path.parent / "p0.weights.bin", dtype="<f8")
    new = np.fromfile(changed.parent / "p0.weights.bin", dtype="<f8")
    assert old[0] == 1.0 and new[0] == 7.0


@pytest.mark.parametrize("edges,weights", [([(0, 3)], [1]), ([(0, 1), (0, 1)], [1, 2]),
                                          ([(0, 1)], [-1]), ([(0.5, 1)], [1])])
def test_invalid_graph_does_not_create_destination(tmp_path, edges, weights):
    target = tmp_path / "invalid"
    with pytest.raises(ValueError):
        write_partitioned_graph(3, edges, target, weights=weights)
    assert not target.exists()


def test_core_numbers_ignore_weight_attributes(monkeypatch):
    monkeypatch.setenv("EASYGRAPH_ENABLE_GPU", "FALSE")
    graph = eg.Graph()
    for u, v, weight in [(0, 1, 0.5), (1, 2, 8.5), (2, 0, 2.5), (2, 3, 100)]:
        graph.add_edge(u, v, weight=weight)
    assert eg.k_core(graph) == [2, 2, 2, 1]


def test_undirected_projection_requires_reciprocal_arcs(tmp_path):
    with pytest.raises(ValueError, match="both arc directions"):
        write_partitioned_graph(3, [(0, 1)], tmp_path / "invalid", directed=False)
    path = write_partitioned_graph(3, [(0, 1), (1, 0)], tmp_path / "valid", directed=False)
    document = json.loads(path.read_text())
    assert document["graph"]["directed"] is False
    assert document["view"]["orientation"] == "undirected"
    verify_manifest(path, document)


def test_successive_weighted_and_unweighted_degree_queries(monkeypatch):
    monkeypatch.setenv("EASYGRAPH_ENABLE_GPU", "FALSE")
    graph = eg.Graph()
    graph.add_edge(0, 1, weight=3, cost=7)
    graph.add_edge(1, 2, weight=4, cost=9)
    assert graph.degree() == {0: 3, 1: 7, 2: 4}
    assert graph.degree(weight=None) == {0: 1, 1: 2, 2: 1}
    assert graph.degree(weight="cost") == {0: 7, 1: 16, 2: 9}
    assert eg.efficiency(graph, weight=None) == {0: 1.0, 1: 1.0, 2: 1.0}
    assert eg.k_core(graph) == [1, 1, 1]
    graph[0][1]["weight"] = 5
    assert graph.degree() == {0: 5, 1: 9, 2: 4}
    assert graph.degree(weight=None) == {0: 1, 1: 2, 2: 1}
