import os
import unittest
from unittest import mock

import numpy as np
import networkx as nx

import easygraph as eg
from easygraph.utils import gpu_eggpu_backend


class EnvGuard:
    def __init__(self, **values):
        self.values = values
        self.old = {}

    def __enter__(self):
        self.old = {name: os.environ.get(name) for name in self.values}
        os.environ.update(self.values)

    def __exit__(self, exc_type, exc, tb):
        for name, value in self.old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def directed_string_graph():
    graph = eg.DiGraph()
    graph.add_edges_from(
        [
            ("source", "left"),
            ("source", "right"),
            ("left", "sink"),
            ("right", "sink"),
            ("sink", "source"),
        ]
    )
    return graph


class TestCentralityPublicContracts(unittest.TestCase):
    def test_cpu_bc_returns_list_in_graph_node_order_for_string_labels(self):
        graph = directed_string_graph()
        sources = ["source", "sink"]
        with EnvGuard(EASYGRAPH_ENABLE_GPU="FALSE"):
            result = eg.betweenness_centrality(
                graph,
                sources=sources,
                normalized=False,
                endpoints=False,
            )

        self.assertIs(type(result), list)
        self.assertEqual(len(result), len(graph))
        self.assertTrue(all(isinstance(value, float) for value in result))
        reference_graph = nx.DiGraph()
        reference_graph.add_edges_from(graph.edges)
        reference = nx.betweenness_centrality_subset(
            reference_graph,
            sources=sources,
            targets=list(reference_graph),
            normalized=False,
            weight=None,
        )
        expected = [reference[graph.index2node[i]] for i in range(len(graph))]
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_cpu_closeness_subset_preserves_requested_string_label_order(self):
        graph = directed_string_graph()
        sources = ["sink", "source", "left"]
        with EnvGuard(EASYGRAPH_ENABLE_GPU="FALSE"):
            result = eg.closeness_centrality(graph, sources=sources)

        self.assertIs(type(result), list)
        self.assertEqual(len(result), len(sources))
        with EnvGuard(EASYGRAPH_ENABLE_GPU="FALSE"):
            each = [eg.closeness_centrality(graph, sources=[node])[0] for node in sources]
        np.testing.assert_allclose(result, each, rtol=0, atol=0)
        reference_graph = nx.DiGraph()
        reference_graph.add_edges_from(graph.edges)
        outgoing_reference = reference_graph.reverse(copy=False)
        expected = [
            nx.closeness_centrality(outgoing_reference, u=node, wf_improved=True)
            for node in sources
        ]
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_gpu_bc_array_is_materialized_as_public_list(self):
        graph = directed_string_graph()
        expected = np.arange(len(graph), dtype=np.float64)
        with EnvGuard(
            EASYGRAPH_ENABLE_GPU="TRUE",
            EASYGRAPH_GPU_STRICT_ERRORS="TRUE",
        ), mock.patch.object(
            gpu_eggpu_backend,
            "betweenness_centrality",
            return_value=expected,
        ):
            result = eg.betweenness_centrality(graph, normalized=False)

        self.assertIs(type(result), list)
        self.assertEqual(result, expected.tolist())

    def test_gpu_closeness_array_is_materialized_as_public_list(self):
        graph = directed_string_graph()
        sources = ["sink", "source"]
        expected = np.asarray([0.25, 0.5], dtype=np.float64)
        with EnvGuard(
            EASYGRAPH_ENABLE_GPU="TRUE",
            EASYGRAPH_GPU_STRICT_ERRORS="TRUE",
        ), mock.patch.object(
            gpu_eggpu_backend,
            "closeness_centrality",
            return_value=expected,
        ):
            result = eg.closeness_centrality(graph, sources=sources)

        self.assertIs(type(result), list)
        self.assertEqual(result, expected.tolist())


if __name__ == "__main__":
    unittest.main()
