import os
import unittest

import easygraph as eg


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


def total_weight(graph):
    return sum(float(data.get("weight", 1.0)) for _, _, data in graph.edges)


class TestEGGPUMSTFractionalWeights(unittest.TestCase):
    def test_fractional_weights_are_not_rounded_for_edge_selection(self):
        graph = eg.Graph()
        graph.add_edge("a", "b", weight=1.49)
        graph.add_edge("a", "c", weight=1.01)
        graph.add_edge("b", "c", weight=1.02)

        with EnvGuard(
            EASYGRAPH_ENABLE_GPU="TRUE",
            EASYGRAPH_GPU_STRICT_ERRORS="TRUE",
        ):
            tree = eg.minimum_spanning_tree(graph, weight="weight")

        self.assertAlmostEqual(total_weight(tree), 2.03, places=12)
        selected = {
            frozenset((u, v)) for u, v, _ in tree.edges
        }
        self.assertEqual(
            selected,
            {frozenset(("a", "c")), frozenset(("b", "c"))},
        )

    def test_sub_float32_weight_differences_still_determine_the_tree(self):
        graph = eg.Graph()
        graph.add_edge("a", "b", weight=1.00000005)
        graph.add_edge("a", "c", weight=1.00000001)
        graph.add_edge("b", "c", weight=1.00000002)

        with EnvGuard(
            EASYGRAPH_ENABLE_GPU="TRUE",
            EASYGRAPH_GPU_STRICT_ERRORS="TRUE",
        ):
            tree = eg.minimum_spanning_tree(graph, weight="weight")

        selected = {frozenset((u, v)) for u, v, _ in tree.edges}
        self.assertEqual(
            selected,
            {frozenset(("a", "c")), frozenset(("b", "c"))},
        )
        self.assertAlmostEqual(total_weight(tree), 2.00000003, places=12)


if __name__ == "__main__":
    unittest.main()
