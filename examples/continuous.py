"""Run versioned graph requests on the selected GPUs."""
import argparse
from pathlib import Path

import numpy as np

from easygraph.gpu import Session
from easygraph.gpu.io import write_partitioned_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0", help="comma-separated CUDA device IDs")
    parser.add_argument("--directory", type=Path, default=Path("example-data"))
    parser.add_argument("--budget-mib", type=int, help="optional retained-state budget per GPU")
    args = parser.parse_args()
    devices = tuple(map(int, args.devices.split(",")))
    n = max(16, len(devices) * 2)
    edges = np.column_stack((np.arange(n - 1), np.arange(1, n)))
    manifest = write_partitioned_graph(n, edges, args.directory / "G0",
                                       weights=np.ones(n - 1), devices=devices)
    topology = write_partitioned_graph(n, edges, args.directory / "topology", devices=devices)
    budgets = None if args.budget_mib is None else {d: args.budget_mib * (1 << 20) for d in devices}
    with Session.native({"G0": manifest, "topology": topology}, devices=devices,
                        route="lease_sparse", budgets=budgets, policy="closure") as session:
        original = session.query("G0", "sssp", {"source": 0})
        np.testing.assert_array_equal(original.values, np.arange(n))
        if budgets is None:
            session.retain_snapshot("G0")
        # Partition 0, its first stored edge: replace weight 1 with weight 3.
        session.commit_weights("G0", "G1", [(0, 0, 3.0)], args.directory / "versions")
        updated = session.query("G1", "sssp", {"source": 0})
        np.testing.assert_array_equal(updated.values, np.r_[0, np.arange(1, n) + 2])
        bfs = session.query("topology", "bfs", {"source": 0})
        np.testing.assert_array_equal(bfs.values, np.arange(n))
        revisited = session.query("G0", "sssp", {"source": 0})
        np.testing.assert_array_equal(revisited.values, original.values)
        print("Original distances:", original.values)
        print("Updated distances: ", updated.values)
        print("BFS levels:        ", bfs.values)
        session.release_snapshot("G0")
    print("Version update and revisit checks passed.")


if __name__ == "__main__":
    main()
