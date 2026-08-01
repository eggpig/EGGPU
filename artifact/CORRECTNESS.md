# Correctness verification

Build the extension and run the deterministic GPU smoke:

```bash
bash scripts/build_eggpu.sh
GPU=0 bash scripts/run_smoke.sh
```

The smoke builds small directed and undirected graphs, evaluates all 16 public
functions once with GPU dispatch disabled and once with strict GPU dispatch
enabled, and compares normalized return values. Floating-point results use
explicit tolerances; components and MST outputs are compared structurally. No
latency is recorded and no performance claim is produced.

Source-level tests cover additional invariants:

```bash
PYTHONPATH=Easy-Graph pytest -q \
  Easy-Graph/easygraph/tests/test_eggpu_runtime.py \
  Easy-Graph/easygraph/tests/test_eggpu_cache_contracts.py \
  Easy-Graph/easygraph/tests/test_eggpu_centrality_contracts.py \
  Easy-Graph/easygraph/tests/test_eggpu_mst_fractional_weights.py
```

The cache tests verify mutation invalidation, generation binding, dense result
shape checks, read-only mappings, and independent cached snapshots. The bulk
CSR tests construct their fixtures in a temporary directory and can be run
separately with `test_eggpu_bulk_csr.py`.

Passing this smoke establishes functional agreement on the included cases. It
does not replace the full dataset- and parameter-level validation used for the
paper's experimental artifact.
