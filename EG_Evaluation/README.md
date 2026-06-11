# EG_Evaluation

This directory contains public evaluation helpers for EGGPU: benchmark runners,
correctness checks, ablation scripts, environment files, and dataset manifests.

The current benchmark runner covers:

```text
PageRank, MST, LCC, WCC, SCC, BFS, Dijkstra, BellmanFord, SSSP,
KCore, BC, Closeness, EffectiveSize, Efficiency, Constraint, Hierarchy
```

## Entry Point

From the repository root:

```bash
cd EG_Evaluation
MAIN_GPU=0 ABL_GPU=0 RUN_PARALLEL=0 bash run_main_and_ablation.sh
```

The workflow runs preflight checks, the main benchmark, correctness/backend
audits, summary generation, optional large-graph Closeness supplement, and
ablation scripts.  Use an idle GPU for comparable timing.

Useful environment variables:

- `MAIN_GPU`: GPU index for the main benchmark.
- `ABL_GPU`: GPU index for ablations.
- `RUN_PARALLEL`: set to `1` only when `MAIN_GPU` and `ABL_GPU` are different
  idle GPUs.
- `RUN_PREFLIGHT`: set to `FALSE` only for local debugging.
- `LIBRARY_TIMEOUT`: per baseline/function timeout for the main benchmark.
- `ABLATION_TIMEOUT`: per run timeout for ablations.
- `EGGPU_GPU_VISIBILITY_MARKER`: optional fixed allocation marker for shared
  servers; default is `FALSE`.

The runner writes logs and generated outputs under `benchmarking/results/`.
These outputs are intentionally ignored by Git.

## Environment

Use the provided conda environment file:

```bash
conda env create -f environment_eggpu_cuda128.yml
conda activate EGGPU
```

Before a full run on a new machine, check the environment:

```bash
python ../scripts/check_eggpu_compat.py --strict --expect-rapids
```

## Datasets

Processed dataset paths, sizes, checksums, and source hints are recorded in
`datasets/MANIFEST_20260609.tsv`.  See `datasets/DATASETS.md` for the dataset
policy.

## Timing Definitions

- `build`: graph object construction, excluding raw file parse/import.
- `e2e`: user function wall time, including per-call conversion, transfer,
  synchronization, and result wrapping.
- `kernel`: CUDA event time for EGGPU; algorithm wall time for CPU baselines.
