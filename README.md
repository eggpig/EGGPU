# EGGPU

EGGPU is an EasyGraph-based GPU graph analytics project.  It keeps the
EasyGraph Python call style and adds optional native CUDA dispatch, C++/pybind
bindings, reusable graph context paths, correctness checks, and benchmark
helpers.

Implemented GPU-facing functions include:

```text
PageRank, MST, LCC, WCC, SCC, BFS, Dijkstra, BellmanFord, SSSP,
KCore, BC, Closeness, EffectiveSize, Efficiency, Constraint, Hierarchy
```

## Repository Layout

- `Easy-Graph/`: EasyGraph source, EGGPU Python dispatch, C++ bindings, and
  CUDA kernels.
- `EG_Evaluation/`: public benchmark, smoke-test, dataset manifest, and
  environment files.
- `scripts/`: build, smoke, and compatibility helpers.

## Install

Clone the repository:

```bash
git clone https://github.com/eggpig/EGGPU.git
cd EGGPU
```

Create the conda environment:

```bash
conda env create -f EG_Evaluation/environment_eggpu_cuda128.yml
conda activate EGGPU
```

Select a user-space CUDA toolkit.  The build should not depend on system-wide
CUDA or GCC paths.

```bash
export EGGPU_CUDA_ROOT="$CONDA_PREFIX"
export CUDA_PATH="$EGGPU_CUDA_ROOT"
export CUDA_HOME="$EGGPU_CUDA_ROOT"
export CUDAToolkit_ROOT="$EGGPU_CUDA_ROOT"
```

Build in place:

```bash
EGGPU_CUDA_ARCHITECTURES=80 bash scripts/build_eggpu.sh
```

Use the matching compute capability for other GPUs, for example
`EGGPU_CUDA_ARCHITECTURES=89` for RTX 4090.

The package name remains `Python-EasyGraph`.  A direct source install also
works:

```bash
EASYGRAPH_ENABLE_GPU=TRUE \
EGGPU_CUDA_ROOT="$CONDA_PREFIX" \
EGGPU_CUDA_ARCHITECTURES=80 \
pip install -v ./Easy-Graph
```

## Runtime

Normal EasyGraph calls stay unchanged:

```python
import easygraph as eg

G = eg.DiGraph()
G.add_edges_from([(0, 1), (1, 2), (2, 0)])
print(eg.pagerank(G))
```

Enable EGGPU explicitly:

```bash
export EASYGRAPH_ENABLE_GPU=TRUE
export EASYGRAPH_GPU_BACKEND=mine
export EASYGRAPH_GPU_STRICT_ERRORS=TRUE
```

Strict mode raises an error when GPU dispatch fails instead of silently timing a
CPU fallback.  Disable GPU or leave strict mode off only for exploratory local
debugging.

## Checks And Benchmarks

Run a compatibility check:

```bash
python scripts/check_eggpu_compat.py --strict --expect-rapids
```

Run a small smoke test on one GPU:

```bash
GPU=0 bash scripts/run_smoke.sh
```

Run the public benchmark workflow on one idle GPU:

```bash
cd EG_Evaluation
MAIN_GPU=0 ABL_GPU=0 RUN_PARALLEL=0 bash run_main_and_ablation.sh
```

Generated outputs are written under `EG_Evaluation/benchmarking/results/` and
are intentionally ignored by Git.

## Datasets

Dataset checksums and source hints are recorded in
`EG_Evaluation/datasets/MANIFEST_20260609.tsv`.  For a lightweight clone, keep
the manifest in Git and stage larger public datasets before running the full
benchmark.  See `EG_Evaluation/datasets/DATASETS.md`.

## Timing Definitions

- `build`: graph object construction, excluding raw file parse/import.
- `e2e`: user function wall time, including per-call conversion, transfer,
  synchronization, and result wrapping.
- `kernel`: CUDA event time for EGGPU; algorithm wall time for CPU baselines.

## Ignored Files

The repository does not track build outputs, compiled shared libraries,
benchmark results, local logs, raw downloads, or third-party baseline source
trees.  Keep project-internal notes and unpublished analysis outside this public
export repository.
