# EGGPU

EGGPU adds native CUDA graph analytics to EasyGraph while retaining its Python
API and reusing prepared graph data across compatible function calls.

This repository contains the implementation and correctness checks for
*EGGPU: An End-to-End GPU Acceleration System for Efficient Large-Scale Network
Analysis*, currently under review at VLDB.

## Requirements

- Linux x86-64 and Python 3.10
- CUDA Toolkit 12.8 and a compatible NVIDIA driver
- An NVIDIA GPU; compute capabilities 8.0 and 8.6 have been tested

## Build and check

```bash
conda env create -f environment.yml
conda activate eggpu-artifact
export EGGPU_CUDA_ROOT="$CONDA_PREFIX"
bash scripts/build_eggpu.sh
GPU=0 bash scripts/run_smoke.sh
```

The smoke test compares the 16 supported GPU functions with their EasyGraph
CPU implementations on small deterministic graphs. It does not run performance
experiments.

## Use

Enable EGGPU once:

```bash
export EASYGRAPH_ENABLE_GPU=TRUE
```

Existing EasyGraph calls then use EGGPU when the function and its parameters
are supported:

```python
import easygraph as eg

graph = eg.DiGraph()
graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
rank = eg.pagerank(graph)
```

Supported functions and inputs are listed in
[`artifact/FUNCTION_SUPPORT.md`](artifact/FUNCTION_SUPPORT.md).

## Repository contents

- [`Easy-Graph/`](Easy-Graph/): Python API, C++ bindings, reusable graph data,
  and CUDA implementations.
- [`artifact/`](artifact/): architecture, supported functions, code map,
  correctness checks, and dataset provenance.
- [`scripts/`](scripts/): build and verification commands.
- [`EG_Evaluation/`](EG_Evaluation/): retained evaluation, preprocessing, and
  evidence assembly source from the paper workspace.

Raw datasets, measurements, manuscript files, archived baseline checkouts,
and machine-specific build products remain local and are excluded from Git.

## Development status

Use this repository's `main` branch as the single development entry point.
The September 2026 consolidation retains the implementation published at
`79d74a5`; it adds the evaluation source without changing algorithm behavior.
The separate EGMGPU project is maintained independently.

A clean A100 CUDA build passed all 39 EGGPU contract tests and all 16 small-graph
CPU/GPU comparisons on September 22, 2026. General EasyGraph compatibility
tests exposed failures that must be fixed before an upstream merge; see
[`artifact/STATUS.md`](artifact/STATUS.md). These checks do not establish full
EasyGraph test-suite compatibility or reproduce the large performance matrix.

## Datasets

The 13 benchmark graph datasets used to reproduce EGGPU results are available
on Zenodo:
[10.5281/zenodo.21746036](https://doi.org/10.5281/zenodo.21746036).

See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and
[`CITATION.cff`](CITATION.cff).
