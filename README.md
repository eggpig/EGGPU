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

This repository contains source code and correctness checks. Paper source,
performance results, evaluation scripts, and raw datasets are not included.

## Datasets

The 13 benchmark graph datasets used to reproduce EGGPU results are available
on Zenodo:
[10.5281/zenodo.21746036](https://doi.org/10.5281/zenodo.21746036).

See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and
[`CITATION.cff`](CITATION.cff).
