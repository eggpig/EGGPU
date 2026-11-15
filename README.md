# EGGPU

EGGPU is a multi-GPU graph processing system built on [EasyGraph](https://github.com/easy-graph/Easy-Graph) that reduces redundant graph preparation across queries by coordinating graph state updates and retention under GPU memory constraints.

*EGGPU: An Efficient Multi-GPU Graph Processing System with Coordinated State Management* is under review at ICDE 2027.

## Functions

| Family | Functions |
| --- | --- |
| Centrality | PageRank, betweenness, closeness |
| Connectivity | Local clustering coefficient, weakly connected components, strongly connected components, K-Core |
| Paths and spanning trees | BFS, Dijkstra, Bellman–Ford, SSSP, MST |
| Structural holes | Effective size, efficiency, constraint, hierarchy |

The EasyGraph interface exposes all 16 functions.

## Datasets

The seven graph datasets (EP, LJ, OR, IN, UK, TW, and US) are available on
[Zenodo](https://zenodo.org/records/24134345), with CSR data packages, dataset
and descriptions.

## Build

Build on Linux with Python 3.10 or later, CMake 3.23 or later, and the CUDA Toolkit
with cuBLAS and cuSPARSE. Use a C++ compiler and NVIDIA driver that support the Toolkit.

```bash
git clone https://github.com/eggpig/EGGPU.git
cd EGGPU
python3 -m venv .venv
source .venv/bin/activate
CMAKE_BUILD_PARALLEL_LEVEL=2 \
  CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" \
  python -m pip install .
```

Use CUDA architecture `89` for RTX 4090 or RTX 4090 D and `80` for A100.

## Run

Run the graph functions on one GPU.

```bash
CUDA_VISIBLE_DEVICES=0 EASYGRAPH_ENABLE_GPU=TRUE python examples/functions.py
```

Run the graph versioning example on two GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1 EGGPU_COMM_BACKEND=host_staging \
  python examples/continuous.py --devices 0,1 --directory "$(mktemp -d)"
```

## License

EGGPU is distributed under the [BSD 3-Clause License](LICENSE). See [NOTICE](NOTICE) for third-party attributions.
