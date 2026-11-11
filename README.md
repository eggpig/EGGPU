# EGGPU

EGGPU is a GPU graph-analysis system built on EasyGraph. It provides Python
graph functions and persistent sessions for analysis across functions, graph
versions, and graph instances, with single- and multi-GPU execution.

- **D-Maintain** reuses compatible graph components and updates affected
  weights through in-place maintenance or copy-on-write.
- **A-Retain** tracks shared physical allocations and selects removable view
  sets under per-GPU memory budgets.
- Owning host results remain valid after subsequent requests and session closure.

## Functions

| Family | Functions |
| --- | --- |
| Centrality | PageRank, betweenness, closeness |
| Connectivity | Local clustering coefficient, weakly connected components, strongly connected components, K-Core |
| Paths and spanning trees | BFS, Dijkstra, Bellman–Ford, SSSP, MST |
| Structural holes | Effective size, efficiency, constraint, hierarchy |

The EasyGraph interface exposes all 16 functions. Partitioned native sessions
provide BFS, SSSP, WCC, PageRank, and weighted PageRank over one or more GPUs.

## Datasets

The seven graph datasets (EP, LJ, OR, IN, UK, TW, and US) are available on
[Zenodo](https://zenodo.org/records/24134345), with CSR data packages, dataset
descriptions, and checksums.

## Build

Requirements: Linux, Python 3.10 or newer, a C++ compiler supported by your
CUDA Toolkit, CMake 3.23 or newer, and the full CUDA Toolkit including cuBLAS
and cuSPARSE. The NVIDIA driver must support the selected Toolkit.

```bash
git clone https://github.com/eggpig/EGGPU.git
cd EGGPU
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
CMAKE_BUILD_PARALLEL_LEVEL=2 \
  CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80" \
  python -m pip install '.[test]'
```

Set `CMAKE_CUDA_ARCHITECTURES` to the target GPU architecture: `80` for A100,
`89` for RTX 4090/4090 D, or a semicolon-separated list for multiple targets.
Set `CUDACXX` to `nvcc` when selecting a non-default Toolkit. For a CPU-only
build, set `EASYGRAPH_ENABLE_GPU=OFF` during installation.

## Usage

Select host-staged communication for PCIe systems without GPU peer access:

```bash
export EGGPU_COMM_BACKEND=host_staging
```

Use `peer_to_peer` when all selected GPU pairs support peer access.

Run the function examples on a visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 EASYGRAPH_ENABLE_GPU=TRUE \
  EASYGRAPH_GPU_STRICT_ERRORS=TRUE python examples/functions.py
```

Run a sequence that updates edge weights, queries the new version, and revisits
the original graph on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/continuous.py --devices 0,1
```

The example constructs its own small graph under `example-data/` and checks the
returned distances. Add `--budget-mib 256` to enable A-Retain with a 256 MiB
retained-state budget per GPU. Select a fresh directory with `--directory` for
another run. Device IDs are relative to `CUDA_VISIBLE_DEVICES`.

For application inputs, `easygraph.gpu.io.write_partitioned_graph` writes
partitioned CSR from directed arcs and optional nonnegative weights. Vertices
are numbered from zero. For WCC, use `directed=False` and supply both arc
directions to construct the undirected projection.

```python
from easygraph.gpu import Session
from easygraph.gpu.io import write_partitioned_graph

manifest = write_partitioned_graph(
    4, [(0, 1), (1, 2), (2, 3)], "example-data/input",
    weights=[1.0, 2.0, 1.0], devices=(0, 1),
)
with Session.native({"G": manifest}, devices=(0, 1), route="lease_sparse",
                    budgets={0: 256 << 20, 1: 256 << 20}, policy="closure") as session:
    distances = session.query("G", "sssp", {"source": 0}).values
    print(distances)
```

In unbudgeted sessions, `retain_snapshot` preserves an older version's prepared
views during weight maintenance; `release_snapshot` removes that retention
request. Budgeted sessions use the selected retention policy. `inventory()` reports the
managed views, allocations, and workspace state. Topology changes enter a
session as a new manifest through `register_snapshot`.

## Tests

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

The tests cover graph-state invalidation, shared allocation accounting, result
ownership, version updates, and numerical input contracts. The continuous
example additionally checks native GPU execution.

## Source layout

- `easygraph/`: Python API, graph state, and execution policies.
- `cpp_easygraph/`: native graph objects, Python bindings, and runtime.
- `gpu_easygraph/`: CUDA operators and device state management.
- `examples/`, `tests/`: usage and correctness checks.

## License

BSD 3-Clause. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for component licenses
and algorithm attributions. The GPU MST implementation adapts
[ECL-MST](https://github.com/burtscher/ECL-MST).
