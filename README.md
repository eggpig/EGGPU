# EGGPU

EGGPU is an EasyGraph-based GPU graph analytics artifact.  It keeps the public
EasyGraph call style and adds native CUDA dispatch, C++ bindings, reusable graph
context/cache paths, correctness/audit tooling, and paper benchmark scripts.

Current paper scope covers 16 EasyGraph functions:

```text
PageRank, MST, LCC, WCC, SCC, BFS, Dijkstra, BellmanFord, SSSP,
KCore, BC, Closeness, EffectiveSize, Efficiency, Constraint, Hierarchy
```

Repository layout:

- `Easy-Graph/`: EasyGraph source plus EGGPU Python dispatch, C++/pybind
  integration, and CUDA kernels.
- `EG_Evaluation/`: benchmark, ablation, correctness, audit, plotting, dataset,
  and environment files.
- `scripts/`: build, smoke, and compatibility helpers for new machines.

## Install On A New Machine

Clone the artifact:

```bash
git clone https://github.com/eggpig/EGGPU.git
cd EGGPU
```

Create the pinned paper environment:

```bash
conda env create -f EG_Evaluation/environment_eggpu_cuda128.yml
conda activate EGGPU
```

Select a user-space CUDA toolkit.  Do not rely on global CUDA/GCC for paper
numbers.

```bash
export EGGPU_CUDA_ROOT="$CONDA_PREFIX"
export CUDA_PATH="$EGGPU_CUDA_ROOT"
export CUDA_HOME="$EGGPU_CUDA_ROOT"
export CUDAToolkit_ROOT="$EGGPU_CUDA_ROOT"
```

Build in-place for development and evaluation:

```bash
EGGPU_CUDA_ARCHITECTURES=80 bash scripts/build_eggpu.sh
```

Use the matching compute capability on other GPUs, for example
`EGGPU_CUDA_ARCHITECTURES=89` for RTX 4090.  NVIDIA publishes the compute
capability table at https://developer.nvidia.com/cuda-gpus.

The package name remains `Python-EasyGraph`.  For source installs, the EGGPU
build can also be driven directly through pip:

```bash
EASYGRAPH_ENABLE_GPU=TRUE \
EGGPU_CUDA_ROOT="$CONDA_PREFIX" \
EGGPU_CUDA_ARCHITECTURES=80 \
pip install -v ./Easy-Graph
```

Once this repository is available on GitHub, the same path works from the Git
URL:

```bash
EASYGRAPH_ENABLE_GPU=TRUE \
EGGPU_CUDA_ROOT="$CONDA_PREFIX" \
EGGPU_CUDA_ARCHITECTURES=80 \
pip install -v "git+https://github.com/eggpig/EGGPU.git#subdirectory=Easy-Graph"
```

## Compatibility Check

Before running benchmarks on a new server:

```bash
python scripts/check_eggpu_compat.py --strict --expect-rapids
```

This check does not launch GPU kernels.  It verifies Python/package imports,
CUDA root/nvcc visibility, `nvidia-smi`, RAPIDS package availability, and the
EGGPU strict-backend contract.

CUDA compatibility has two layers:

- EGGPU source build: use a CUDA toolkit visible through `EGGPU_CUDA_ROOT` and a
  GPU architecture selected through `EGGPU_CUDA_ARCHITECTURES`.
- Full paper reproduction: RAPIDS/cuGraph/nx-cugraph must also solve and import.
  RAPIDS documents release-specific CUDA/Python support and requires Volta or
  newer NVIDIA GPUs with compute capability 7.0+:
  https://docs.rapids.ai/install and https://docs.rapids.ai/platform-support/.

CUDA 11.x is not the recommended full-paper target.  It can be treated as an
EGGPU-only porting experiment, but the locked paper baseline matrix is written
for CUDA 12.x-or-newer RAPIDS availability.  If a CUDA 11.x machine is required,
first run the compatibility script and expect to rerun the full benchmark matrix
after resolving RAPIDS/baseline differences.

## Runtime Contract

Normal EasyGraph calls stay unchanged:

```python
import easygraph as eg

G = eg.DiGraph()
G.add_edges_from([(0, 1), (1, 2), (2, 0)])
print(eg.pagerank(G))
```

Enable native EGGPU explicitly:

```bash
export EASYGRAPH_ENABLE_GPU=TRUE
export EASYGRAPH_GPU_BACKEND=mine
export EASYGRAPH_GPU_STRICT_ERRORS=TRUE
```

`EASYGRAPH_GPU_STRICT_ERRORS=TRUE` is the paper and debugging mode.  GPU
dispatch failures and unsupported backend names raise clear exceptions instead
of silently timing a CPU fallback.  CPU fallback remains available only when GPU
is disabled or when non-strict exploratory use is intentional.

## Smoke And Full Evaluation

Smoke test on one idle GPU:

```bash
GPU=0 bash scripts/run_smoke.sh
```

Full main benchmark plus ablation:

```bash
cd EG_Evaluation
RUN_LOG="benchmarking/results/main_then_ablation_$(date +%Y%m%d_%H%M%S).console.log" && \
MAIN_GPU=0 ABL_GPU=0 LIBRARY_TIMEOUT=100 ABLATION_TIMEOUT=300 \
bash run_main_and_ablation.sh |& tee "${RUN_LOG}"
```

The workflow runs preflight, main benchmark, main audit, backend separation
audit, pair-level SOTA summary, final summary, and ablation.  It writes
`.run.log` files while also printing progress to tmux.

For parallel main/ablation on two idle GPUs:

```bash
RUN_PARALLEL=1 MAIN_GPU=0 ABL_GPU=1 bash run_main_and_ablation.sh
```

The runner checks GPU idleness before GPU work and fails closed if another
process appears.  Override only for deliberate debugging:
`EGGPU_ALLOW_BUSY_GPU=1`.

## Baselines And Data

Pinned package versions are recorded in
`EG_Evaluation/BASELINE_VERSIONS_20260609.md`; the conda install target is
`EG_Evaluation/environment_eggpu_cuda128.yml`.

Dataset checksums and source hints are recorded in
`EG_Evaluation/datasets/MANIFEST_20260609.tsv`.  For a small GitHub repository,
track the manifest and fetch/stage large processed datasets before full
benchmarking.  For a fully self-contained artifact, the current processed suite
is about 212 MiB and can be tracked, but clones will be heavier.

Generated outputs are intentionally ignored:

- `EG_Evaluation/benchmarking/results/`
- build directories and compiled shared libraries
- raw compressed dataset downloads
- third-party baseline source trees
- local sync or conversation-progress notes

## Methodology Notes

The core evaluation definitions are maintained in:

- `EG_Evaluation/BASELINE_SUPPORT_MATRIX_20260525.md`
- `EG_Evaluation/EGGPU_SECURITY_EVAL_AUDIT_20260602.md`
- `EG_Evaluation/EGGPU_LITERATURE_MAPPING_20260602.md`
- `EG_Evaluation/REPRODUCE_ON_NEW_GPU_20260531.md`

Timing definitions:

- `build`: graph object construction, excluding raw file parse/import.
- `e2e`: user function wall time, including per-call CSR/transfer/sync/result
  wrapping.
- `kernel`: CUDA event time for EGGPU; CPU backend algorithm wall time for CPU
  baselines.
