# EGGPU Cross-GPU Reproduction Plan

This note freezes the current repository layout and the recommended path for
reproducing EGGPU on another GPU machine, such as an RTX 4090 host.

## Current Local State

- EasyGraph source tree: `Easy-Graph/`
- Evaluation and paper artifact tree: `EG_Evaluation/`
- Original paper machine class: NVIDIA A100, CUDA-capable user-space conda
  environment, no reliance on global CUDA or GCC.
- The original local workspace path is intentionally not part of the public
  reproduction contract.  Clone the repository anywhere writable by the user.

For paper reproduction, treat the current artifact repository as the unit of
versioning. Do not merge upstream EasyGraph casually after paper numbers are
frozen; upstream refresh should happen on a separate branch followed by a full
rerun.

## Recommended Git Layout

Use one GitHub repository for the paper artifact:

1. `EGGPU`
   - Contains `Easy-Graph/` with the EGGPU Python dispatch, C++ bindings, CUDA
     kernels, graph context/cache, adaptive policies, and public APIs.
   - Contains `EG_Evaluation/` with benchmark scripts, ablation scripts, audit
     scripts, environment files, dataset manifests, and paper artifact
     generators.
   - Does not track build outputs, compiled shared libraries, raw downloads,
     third-party source trees, or large benchmark result folders.

This single-repository layout is the current reproducibility target. It avoids
commit drift between a code repository and an evaluation repository.

## EasyGraph Patch Scope

The current EGGPU patch is not just a collection of kernels. It changes three
layers of EasyGraph:

1. Public Python layer
   - Adds GPU-aware dispatch while preserving EasyGraph-style function calls.
   - Keeps CPU behavior available when GPU is disabled.
   - Touches graph classes, decorators, connected components, clustering,
     PageRank, paths, MST, KCore, BC, and structural-hole APIs.

2. C++ binding and middle layer
   - Adds graph conversion and reusable graph context/cache paths.
   - Stores CSR-like graph layouts and avoids rebuilding/transferring the same
     graph repeatedly when possible.
   - Adds lighter return paths for functions where returning full Python objects
     dominates user-facing time.

3. CUDA layer
   - Implements or integrates GPU kernels for:
     `PageRank`, `MST`, `LCC`, `WCC`, `SCC`, `BFS`, `Dijkstra`,
     `BellmanFord`, `SSSP`, `KCore`, `BC`, `Closeness`, `EffectiveSize`,
     `Efficiency`, `Constraint`, and `Hierarchy`.
   - Adds common runtime utilities, buffer/device graph cache, adaptive transfer
     policy, and per-function hot-path policies.

The four paper-facing categories can be presented as:

- Ranking and centrality: `PageRank`, `BC`, `Closeness`
- Connectivity and core: `LCC`, `WCC`, `SCC`, `KCore`
- Paths and spanning structure: `BFS`, `Dijkstra`, `BellmanFord`, `SSSP`, `MST`
- Structural holes: `EffectiveSize`, `Efficiency`, `Constraint`, `Hierarchy`

The evaluation repository contains the scripts that measure E2E time, kernel
time, build/load time, GPU/process memory, correctness, backend isolation, and
ablation variants.

## What To Commit

Commit in the artifact repository:

- `easygraph/` public API and GPU dispatch changes.
- `cpp_easygraph/` pybind11 and C++ graph/cache changes.
- `gpu_easygraph/` CUDA kernels and common runtime/cache code.
- Build metadata changes needed for GPU compilation.
- `benchmarking/*.py`
- `run_main_and_ablation.sh`
- `environment_eggpu_cuda128.yml`
- reproducibility and methodology notes that are still current.
- dataset manifests and any intentionally selected smoke-test data.
- paper table/figure generation scripts.

Do not commit:

- `Easy-Graph/cpp_easygraph.cpython-*.so`
- `Easy-Graph/build/`, `Easy-Graph/.eggs/`, `*.egg-info/`
- `benchmarking/results/`
- `paper_sync/`
- local build folders and compiled binaries.
- third-party baseline source trees as ordinary vendored files. Prefer
  submodules, pinned clone scripts, or documented download commands.
- raw compressed downloads under `datasets/raw_downloads/`.

## Environment Policy

Do not rely on global CUDA or global GCC.

Use an isolated conda environment and a user-space CUDA toolkit. The current
artifact assumes:

- Python 3.10.x, with baseline package pins recorded in
  `BASELINE_VERSIONS_20260609.md`
- conda env name: `EGGPU`
- benchmark env file: `environment_eggpu_cuda128.yml`
- user-space CUDA 12.x toolkit, normally from `$CONDA_PREFIX` or another
  user-writable CUDA root selected through `EGGPU_CUDA_ROOT`
- no reliance on system `nvcc`, system CUDA headers, or global GCC

The evaluation env currently includes RAPIDS/cuGraph/nx-cugraph 26.02. On a new
RTX 4090 machine, use a recent NVIDIA driver and a CUDA 12.x or newer runtime.
The build script sets `CMAKE_CUDA_ARCHITECTURES` from
`EGGPU_CUDA_ARCHITECTURES`, defaulting to `80` for the original A100 server.
Use `EGGPU_CUDA_ARCHITECTURES=89` for RTX 4090, or the matching compute
capability for another GPU.  Avoid `all` for paper builds because it is slower
and can introduce PTX/toolchain compatibility variance across machines.

CUDA 11.x is not the recommended full-paper reproduction target.  EGGPU-only
source builds may be investigated on CUDA 11.8-class toolchains, but the full
baseline matrix depends on RAPIDS/cuGraph/nx-cugraph packages that are pinned
here for the CUDA 12.x-or-newer RAPIDS path.  Run
`python scripts/check_eggpu_compat.py --strict --expect-rapids` on every new
machine before starting expensive benchmark runs.

## GPU Selection Guidance, 2026-06-08

The latest reviewed successful EGGPU memory rows are small relative to modern
GPU memory sizes: process-tree GPU peak mean about `456 MiB`, P95 about
`623 MiB`, and maximum about `2002 MiB`; whole-device peak maximum about
`2778 MiB`.  These are diagnostic sizing numbers from
`full_eval_gpu0_20260603_234802_selfloop_filtered_scc_trim16_kcore_threshold_constraint_auto`,
not final paper evidence.

Recommended cross-machine comparison set:

- A100 80GB: original datacenter reference.  NVIDIA documents A100 40GB/80GB
  configurations at https://www.nvidia.com/en-us/data-center/a100/.
- RTX 4090 24GB or RTX 5090 32GB: high-end consumer/workstation comparison.
  NVIDIA documents RTX 4090 24GB GDDR6X at
  https://www.nvidia.com/en-me/geforce/graphics-cards/40-series/rtx-4090/ and
  RTX 5090 32GB GDDR7 at
  https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/.
- One 16GB consumer card: good evidence that EGGPU does not need 80GB-class
  memory for the current graph suite.

Do not make an 8GB GPU the main comparison host unless the goal is explicitly a
portability/stress experiment.  EGGPU itself appears to fit easily in the
current suite, but RAPIDS/nx-cugraph, CUDA contexts, memory fragmentation, and
future larger graphs need headroom.  RAPIDS/cuGraph currently requires an
NVIDIA Volta-or-newer GPU with compute capability 7.0+:
https://docs.rapids.ai/api/cugraph/stable/tutorials/basic_cugraph/.

## New Machine Bootstrap

Example layout:

```bash
mkdir -p ~/workspace/haorandu
cd ~/workspace/haorandu
git clone https://github.com/eggpig/EGGPU.git EGGPU
```

Create the env:

```bash
cd ~/workspace/haorandu/EGGPU
conda env create -f EG_Evaluation/environment_eggpu_cuda128.yml
conda activate EGGPU
```

Set a user-space CUDA root. Replace this path with the CUDA toolkit installed
inside conda or another user-writable location on the 4090 machine:

```bash
export EGGPU_CUDA_ROOT="$CONDA_PREFIX"
export CUDA_PATH="$EGGPU_CUDA_ROOT"
export CUDA_HOME="$EGGPU_CUDA_ROOT"
export CUDAToolkit_ROOT="$EGGPU_CUDA_ROOT"
```

Build EasyGraph in-place:

```bash
cd ~/workspace/haorandu/EGGPU
EGGPU_CUDA_ARCHITECTURES=89 bash scripts/build_eggpu.sh
```

Use `EGGPU_CUDA_ARCHITECTURES=80` or omit the variable on an A100.

Run the non-kernel compatibility check first:

```bash
cd ~/workspace/haorandu/EGGPU
python scripts/check_eggpu_compat.py --strict --expect-rapids
```

Then verify GPU preflight on an idle GPU:

```bash
cd ~/workspace/haorandu/EGGPU/EG_Evaluation
CUDA_VISIBLE_DEVICES=0 \
EGGPU_MONITOR_GPU_INDEX=0 \
EGGPU_CUDA_ROOT="$CONDA_PREFIX" \
CUDA_PATH="$CONDA_PREFIX" \
CONDA_PREFIX="$CONDA_PREFIX" \
EASYGRAPH_ENABLE_GPU=TRUE \
EASYGRAPH_GPU_BACKEND=mine \
EASYGRAPH_GPU_STRICT_ERRORS=TRUE \
PYTHONPATH=~/workspace/haorandu/EGGPU/Easy-Graph:${PYTHONPATH:-} \
python benchmarking/preflight_full_eval_ready.py
```

The standalone preflight includes a small GPU structural-hole check.  Run it
only on an idle GPU.  The official `run_main_and_ablation.sh` command below
runs the same preflight automatically before the main benchmark and stops if it
fails.

## Full Benchmark Command

Run inside your own tmux session. Pick idle GPU IDs on the new machine.

```bash
cd ~/workspace/haorandu/EGGPU/EG_Evaluation && \
RUN_LOG="benchmarking/results/main_then_ablation_$(date +%Y%m%d_%H%M%S).console.log" && \
MAIN_GPU=0 ABL_GPU=1 LIBRARY_TIMEOUT=100 ABLATION_TIMEOUT=300 \
bash run_main_and_ablation.sh |& tee "${RUN_LOG}"
```

For a single-GPU sequential run:

```bash
MAIN_GPU=0 ABL_GPU=0 LIBRARY_TIMEOUT=100 ABLATION_TIMEOUT=300 \
bash run_main_and_ablation.sh |& tee "${RUN_LOG}"
```

The script checks whether the selected GPU is idle. Override the check only for
debugging, not for paper numbers:

```bash
EGGPU_ALLOW_BUSY_GPU=1 MAIN_GPU=0 ABL_GPU=0 bash run_main_and_ablation.sh
```

The official workflow also runs a preflight before the main benchmark and writes
`benchmarking/results/<full_eval_id>.preflight.log`.  Leave
`RUN_PREFLIGHT=TRUE` for paper-quality runs; use `RUN_PREFLIGHT=FALSE` only when
debugging a known preflight issue.

The full runner also checks the selected GPU before every EGGPU child process.
If another job starts using the card during a long run, the affected EGGPU rows
are written as failed rows with a `gpu_busy_before_eggpu_child` note, and the
audit rejects the result.  This is deliberate: mid-run GPU contention is not
paper-quality timing evidence.

## Historical Smoke Check

The workspace had the following smoke checks on 2026-05-31 before repository
publication:

- Direct user API smoke passed with `EASYGRAPH_ENABLE_GPU=TRUE` and
  `EASYGRAPH_GPU_BACKEND=mine`:
  - `eg.clustering`
  - `eg.connected_components`
  - `eg.pagerank`
  - `eg.minimum_spanning_edges`
- Non-strict release fallback smoke passed:
  - with `EASYGRAPH_ENABLE_GPU=TRUE`, `EASYGRAPH_GPU_STRICT_ERRORS=FALSE`, and
    an invalid backend name, path and structural-hole calls returned through CPU
    fallback instead of crashing.
  - set `EASYGRAPH_GPU_STRICT_ERRORS=TRUE` for paper/debugging mode.  In strict
    mode, unsupported backend names and GPU dispatch failures raise hard
    exceptions rather than silently timing CPU fallback.
- Full-eval preflight passed:
  - workspace `easygraph` import path
  - workspace `cpp_easygraph` import path
  - run-script configuration
  - structural-hole correctness preflight
- Mini full benchmark smoke passed:
  - output directory:
    `benchmarking/results/repro_smoke_full_20260531_233753`
  - dataset/functions: `LastFM`, `PageRank,Constraint`
  - expected note: NetworkX `Constraint` timed out under the intentionally short
    60-second smoke timeout; EGGPU rows completed.
- Mini workflow ablation smoke passed:
  - output file:
    `benchmarking/results/repro_smoke_ablation_20260531_233753.csv`
  - dataset/functions: `LastFM`, `PageRank,Constraint`

Those checks are historical evidence.  The current official entry script now
has additional preflight, metadata, validation, and backend-separation gates.
A true clone-from-GitHub reproduction still requires committing all EasyGraph
EGGPU changes and all evaluation scripts listed in this document, then running
the full command above on an idle GPU.

## Shell Hygiene

Do not capture paper logs from a shell that prints or exports machine service
environment variables during startup.  The benchmark child processes sanitize
compiler/CUDA variables, but a noisy outer shell can still leak environment
values into console logs before the benchmark starts.

The official preflight checks this explicitly.  If your `.bashrc` reads
`/proc/1/environ` before the non-interactive-shell guard, the
`outer_shell_hygiene` check is a warning by default because user dotfiles are
outside the artifact repository.  Set `EGGPU_STRICT_OUTER_SHELL_HYGIENE=1`
only when you want this to be a hard local audit failure.  The cleanest fix is
to remove those lines, or move them after:

```bash
case $- in
    *i*) ;;
      *) return;;
esac
```

Do not use `EGGPU_ALLOW_BUSY_GPU=1`, `RUN_PREFLIGHT=FALSE`, or
`EGGPU_USE_CONDA_RUN=TRUE` for paper numbers unless the resulting run is labeled
as a debug-only run.  The audit expects the default direct-child Python path,
strict GPU errors, EGGPU warmup of two calls, and no baseline warmup.

The default official run also records `EASYGRAPH_GPU_BC_WARP_SIZE=AUTO`,
`EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS=1024`,
`EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE=AUTO`, and
`EGGPU_CLOSENESS_EXACT_MAX_NODES=1000000` in metadata.  Keep these defaults for
paper reproduction unless you are running a named ablation.  KCore's default
single-block gate is graph-aware and does not select the single-block path from
high maximum degree alone.  The generated final summary includes the full
result plus `nodes>=10000`, `gpu-friendly`, and `paper-core` filtered views,
with a documented `0.05%` timing-tie tolerance and a `2%` near-miss table for
targeted reruns.

EasyGraph optional PyTorch/torch-geometric import warnings are quiet by default
so benchmark logs stay focused on graph timing, correctness, and backend
separation.  Set `EASYGRAPH_SHOW_OPTIONAL_IMPORT_WARNINGS=1` only when debugging
optional neural-network or hypergraph dependencies.

## Result Management

Keep raw full-result directories outside Git. For each official run, archive:

- `benchmarking/results/<full_eval_dir>/`
- `benchmarking/results/<ablation_dir>/`
- console log from `RUN_LOG`
- `git rev-parse HEAD` for the artifact repository
- `nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv`
- `conda list --explicit` or a lockfile if exact package reproduction is needed.

## Upstream Update Policy

Use two branches:

- `eggpu-paper`: frozen paper branch, no upstream merge unless numbers
  are rerun.
- `eggpu-rebase-upstream`: experimental upstream merge branch for future
  maintainability.

Do not mix upstream merge conflict resolution with paper result reproduction.
