# Baseline Support Matrix, 2026-05-25

Last rechecked against upstream documentation on 2026-06-03.

This document reflects the current EGGPU benchmark scope after adding path
algorithms and structural-hole metrics.

Current benchmark functions:

- Ranking / centrality: `PageRank`, `BC`, `Closeness`
- Connectivity / local structure: `LCC`, `WCC`, `SCC`, `KCore`
- Paths / trees: `MST`, `BFS`, `Dijkstra`, `BellmanFord`, `SSSP`
- Structural holes: `EffectiveSize`, `Efficiency`, `Constraint`, `Hierarchy`

Legend:

- `Y`: supported and attempted by the current benchmark.
- `P`: partial support, semantic caveat, native fallback, timeout/OOM risk, or
  not fully comparable kernel timing.
- `N`: not supported or deliberately skipped.

## Current Runner Coverage

| Baseline | PR | MST | LCC | WCC | SCC | BFS | Dijkstra | BellmanFord | SSSP | KCore | BC | Closeness | EffSize | Efficiency | Constraint | Hierarchy | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EGGPU | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Integrated EasyGraph GPU path. EGGPU gets `--easygraph-warmup 2`; other baselines get no EasyGraph warmup. |
| EasyGraph CPU | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Pure Python/EasyGraph CPU path with GPU explicitly disabled in runner. |
| EasyGraph C++ | Y | Y | Y | Y | Y | Y | Y | N | Y | Y | Y | Y | N | N | N | N | C++ Bellman-Ford binding is absent. Structural-hole C++ bindings route to CUDA in the GPU-enabled build, so they are skipped to keep this baseline CPU-only. |
| NetworkX | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | P | Y | N | Directed Closeness is run on a reverse graph view to match EasyGraph outward-distance semantics. `Efficiency` is derived as `effective_size / degree`; NetworkX has no aligned Burt hierarchy API. |
| igraph | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | P | N | N | Y | N | Closeness uses `mode=OUT` for directed graphs and an explicit Wasserman-Faust reachable-fraction correction to align with EasyGraph/NetworkX. `Constraint` is native. Effective size, efficiency, and hierarchy have no aligned native API in python-igraph. |
| nx-cugraph / native cuGraph fallback | P | P | Y | Y | P | Y | Y | Y | Y | P | Y | N | N | N | N | N | nx-cugraph supports many NetworkX shortest-path APIs, PageRank, clustering, WCC, core-number, and BC; runner may use native cuGraph fallback where NetworkX backend coverage is incomplete. Closeness and structural-hole metrics are not in the supported nx-cugraph/cuGraph algorithm list. SCC has fallback/OOM risk on very large directed graphs. KCore/core-number is partial because directed-graph support is not universal across NetworkX backend and native cuGraph paths. |
| Gunrock | Y | P | Y | N | N | Y | N | N | Y | Y | Y | N | N | N | N | N | Local executable set is `pr,mst,lcc,bfs,sssp,kcore,bc`. Gunrock docs list CC, but no usable local `cc` binary is available in the current build, and there is no aligned local Closeness executable. MST has connected-component semantics caveats on disconnected graphs. |

## Timing And Memory Semantics

The benchmark emits only three time metrics:

- `build`: baseline-native graph-object construction after the raw edge list has
  already been parsed. Import time is excluded. Function-specific preparation
  such as CSR conversion is not counted here; it remains in `e2e` because users
  pay it at function-call time.
- `kernel`: exact CUDA kernel time when the backend exposes it. For CPU
  libraries, `kernel` equals algorithm wall time by definition. For
  nx-cugraph/cuGraph and Gunrock CLI paths that do not expose a clean in-process
  event timer, `kernel` is the best available algorithm timer and is annotated
  in notes.  EGGPU's Closeness CUDA event stops before the final device-to-host
  copy of the score vector, so its `kernel` value excludes Python/result
  wrapping and output transfer; those costs remain in `e2e`.
- `e2e`: user-visible function-call wall time after graph construction,
  including per-call preparation, transfer, synchronization, and result wrapping.

EGGPU-specific tuning values that affect timing are recorded in run metadata.
As of 2026-06-04, BC has a BC-only warp-size policy for large sparse directed
graphs; `EASYGRAPH_GPU_BC_WARP_SIZE=AUTO` keeps that policy, while explicit
`1/2/4/8/16/32` values force a sweep/debug setting.
KCore records `EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS=1024` and
`EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE=AUTO`; the default single-block
gate is graph-aware and no longer uses high maximum degree alone to select the
single-block path on medium low-average-degree graphs.

SOTA summaries use a fixed relative timing-tie tolerance of `0.05%`.  This is
only a noise tie rule: pairs outside that tolerance are non-SOTA.  The final
summary also emits a `2%` near-miss table so close losses can be rerun or
targeted without silently changing the verdict.

Memory metrics:

- `memory_peak_rss_mb`: process-tree peak CPU RSS during the measured algorithm
  window.
- `memory_peak_gpu_mb` / `memory_avg_gpu_mb`: whole-device GPU memory usage
  sampled through NVML during the subprocess window.
- `memory_peak_gpu_delta_mb` / `memory_avg_gpu_delta_mb`: device memory delta
  from the subprocess-start baseline.
- `memory_peak_gpu_proc_mb` / `memory_avg_gpu_proc_mb`: process-tree GPU memory
  where NVML can attribute memory to benchmark child processes.
- `memory_peak_gpu_proc_delta_mb` / `memory_avg_gpu_proc_delta_mb`: process-tree
  GPU memory delta from the subprocess-start baseline.

For paper tables, prefer the process-tree GPU memory metrics when available.
Whole-device memory is retained as a diagnostic to catch external interference
and context pressure, but it can include other processes if the idle-GPU gate is
disabled or the machine is otherwise contaminated.

The optional `EGGPU_GPU_VISIBILITY_MARKER` CUDA-context marker is disabled by
default in the official runner.  Enable it only for interactive GPU-reservation
visibility in `nvitop`/`nvidia-smi`; keep it disabled for paper-quality memory
tables unless the run metadata explicitly records a fixed marker allocation.
When `EGGPU_GPU_VISIBILITY_MARKER=TRUE` and
`EGGPU_GPU_VISIBILITY_MARKER_MB=<N>` are set, the long-lived driver process
allocates `<N>` MiB as a visible marker.  If no marker size is supplied, the
runner uses `256` MiB.  The default
`EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB=AUTO` measures the actual whole-device
increment caused by the marker process, including CUDA context overhead, and
subtracts that fixed value only from whole-device absolute memory metrics
(`memory_peak_gpu_mb` / `memory_avg_gpu_mb`).  Process-tree GPU memory and all
delta memory metrics are left unadjusted because the fixed marker is outside
the measured child process tree and already belongs to the subprocess-start
device baseline.

The official runner performs two levels of idle-GPU protection.  The entry
script checks the selected GPU before preflight, main, and ablation stages; the
full runner also checks before every EGGPU child process.  If another process
starts using the selected card during a long run, the affected EGGPU timing rows
are failed with a `gpu_busy_before_eggpu_child` note and the audit rejects the
result.

Latest memory-sizing observation, reviewed 2026-06-08:

- Source:
  `benchmarking/results/full_eval_gpu0_20260603_234802_selfloop_filtered_scc_trim16_kcore_threshold_constraint_auto/results_long.csv`.
- These numbers are diagnostic successful-row observations from a run whose
  audit did not pass; they are not final paper evidence.
- EGGPU process-tree GPU peak memory over successful rows: mean `456 MiB`,
  median `420 MiB`, P95 `623 MiB`, max `2002 MiB`.
- In the gpu-friendly successful subset: mean `476 MiB`, median `428 MiB`, P95
  `675 MiB`, max `2002 MiB`.
- Whole-device peak memory over successful EGGPU rows: mean `1232 MiB`, median
  `1196 MiB`, P95 `1399 MiB`, max `2778 MiB`.
- Largest process-tree GPU-memory cases were `wiki-Talk / BC` (`2002 MiB`) and
  `web-NotreDame / Closeness` (`1240 MiB`).
- Against nx-cugraph on common successful gpu-friendly rows, EGGPU was the same
  memory order: process-tree peak mean `485 MiB` versus nx-cugraph `471 MiB`,
  median ratio about `0.991x`, with EGGPU lower on about `63%` of pairs.  The
  `wiki-Talk / BC` outlier makes the mean ratio slightly above 1.

## Reporting Filters

The final summary reports multiple fixed views:

- `full`: all valid benchmark pairs.
- `nodes>=10000`: pairs whose dataset has at least 10k raw nodes.
- `gpu-friendly`: excludes the fixed small/low-work dataset set in
  `summarize_final_result.py`.
- `paper-core`: starts from `gpu-friendly` and additionally excludes fixed
  component-output-dominated WCC/SCC pairs where the public API must materialize
  many Python component sets.  The excluded pairs are listed in the generated
  final summary.  This filter is code-defined and reproducible; it must not be
  edited ad hoc for a result directory.

## Closeness Semantic Alignment

EasyGraph's Closeness benchmark uses outward shortest-path distances on
directed graphs and the Wasserman-Faust correction for disconnected graphs. The
runner therefore applies the following baseline-specific rules:

- `EGGPU`, `easygraph-cpu`, and `easygraph-cpp` call EasyGraph-compatible
  Closeness directly. The CPU-only preflight checks all-source and
  `sources=[...]` subset returns against EasyGraph CPU.  In GPU mode,
  unweighted Closeness uses an exact source-parallel BFS path over cached device
  CSR; weighted Closeness keeps the Dijkstra-style CUDA path.
- `NetworkX` computes directed closeness as inward distance by default, so the
  benchmark evaluates `nx.closeness_centrality(G.reverse(copy=False),
  wf_improved=True)` for directed inputs.
- `python-igraph` supports direction through `mode="OUT"`, but its normalized
  output does not by itself match the EasyGraph/NetworkX disconnected-graph
  scale. The benchmark multiplies igraph's value by
  `(reachable_count - 1) / (N - 1)` to apply the Wasserman-Faust correction.
- `nx-cugraph` and native cuGraph are marked unsupported for Closeness because
  the official supported-algorithm pages list centrality algorithms such as
  betweenness, degree, eigenvector, and Katz, but not Closeness.
- `Gunrock` is marked unsupported for Closeness because the current local
  executable set has no aligned Closeness binary.
- Exact all-source Closeness on graphs above
  `EGGPU_CLOSENESS_EXACT_MAX_NODES` is skipped symmetrically across exact
  library baselines.  The default threshold is `1,000,000` nodes.  This is a
  scale guard, not unsupported-backend fallback; skipped rows carry an explicit
  scale-guard note and are accepted by the audit only with that note.

The implementation check is `benchmarking/preflight_closeness_semantics.py`.
It compares EasyGraph CPU, EasyGraph C++, NetworkX, and igraph on both directed
outward-distance and disconnected-undirected cases, including source-subset
return order.

## Upstream Evidence Checked

- NetworkX shortest-path docs list BFS/unweighted shortest path, Dijkstra, and
  Bellman-Ford as core single-source choices:
  https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
- NetworkX Closeness docs state that directed Closeness uses inward distance by
  default and outward distance requires acting on `G.reverse()`. They also
  document the Wasserman-Faust disconnected-graph correction:
  https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
- NetworkX structural-hole docs expose `constraint` and `effective_size`, but
  not hierarchy:
  https://networkx.org/documentation/stable/reference/algorithms/structuralholes.html
- python-igraph `GraphBase.closeness` documents `mode="out"` and normalized
  Closeness, and the `Graph` API lists `pagerank`, `spanning_tree`,
  `connected_components`, `distances`, `betweenness`, `constraint`, and
  `coreness`:
  https://python.igraph.org/en/develop/api/igraph.GraphBase.html
- nx-cugraph supported algorithms include PageRank, clustering, connected/WCC,
  core number, shortest-path APIs including single-source Dijkstra and
  Bellman-Ford, BFS traversal APIs, and BC, but not Closeness:
  https://docs.rapids.ai/api/cugraph/nightly/nx_cugraph/supported-algorithms/
- NetworkX's `core_number` page notes that the cuGraph backend exists, but
  directed graphs are not yet supported for that backend:
  https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.core.core_number.html
- cuGraph API docs list PageRank, BC, connected components, k-core, BFS, SSSP,
  and MST families. The supported/planned algorithm table lists centrality
  support for Katz, betweenness, edge betweenness, eigenvector, and degree, but
  not Closeness:
  https://docs.rapids.ai/api/cugraph/nightly/graph_support/algorithms/
- Gunrock algorithm docs list BFS, SSSP, PageRank, MST, KCore, BC, CC, and LGC.
  The current benchmark still depends on local compiled executables, so CC
  remains unavailable until a working `cc` binary exists:
  https://gunrock.github.io/gunrock/gunrock.wiki/Graph-Algorithms.html

The 2026-06-03 recheck changed the document, not the benchmark behavior:

- NetworkX still exposes structural-hole `effective_size` and `constraint`, but
  not EasyGraph-compatible hierarchy.
- Closeness is now documented explicitly as an EGGPU/EasyGraph
  CPU/EasyGraph C++/NetworkX/igraph benchmarked function and as unsupported
  for nx-cugraph/cuGraph/Gunrock in the current runner.
- python-igraph still exposes fast C-core APIs for PageRank, spanning tree,
  connected components, shortest paths, betweenness, coreness, and Burt
  constraint, but not aligned effective size, efficiency, or hierarchy.
- nx-cugraph/cuGraph still supports the standard graph-algorithm subset used in
  the runner, including PageRank, clustering, connected/WCC, core number,
  shortest-path families, BFS traversal APIs, and BC.  Structural-hole metrics
  remain unsupported.  KCore remains partial because directed-graph support is
  not universal across the NetworkX backend and native cuGraph paths.
- Gunrock documentation lists CC, but the current local benchmark remains
  constrained by the compiled executable set; no usable local CC binary is
  available in this workspace.

Checked source pages:

- nx-cugraph supported algorithms page lists PageRank, clustering, WCC,
  `core_number`, BFS traversal APIs, Dijkstra/Bellman-Ford shortest-path APIs,
  and BC.
- Gunrock algorithms page lists BC, BFS, CC, KCore, MST, PageRank, and SSSP
  among supported primitives; local executable availability is still narrower.
- python-igraph `GraphBase` documents `closeness(mode="out")`, and the
  `Graph` API lists `pagerank`, `spanning_tree`,
  `connected_components`/`components`, `distances`, `betweenness`,
  `constraint`, and `coreness`.

## Local Verification Notes

- `benchmarking/run_full_baselines.py` and
  `benchmarking/library_baselines.py` both define the 16-function scope above.
- `run_main_and_ablation.sh` now calls `--functions all` for the full benchmark
  and passes `all` to workflow/return ablations, so the one-command script no
  longer drops the newly added functions.
- `run_main_and_ablation.sh` now runs `benchmarking/audit_full_result.py` after
  the main full benchmark and before ablations. By default ablations still run
  after a failed main audit so the diagnostic data is produced, but the final
  command exit code remains nonzero. Set `RUN_ABLATION_ON_MAIN_FAILURE=FALSE`
  for fail-fast behavior.
- `run_full_baselines.py` pins CuPy/NVRTC child-process CUDA headers to the
  selected local CUDA toolkit and uses the selected EGGPU Python directly by
  default. `conda run` is opt-in through `EGGPU_USE_CONDA_RUN=TRUE`.
- Local Gunrock binaries currently found:
  `bc`, `bfs`, `kcore`, `lcc`, `mst`, `pr`, `sssp`.
- `benchmarking/validate_correctness.py` validates full detail arrays for
  vector outputs, path distance matrices, connected-component labels, k-core
  numbers, and structural-hole vectors where detail files are emitted. Gunrock
  BFS/SSSP rows use Gunrock internal validation when the CLI reports zero
  validation errors.
