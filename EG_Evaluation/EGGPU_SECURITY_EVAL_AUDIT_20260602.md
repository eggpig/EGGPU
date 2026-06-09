# EGGPU Security and Evaluation Audit, 2026-06-02

This note records the current benchmark workflow audit after the
`full_eval_gpu0_20260602_000626_selfloop_filtered_scc_trim16_kcore_threshold_constraint_auto`
run.

## Current Result Status

The 2026-06-02 full run is not a final paper result because the full audit gate
failed on four EGGPU runtime timeout rows:

- `pgp / PageRank`
- `com-youtube / Efficiency`
- `com-youtube / Constraint`
- `com-youtube / Hierarchy`

The completed EGGPU correctness rows are clean:

- EGGPU validation rows: 251
- EGGPU validation bad rows: 0

The timeout logs did not print the `library_baselines.py` mode header. This
indicates timeout before backend entry rather than an EGGPU/easygraph-cpp path
mix. A backend log scan found no backend-mode mismatch among non-timeout rows:

- EGGPU: `mode=gpu`, `EASYGRAPH_ENABLE_GPU=TRUE`,
  `EASYGRAPH_GPU_BACKEND=mine`, `easygraph_warmup=2`
- easygraph-cpp: `mode=cpp`, GPU disabled, `easygraph_warmup=0`
- easygraph-cpu: `mode=cpu`, GPU disabled, `easygraph_warmup=0`

## Fixes Applied

The main benchmark previously used `conda run` wrappers for child benchmark
processes. The timeout rows showed that this wrapper could time out before the
child benchmark script emitted its mode header.

The official runner now uses the selected EGGPU Python directly by default:

- `run_main_and_ablation.sh` exports `EGGPU_CHILD_PYTHON="${COMMON_PY}"`.
- `run_main_and_ablation.sh` exports `EGGPU_USE_CONDA_RUN=FALSE`.
- `run_full_baselines.py` constructs child commands as
  `[DIRECT_CHILD_PYTHON, benchmarking/library_baselines.py, ...]`.
- `conda run` remains available only when `EGGPU_USE_CONDA_RUN=TRUE` is set
  explicitly.

Direct child processes now also pin the CUDA toolkit variables explicitly:

- `EGGPU_CUDA_ROOT`
- `CUDA_PATH`
- `CUDA_HOME`
- `CUPY_CUDA_PATH`
- `CUDAToolkit_ROOT`
- `CONDA_PREFIX`

This keeps CuPy/cuGraph JIT paths aligned with the user-space CUDA toolkit
without changing global CUDA or GCC.

## Safety Checks

Repository-level checks performed:

- Shell syntax check passed for `run_main_and_ablation.sh`,
  `scripts/build_eggpu.sh`, and `scripts/run_smoke.sh`.
- Python source syntax check passed for the benchmark runner, ablation runner,
  paper-artifact generator, preflight script, and legacy SCC/cuGraph comparison
  script.
- Legacy `shell=True` usage in the SCC/cuGraph comparison helper was removed.
- Paper artifact generation now reads the current `no_adaptive_policy`
  ablation variant instead of the obsolete `adaptive_policy` file.
- The reproduction document now describes the current single-repository
  `EGGPU` layout.

Additional workflow hardening after the follow-up safety review:

- `run_main_and_ablation.sh` now runs `benchmarking/summarize_final_result.py`
  as part of the main audit group.  If final SOTA/result summary generation
  fails, the whole workflow exits nonzero instead of silently producing an
  incomplete result directory.
- `benchmarking/preflight_full_eval_ready.py` now checks that the final summary
  stage is wired into the main workflow.
- `scripts/build_eggpu.sh` refuses to clean any build directory except the
  repository-local `Easy-Graph/build` path.
- `scripts/run_smoke.sh` now performs the same idle-GPU guard as the main
  workflow before launching GPU preflight or smoke benchmarks.
- `benchmarking/summarize_final_result.py --out-dir` now creates the output
  directory before writing CSV/Markdown artifacts.
- The final summary report now avoids duplicate worst-case rows across
  multiple dataset filters and counts EGGPU timeout rows as non-SOTA.
- A process scan found no leftover `run_main_and_ablation.sh`,
  `run_full_baselines.py`, `run_eggpu_ablations.py`, or
  `library_baselines.py` benchmark process after the review.
- `scripts/build_eggpu.sh` now checks for future timestamps before building.
  Future timestamps in `Easy-Graph/build` are treated as stale build cache and
  removed.  Future timestamps in source files stop the build unless
  `EGGPU_ALLOW_CLOCK_SKEW=1` is set for a debug-only build.
- `scripts/build_eggpu.sh` now defaults to the conda-environment `Ninja`
  generator when available.  This avoids the Makefile clock-skew warning seen
  on the shared filesystem while still using the same local CUDA toolkit.
- `scripts/build_eggpu.sh` now prefers `libgomp` from the selected CUDA root
  before falling back to the Python environment.  This keeps OpenMP and CUDA
  runtime libraries in the same user-space toolkit when possible and reduces
  unsafe runtime search-path cycles.
- The C++ `CSRMatrix` helper types in `eigenvector.cpp` and
  `katz_centrality.cpp` are now in anonymous namespaces.  This removes the
  previous LTO One Definition Rule warning where two different file-local helper
  classes had the same external name.
- `benchmarking/run_full_baselines.py` now writes `run_metadata.json` at the
  start of every full run and refreshes it at completion.  The metadata records
  the git commit, dirty status, selected local CUDA toolkit, Python executable,
  benchmark arguments, whitelisted EGGPU environment variables, and
  `cpp_easygraph` shared-library paths/mtimes/sizes.  This closes the
  reproducibility gap where a result directory did not prove which code and
  compiled extension produced it.
- `benchmarking/preflight_full_eval_ready.py` now checks that result metadata
  generation is wired into the full runner and that the build script contains
  the clock-skew guard, safe build-root deletion guard, Ninja default, local
  CUDA pinning, and `libgomp` pinning.
- `benchmarking/audit_full_result.py` now treats missing or inconsistent
  `run_metadata.json` as a hard audit failure.  The audit writes
  `audit/metadata_issues.csv` and includes metadata status in
  `audit/audit_summary.json`, while leaving `git.dirty=true` as a visible
  warning rather than a development-time hard blocker.
- `benchmarking/preflight_full_eval_ready.py` now includes a lightweight
  metadata-gate self-test: it creates a temporary minimal result without
  `run_metadata.json` and verifies that `benchmarking/audit_full_result.py`
  fails that result and writes `audit/metadata_issues.csv`.
- `benchmarking/validate_correctness.py` no longer treats an EGGPU-only row as
  established correctness.  If EGGPU is the selected reference because no other
  comparable baseline produced correctness fields, the row is marked
  `inconclusive_self_reference`.  `benchmarking/audit_full_result.py` now
  requires EGGPU validation rows to be `pass`; `weak_pass`, `reference`, and
  inconclusive rows are hard blockers for the EGGPU correctness gate.

## Environment Risk Found

The current shell environment has a machine-level hazard outside the EGGPU
repository: `/home/batchcom/.bashrc` reads `/proc/1/environ` and exports those
variables before the usual non-interactive-shell guard.  When a non-interactive
`bash` command starts, this can print or expose service environment variables
and pollute build or benchmark logs.

This file was not modified by the EGGPU audit because it is outside the paper
artifact repository.  For paper-quality reproduction, remove those lines or move
them after the interactive-shell check before running build or benchmark
commands in a login shell.

## Follow-up Safety Review

Additional source-level review on 2026-06-02 found two workflow hardening gaps
and both were fixed:

- `benchmarking/audit_full_result.py` no longer relies on a stale static
  expected-function list. It derives the expected functions from
  `run_metadata.json` when available, falling back to observed result rows only
  for legacy directories. Missing EGGPU coverage is now a hard audit failure.
- `easygraph/functions/components/weakly_connected.py` now computes the strict
  GPU-error flag before entering the dispatch body, so an import/setup failure
  cannot accidentally bypass `EASYGRAPH_GPU_STRICT_ERRORS=TRUE`.
- `benchmarking/audit_full_result.py` now treats a missing/empty
  `correctness_validation.csv`, or any `ok` EGGPU e2e row without a matching
  `pass` validation row, as a hard audit failure.  The audit writes
  `audit/validation_issues.csv`, so missing correctness evidence cannot be
  mistaken for a successful run.
- `benchmarking/audit_full_result.py` also checks
  `run_metadata.json -> artifacts.validation_error`.  If the full runner
  recorded a correctness-validation generation error, the result is rejected
  explicitly through `audit/metadata_issues.csv`.
- `benchmarking/preflight_full_eval_ready.py` now self-tests both the metadata
  gate, validation-evidence gate, and validation-error metadata gate before a
  final full benchmark.
- `benchmarking/library_baselines.py` now re-raises graph-context prewarm
  failures under `EASYGRAPH_GPU_STRICT_ERRORS=TRUE`.  This prevents benchmark
  mode from hiding an EGGPU C++ graph/cache construction failure behind a later
  fallback or rebuild attempt.
- `benchmarking/preflight_closeness_semantics.py` was added as a CPU-only
  semantic check for Closeness.  It verifies that EasyGraph C++,
  NetworkX-with-directed-reverse, and igraph-with-Wasserman-Faust correction
  match EasyGraph CPU on directed-outward and disconnected-undirected cases.
  `benchmarking/preflight_full_eval_ready.py` runs this check before the GPU
  structural-hole preflight.
- `run_main_and_ablation.sh` now runs
  `benchmarking/preflight_full_eval_ready.py` before the main benchmark by
  default.  The preflight uses the same local CUDA, Python, strict-error, and
  EGGPU environment as the main run.  If preflight fails, the official workflow
  stops before producing timing rows.  This prevents a bad environment from
  generating a result directory that only fails after a long run.
- `RUN_PREFLIGHT=FALSE` remains available only for deliberate debugging.  A
  paper-quality run should leave the default enabled.
- `benchmarking/preflight_full_eval_ready.py` now checks that the official
  entry script still wires the preflight stage and writes a `.preflight.log`;
  removing that stage will cause preflight itself to fail.
- `run_main_and_ablation.sh` now runs the main benchmark and ablation bodies in
  child subshells.  This keeps any internal `set -e` behavior from bypassing
  the parent script's explicit exit-code handling, so main-run failure, audit
  failure, and ablation failure remain separately reportable.
- `benchmarking/preflight_full_eval_ready.py` now checks function-registry
  consistency across the main runner, library runner, ablation runner, paper
  artifact generator, final summary, and audit script.  This prevents newly
  integrated functions such as `Closeness` from being measured in one stage but
  omitted from correctness validation, paper tables, or audit coverage.
- `benchmarking/preflight_full_eval_ready.py` now checks the GPU routing
  contract explicitly: the runner exposes the requested physical GPU through
  `CUDA_VISIBLE_DEVICES`, NVML memory accounting uses
  `EGGPU_MONITOR_GPU_INDEX`, and the visibility marker uses logical CUDA device
  0 inside that restricted process.  This prevents future edits from mixing
  physical GPU IDs with CUDA logical device IDs.
- `benchmarking/preflight_full_eval_ready.py` now checks the public GPU
  dispatch strict-error contract.  Every EasyGraph public GPU dispatch entry
  must check `gpu_runtime_enabled()`, route through the native `mine_backend`
  path, catch GPU dispatch exceptions, and re-raise them when
  `EASYGRAPH_GPU_STRICT_ERRORS=TRUE`.  This prevents a paper benchmark from
  silently timing a CPU fallback as an EGGPU success row after a native GPU
  failure.
- `benchmarking/preflight_full_eval_ready.py` now checks the Closeness baseline
  semantic contract.  NetworkX must use a reverse directed graph view,
  igraph must use outward mode plus the Wasserman-Faust reachable-fraction
  correction, and nx-cugraph/cuGraph must stay skipped for Closeness because
  the official supported-algorithm lists do not include it.  This prevents a
  future edit from turning a semantic mismatch or unsupported GPU backend into
  a paper timing row.
- `benchmarking/preflight_full_eval_ready.py` now has its own idle-GPU guard
  before the structural-hole GPU smoke check.  This protects direct/manual
  preflight invocations: if the target GPU is already occupied, the preflight
  fails clearly and does not touch CUDA kernels or allocate device memory.  The
  official `run_main_and_ablation.sh` script still performs the same idle check
  before invoking preflight.
- `benchmarking/summarize_category_estimate.py` no longer has hard-coded
  historical result directories as defaults.  The complete full result and
  targeted replacement result must be supplied explicitly, avoiding accidental
  reuse of stale timing evidence.
- `run_main_and_ablation.sh` now leaves
  `EGGPU_GPU_VISIBILITY_MARKER` disabled by default.  The marker can still be
  enabled explicitly for an interactive "please do not use this GPU" nvitop
  signal.  If `EGGPU_GPU_VISIBILITY_MARKER_MB=<N>` is set, the marker reserves a
  fixed `<N>` MiB allocation in the long-lived driver process; if no size is
  supplied, the official runner uses `256` MiB.  The default
  `EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB=AUTO` measures the actual whole-device
  memory increment caused by the marker process, including CUDA context
  overhead, and subtracts that value only from whole-device absolute memory
  metrics.  Process-tree memory and delta memory are not adjusted.  A
  paper-quality timing run can keep the marker off, or must report the measured
  marker adjustment through `run_metadata.json`.
- `benchmarking/preflight_full_eval_ready.py` now checks that the official
  runner keeps the visibility marker disabled by default, uses explicit
  marker-size/auto-adjust rules when enabled, and propagates the adjustment path.
  This makes the fair-memory policy a preflight contract rather than an operator
  convention.
- `benchmarking/run_full_baselines.py` now isolates plot/table generation in a
  child process with `EGGPU_PLOT_TIMEOUT`, and supports
  `EGGPU_SKIP_PLOTS=TRUE` for timing-only/debug runs.  Plotting failures are
  recorded in `run_metadata.json -> artifacts.plot_error` instead of killing a
  completed timing/correctness result before `completed_at` is written.

2026-06-08 follow-up:

- `benchmarking/gpu_visibility_marker.py` now supports an explicit fixed
  marker allocation through `EGGPU_GPU_VISIBILITY_MARKER_MB`.  The marker uses
  `cudaMalloc` in the long-lived driver process, launches no kernels, and logs
  `marker_mb=<N>` and `whole_device_adjust_mb=<M>` to stderr.
- `benchmarking/run_full_baselines.py` and
  `benchmarking/library_baselines.py` subtract
  `EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB` only from whole-device absolute memory
  metrics.  They do not adjust process-tree memory or delta memory.
- The adjustment path is deliberately controlled by
  `EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB`.  The official default is `AUTO`,
  which marker initialization resolves to a numeric value before benchmark child
  processes and memory monitors run.
- Preflight now checks marker default-off behavior, enabled-marker defaults,
  propagation through the official script, fixed-memory allocation support, and
  both runner-side memory adjustment hooks.
- Verified in this session with Python bytecode compilation, `bash -n`,
  full preflight on GPU 0, a real 16 MiB marker `cudaMalloc`/`cudaFree` smoke
  test, and an adjustment-rule smoke test.

The restricted source scan did not find real secret material in project source,
scripts, or reproduction documents. Generic scanner hits were code-variable
false positives. Historical `benchmarking/results` directories were excluded
from the focused scan because they are large generated artifacts and should not
be versioned.

The scan did find many historical notes that intentionally mention old result
directories and old absolute paths.  These are archival documents, not active
workflow inputs.  For a public paper repository, keep the latest reproduction
document and mark older notes as historical or move them under an archive
directory so they cannot be mistaken for current instructions.

During this review the non-interactive shell issue in `/home/batchcom/.bashrc`
was observed again: starting a regular `bash` command can print machine
environment variables into command output.  The EGGPU scripts sanitize the
benchmark child environment, but the outer interactive shell should still be
fixed before publishing logs or running commands that may be captured.

The official preflight now reports this pattern as an `outer_shell_hygiene`
warning by default instead of a hard blocker.  User dotfiles such as
`/home/batchcom/.bashrc` live outside the EGGPU workspace and should not be
modified by the benchmark workflow.  The benchmark child processes still
sanitize their own environment, which is the part under project control.  Set
`EGGPU_STRICT_OUTER_SHELL_HYGIENE=1` only for a deliberate local reproducibility
audit where blocking on the user's shell startup files is acceptable.

Import-time optional-dependency noise was also reduced.  Several EasyGraph
package `__init__.py` files previously printed PyTorch/torch-geometric missing
messages during every `import easygraph`, even when the user only called graph
algorithms.  Those messages now stay silent by default and are printed only when
`EASYGRAPH_SHOW_OPTIONAL_IMPORT_WARNINGS=1` is set.  This removes irrelevant
text from benchmark and preflight logs without hiding actual GPU dispatch
errors, which are still controlled by `EASYGRAPH_GPU_STRICT_ERRORS`.

All existing `full_eval_gpu*_2026060*_selfloop_filtered_scc_trim16_kcore_threshold_constraint_auto`
result directories found during this review lacked `run_metadata.json`; they
therefore remain useful for debugging/trend analysis only. A paper-quality
result must be generated by the current runner and must pass the metadata gate.

CPU-only checks after these hardening changes passed:

- Shell syntax: `run_main_and_ablation.sh`, `scripts/build_eggpu.sh`,
  `scripts/run_smoke.sh`.
- Python syntax: `benchmarking/audit_full_result.py`,
  `benchmarking/library_baselines.py`, and related benchmark/audit scripts.
- A minimal synthetic result without `correctness_validation.csv` now fails the
  full audit with exit code 2 and writes `audit/validation_issues.csv`.
- On 2026-06-03, the Closeness CPU semantic preflight passed for EasyGraph C++,
  NetworkX, and igraph on directed-outward, source-subset, and disconnected
  undirected cases.  A no-bytecode Python syntax compile also passed for the
  main runner, library runner, preflight, full audit, backend-separation audit,
  and GPU visibility marker helper.
- Direct preflight gate checks passed for both missing metadata and missing
  validation evidence.
- Direct preflight gate checks passed for a synthetic non-empty
  `artifacts.validation_error` in `run_metadata.json`.
- The new GPU routing contract preflight check passed with no missing clauses.
- The Closeness CPU-only preflight passed.  The largest observed mismatch was
  floating-point roundoff (`5.55e-17`) on the disconnected-undirected case.
- A direct manual preflight invocation on the current machine did not enter the
  structural-hole CUDA smoke check because `nvidia-smi` could not communicate
  with the driver.  This is the intended fail-closed behavior: when GPU
  idleness cannot be verified, the workflow stops before allocating device
  memory or producing timing rows.
- The Closeness CUDA kernel launch configuration was corrected.  The previous
  source used the Dijkstra occupancy configuration to launch the min-edge
  kernel and the min-edge configuration to launch the Dijkstra kernel, while
  allocating the Dijkstra workspace with the Dijkstra grid size.  That could
  corrupt memory or make runtime unstable if the two grid sizes diverged.  The
  Dijkstra grid is now capped by the number of active sources, and preflight has
  a static launch-contract check to prevent this regression from returning.
- `benchmarking/preflight_full_eval_ready.py` now also checks compiled-extension
  freshness for the Closeness C++/CUDA path.  It compares the in-place
  `cpp_easygraph` shared library mtime against the tracked C++/CUDA inputs
  (`cpp_easygraph.cpp`, centrality binding files, `gpu_easygraph.h`,
  `centrality.cpp`, and `closeness_centrality.cu/.cuh`).  This catches the
  common failure mode where source code was fixed but the benchmark would still
  import an older `.so`.
- After rebuilding with the local conda CUDA toolkit, the freshness check
  passed: the loaded `cpp_easygraph` shared library mtime was newer than
  `gpu_easygraph/functions/centrality/closeness_centrality.cu`, the newest
  tracked source input.
- The Closeness GPU output length now follows the active source count instead
  of always returning `|V|` values.  This preserves all-source behavior because
  `sources.size()==|V|` in the benchmark path, and it also aligns the lower
  C++/CUDA path with EasyGraph's `sources` subset semantics.  A CPU-only binding
  check confirmed that `sources=[0, 10]` returns two values matching the
  corresponding entries from the all-source result.

## Workflow Safety Review, 2026-06-03

The current workflow was reviewed from the perspective of paper-result safety,
shared-server safety, and reproducibility.  The important invariant is now:
if the GPU path, build artifact, baseline isolation, validation evidence, or
GPU-idle condition is not provable, the official workflow fails instead of
quietly producing paper timing rows.

Checks that are now covered before a full run:

- The official runner uses the repository-local EasyGraph tree through
  `PYTHONPATH` and the in-place `cpp_easygraph` extension.
- The loaded `cpp_easygraph` shared object must be newer than the tracked
  C++/CUDA source inputs for the Closeness path.
- The run script must keep EGGPU warmup at two EasyGraph warmups while all
  other baselines use zero extra warmup.
- The function registry must match across the main runner, library runner,
  ablation runner, paper artifact generator, final summary, and audit script.
- Child benchmark processes use the selected EGGPU Python directly by default;
  `conda run` is only opt-in.
- GPU routing is split correctly: `CUDA_VISIBLE_DEVICES` exposes the chosen
  physical GPU, CUDA code uses logical device 0 inside that process, and NVML
  memory monitoring uses `EGGPU_MONITOR_GPU_INDEX`.
- Build safety is checked statically: local CUDA root, local `nvcc`, safe
  build-root deletion, clock-skew guard, Ninja default, compile-environment
  sanitization, and OpenMP `libgomp` pinning.
- Public EasyGraph GPU dispatch functions must re-raise native GPU failures
  under `EASYGRAPH_GPU_STRICT_ERRORS=TRUE`, preventing hidden CPU fallback
  from being timed as EGGPU.
- Closeness baseline semantics are checked before a run: NetworkX uses a
  reverse directed graph view, igraph uses outward mode with
  Wasserman-Faust correction, and nx-cugraph/cuGraph remains skipped for
  unsupported Closeness.
- Backend separation is now checked before and after a run.  The new preflight
  static gate verifies that EGGPU is the only EasyGraph mode with GPU enabled,
  result cache disabled, graph-context prewarm enabled, and warmup applied.
  EasyGraph CPU/C++ modes must explicitly disable GPU and CUDA sync.  The
  post-run backend audit still verifies the actual logs and result rows.
- Main audit, backend separation audit, pair-level SOTA summary, and final
  summary are all part of the official workflow's exit status.

Manual checks performed during this review:

- AST syntax check passed for `preflight_full_eval_ready.py`,
  `audit_backend_separation.py`, `audit_full_result.py`,
  `run_full_baselines.py`, and `library_baselines.py`.  A normal
  `py_compile` was not used because the restricted filesystem cannot write
  `__pycache__`; this is a permission issue, not a syntax issue.
- A focused repository scan did not find real credential material such as API
  keys, private keys, or tokens outside expected documentation mentions of the
  previous `/proc/1/environ` shell hazard.
- Generated result directories, compiled shared objects, and local build
  directories are ignored by git: `EG_Evaluation/benchmarking/results/`,
  `Easy-Graph/*.so`, and `Easy-Graph/build/` all resolve to ignore rules.
  This prevents accidental publication of large generated artifacts or
  machine-specific binaries.
- Direct preflight passed every static and CPU-only gate, including
  `backend_separation_static_contract=ok`,
  `public_gpu_dispatch_strict_errors=ok`,
  `closeness_baseline_semantics_contract=ok`, and the Closeness CPU semantic
  preflight.
- Direct preflight then failed at `gpu_preflight_idle_guard` because
  `nvidia-smi` could not communicate with the driver on the current machine.
  This is the intended fail-closed behavior: the workflow stops before the
  structural GPU preflight or any timing benchmark can touch device memory.

Remaining risks are operational rather than hidden-code risks:

- The repo is still dirty, so a public/paper repository should commit or
  otherwise freeze the exact source state before claiming final reproducibility.
- `RUN_PREFLIGHT=FALSE`, `EGGPU_ALLOW_BUSY_GPU=1`, and
  `EGGPU_USE_CONDA_RUN=TRUE` remain available for debugging only.  They should
  not be used for paper-quality runs.
- `RUN_ABLATION_ON_MAIN_FAILURE=TRUE` lets ablations continue after a main
  failure, which is useful for collecting evidence, but the overall script
  still exits nonzero if the main run or audits fail.  Treat any such result as
  diagnostic, not final.
- Generated historical result directories and old notes should not be versioned
  as active evidence.  Use a fresh full run that passes `run_metadata.json`,
  correctness, backend separation, and final-summary audits.

## Follow-up Hardening, 2026-06-03

The 2026-06-03 optimization pass added three safety changes that should be
treated as part of the current paper workflow:

- Direct full-run invocations are strict by construction.  `run_full_baselines.py`
  now forces `EASYGRAPH_GPU_STRICT_ERRORS=TRUE` for every EGGPU child process,
  disables result-cache hits, and sets `EASYGRAPH_GPU_SCC_HOST_ENABLE=FALSE`,
  `EASYGRAPH_GPU_KCORE_HOST_ENABLE=FALSE`, and
  `EASYGRAPH_GPU_SSSP_HOST_ENABLE=FALSE`.  Non-EGGPU EasyGraph baselines set
  `EASYGRAPH_ENABLE_GPU=FALSE`, clear `EASYGRAPH_GPU_BACKEND`, and disable the
  same host-policy flags.
- `library_baselines.py` applies the same mode separation internally, so a
  direct `--backend easygraph-cpu` or `--backend easygraph-cpp` run cannot
  inherit an outer-shell EGGPU backend setting.
- The full runner now checks GPU idleness before every EGGPU child process, not
  only at stage start.  If the selected card is occupied, the row is recorded as
  failed with a `gpu_busy_before_eggpu_child` note.  The full audit then rejects
  the result instead of letting a contended timing row enter paper tables.

The build path was also tightened.  `scripts/build_eggpu.sh` now passes
`-DCMAKE_CUDA_ARCHITECTURES=${EGGPU_CUDA_ARCHITECTURES:-80}`.  This pins the
original A100 build to `sm_80` and avoids driver/toolchain-dependent PTX
fallback.  New machines should set the variable to the local GPU compute
capability, for example `89` for RTX 4090.

The Closeness CUDA path was optimized and made easier to audit:

- unweighted Closeness now uses an exact source-parallel BFS kernel over cached
  device CSR instead of paying the weighted Dijkstra-style path;
- device CSR and per-source workspaces are reused through the existing device
  graph/buffer cache utilities;
- kernel timing stops before copying the result vector back to host, so
  `kernel` excludes output transfer while `e2e` still includes transfer and
  Python-compatible wrapping.

## Follow-up Hardening, 2026-06-04

The 2026-06-04 optimization pass added the following auditable changes:

- Exact all-source Closeness now has a symmetric scale guard controlled by
  `EGGPU_CLOSENESS_EXACT_MAX_NODES` and defaulting to `1,000,000` nodes.  Rows
  skipped by this guard carry the note
  `exact all-source Closeness skipped by symmetric scale guard`; the main audit
  and backend-separation audit accept only that explicit skip form.  This avoids
  timing all exact baselines as timeouts on graphs where the experiment is not
  part of the intended exact-Closeness scope.
- BC has a metadata-tracked warp override, `EASYGRAPH_GPU_BC_WARP_SIZE`.  The
  official script passes `AUTO`, which leaves the code-defined adaptive policy
  active; explicit numeric values are for sweeps and ablations.
- KCore GPU return normalization now keeps the C++ dense `int` result as int32
  when the length already matches the node count.  This reduces timed Python
  copy/upcast overhead without changing correctness-detail normalization, which
  still happens after the timed call.
- KCore's default single-block gate was tightened after targeted tests on
  `ca-CondMat`: high maximum degree alone no longer selects the single-block
  path on medium low-average-degree graphs.  This keeps the fast path for
  small/high-average-degree cases while avoiding a graph-regime-specific
  slowdown.
- Official scripts now pass
  `EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE=AUTO` instead of the older
  `1000` override, so the CUDA-side graph-aware default is not accidentally
  bypassed during full evaluation or ablation.
- EasyGraph-mode timing is stricter: EGGPU warmup exceptions now fail the row
  instead of silently continuing with a cold timed call; PageRank no longer
  retries lower `alpha` values; KCore no longer retries through a directed
  fallback path after an exception.  The preflight static contract checks these
  conditions before full evaluation.
- The final summary now emits a fixed `paper-core` view.  It is a reproducible
  reporting filter for small/low-work and component-output-dominated regimes,
  not a replacement for the full and gpu-friendly views.
- The final summary also makes the `0.05%` SOTA timing-tie tolerance explicit
  and emits a `2%` near-miss table.  Near-miss rows are diagnostic only and do
  not change the SOTA verdict.
- For memory reporting, process-tree GPU memory metrics should be preferred
  when available.  Whole-device NVML memory remains a contamination diagnostic
  and is still useful for catching external GPU use.

Operational note from this review: the long full run started on 2026-06-03 at
`full_eval_gpu0_20260603_164405_selfloop_filtered_scc_trim16_kcore_threshold_constraint_auto`
was later overlapped by external `sglang` processes occupying both GPUs.  Any
EGGPU rows launched after that contention began are diagnostic only and must not
be used as final paper timing evidence.  A clean 16-function rerun on an idle
GPU is still required.

## Remaining Required Evidence

The current goal is not complete until a new full run passes:

- full correctness audit,
- backend separation audit,
- pair-level SOTA summary,
- ablation run.
- `run_metadata.json` inspection for the final result directory.

The rerun should use an idle GPU and must not set `EGGPU_ALLOW_BUSY_GPU=1` for
paper-quality numbers.
