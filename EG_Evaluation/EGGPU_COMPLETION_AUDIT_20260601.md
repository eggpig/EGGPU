# EGGPU Completion Audit 2026-06-01

## 2026-06-03 Update

This audit is now superseded for final paper claims because `Closeness` has
been integrated after the 2026-06-01 full run.  The current benchmark scope is
16 functions:

```text
BC, BFS, BellmanFord, Closeness, Constraint, Dijkstra, EffectiveSize,
Efficiency, Hierarchy, KCore, LCC, MST, PageRank, SCC, SSSP, WCC
```

Current non-GPU evidence:

- The 16-function registry is consistent across the main full runner, library
  runner, and ablation runner.
- `benchmarking/preflight_closeness_semantics.py` passes for EasyGraph C++,
  NetworkX, and igraph on directed-outward, source-subset, and disconnected
  undirected cases.
- The official runner keeps `EGGPU_GPU_VISIBILITY_MARKER` disabled by default
  so paper memory metrics are not polluted by an idle CUDA context.
- The 2026-06-03 follow-up optimization adds an unweighted exact BFS CUDA path
  for Closeness, device CSR/workspace reuse, strict direct-run EGGPU child
  environment, disabled host-policy flags for SCC/KCore/SSSP, and per-EGGPU
  child GPU-idle checks.

Remaining blocker for closing the current goal:

- A fresh 16-function full benchmark and ablation run must be generated on an
  idle GPU after rebuilding `cpp_easygraph` with the current source and a
  machine-appropriate `EGGPU_CUDA_ARCHITECTURES` value.  Partial local evidence
  for the new Closeness path is not a replacement for this clean full rerun.

## Historical 15-Function Verdict

The latest clean main benchmark is valid for correctness, backend separation, and pair-level SOTA analysis.  The goal is not closed yet because the ablation runner was corrected after the latest ablation run, so the ablation evidence must be regenerated before final paper claims.

Latest clean main result:

```text
benchmarking/results/full_eval_gpu1_20260601_005630_selfloop_filtered_scc_trim16_kcore_threshold_constraint_auto
```

Latest ablation result, superseded for final ablation claims after the script fix:

```text
benchmarking/results/ablation_gpu1_20260601_005630_selfloop_filtered_scc_trim16_kcore_threshold_constraint_auto
```

## Correctness And Coverage

Audit summary from the latest main result:

| Item | Value |
|---|---:|
| gate_status | pass |
| datasets | 17 |
| functions | 15 |
| EGGPU e2e rows | 255 |
| EGGPU runtime bad rows | 0 |
| EGGPU validation bad rows | 0 |
| coverage issues | 0 |

The 15 EGGPU functions covered by this run are:

```text
BC, BFS, BellmanFord, Constraint, Dijkstra, EffectiveSize, Efficiency,
Hierarchy, KCore, LCC, MST, PageRank, SCC, SSSP, WCC
```

The correctness validator reports 42 non-EGGPU failures.  These are baseline semantic/numeric mismatches, mainly `nx-cugraph / BC`, `igraph / PageRank`, and tiny floating-point differences in `igraph / Constraint`.  They do not involve EGGPU rows.

## Backend Separation And Warmup

Backend separation audit passed with zero failures.

Expected and verified contract:

| Baseline | Mode | GPU env | Warmup |
|---|---|---|---:|
| EGGPU | gpu | `EASYGRAPH_ENABLE_GPU=TRUE`, `EASYGRAPH_GPU_BACKEND=mine` | 2 |
| easygraph-cpu | cpu | `EASYGRAPH_ENABLE_GPU=FALSE`, empty GPU backend | 0 |
| easygraph-cpp | cpp | `EASYGRAPH_ENABLE_GPU=FALSE`, empty GPU backend | 0 |

Important timing boundary:

- `build`: baseline-native graph construction after raw edge-list parsing and imports are excluded.
- `e2e`: user-visible function call after graph construction, including per-call preparation, transfer, synchronization, and result wrapping.
- `kernel`: exact CUDA event time where exposed.  For CPU baselines, kernel equals algorithm wall time.  After the 2026-06-01 code update, structural-hole EGGPU dense paths (`Constraint`, `EffectiveSize`, `Hierarchy`, and `Efficiency` through `EffectiveSize`) also report CUDA event time around the GPU kernels rather than C++ wrapper wall time.  The latest clean main result predates this code update, so final structural-hole kernel tables should be regenerated.

EGGPU may construct or reuse an EasyGraph C++ graph object internally for GPU bindings.  This is part of EGGPU's own integrated path and is included in EGGPU e2e when paid by the user-facing call.  The `easygraph-cpp` baseline is measured separately with GPU disabled; structural-hole easygraph-cpp rows are skipped because those bindings route to CUDA in this GPU-enabled build, which would otherwise contaminate the CPU C++ baseline.

## Pair-Level SOTA Summary

Full 17-dataset result:

| Metric | EGGPU SOTA pairs | Total pairs | Rate |
|---|---:|---:|---:|
| E2E | 222 | 255 | 87.1% |
| Kernel | 241 | 255 | 94.5% |

Filtered result with `nodes >= 10000`:

| Metric | EGGPU SOTA pairs | Total pairs | Rate |
|---|---:|---:|---:|
| E2E | 161 | 180 | 89.4% |
| Kernel | 171 | 180 | 95.0% |

By category on the full result:

| Category | Functions | E2E SOTA | Kernel SOTA | Interpretation |
|---|---|---:|---:|---|
| Path | BFS, SSSP, Dijkstra, BellmanFord, MST | 81/85 | 85/85 | Strong; remaining E2E misses are mostly small graph launch/return overhead. |
| Centrality | PageRank, BC, LCC | 50/51 | 51/51 | Strong; one E2E miss is `web-NotreDame / BC`. |
| Connectivity/Core | WCC, SCC, KCore | 27/51 | 38/51 | Current weak category, dominated by SCC and small/medium KCore. |
| Structural holes | Constraint, EffectiveSize, Efficiency, Hierarchy | 64/68 | 67/68 | Strong overall, but `Constraint / ca-HepPh` is a policy regression to investigate. |

This satisfies the looser target that EGGPU is SOTA on the great majority of function-dataset pairs, especially for kernel time.  It does not satisfy the stricter target that every category average is SOTA because Connectivity/Core remains weak.

## GPU-Unfriendly Graph Regimes

The main regimes where GPU E2E is naturally weak are:

1. Small graphs where launch, synchronization, and Python result wrapping dominate: `ca-GrQc`, `p2p-Gnutella08`, `wiki-Vote`, `LastFM`, `ca-HepTh`.
2. Very sparse directed graphs with many singleton SCCs: `wiki-Talk`, `email-EuAll`, `web-NotreDame` for SCC-like workloads.
3. Low-constant CPU kernels in igraph/easygraph-cpp for KCore and SCC on small/medium graphs.

SCC structure evidence from the latest result:

| Dataset | SCC count | Singleton SCCs | Largest SCC | Reason it is hard |
|---|---:|---:|---:|---|
| wiki-Talk | 2,281,879 | 2,281,311 | 111,881 | Enormous singleton output and fragmented directed reachability. |
| web-NotreDame | 203,609 | 202,462 | 53,968 | Many small SCCs plus directed skew. |
| soc-Epinions1 | 42,176 | 41,112 | 32,223 | Many singleton SCCs and skewed in/out degree. |
| ER-100k | 9 | 8 | 99,992 | GPU-friendly for SCC; EGGPU is SOTA here. |

## Optimization Effects

Compared with the previous `constraint_intersection` run, the latest main result has a geometric mean speedup of about 1.106x on E2E and 1.148x on kernel across matched EGGPU rows.  Useful changes include:

- SCC trim/pivot/epoch/set-return changes improved `wiki-Vote`, `ER-100k`, `email-EuAll`, and several directed cases.
- Path functions are now stable and broadly SOTA after removing unsafe host fallback behavior.
- KCore is no longer globally broken; large graphs are mostly SOTA, while small/medium misses remain low-constant CPU cases.
- Structural-hole functions are generally strong, but the current `Constraint` AUTO policy has mixed behavior.

Constraint policy warning:

- `Constraint / ca-HepPh` regressed from 0.0318s kernel in the prior run to 0.4087s in the latest run while producing the same correctness detail hash.
- Several directed structural-hole cases also regressed, while `email-Enron`, `com-youtube`, `email-EuAll`, `ca-GrQc`, and others improved.
- The current policy is therefore not final.  It should be tuned or narrowed before claiming the adaptive policy is universally beneficial.

## Ablation Status

The latest ablation result contains 77,462 rows, with 77,449 ok rows and 13 timeout rows.  Timeouts are concentrated in large workflow/return cases, especially `wiki-Talk` and no-cache variants.

However, the ablation runner and structural-hole kernel timing have now been corrected after that run:

- `full` now inherits the same policy defaults as the main experiment.
- `EASYGRAPH_GPU_COMPONENT_DENSE_RETURN` is no longer forced to `TRUE` in ablation full mode.
- workflow ablation now runs `no_adaptive_policy` instead of a redundant `adaptive_policy` variant.
- structural-hole GPU dense functions now pass back CUDA event kernel time instead of wrapper wall time.

Therefore, the latest main result is valid for correctness/backend separation and E2E trends, but final ablation figures and structural-hole kernel numbers should be regenerated.

## Literature Mapping

| Work | Venue | CCF status | Where to use | How it informs EGGPU |
|---|---|---|---|---|
| HyTGraph: GPU-Accelerated Graph Processing with Hybrid Transfer Management | ICDE 2023 | CCF A | Related Work; Methodology data movement | Supports the GraphContext, graph-cache, transfer-reuse, and workflow-reuse story. |
| A GPU Algorithm for Detecting Strongly Connected Components / ECL-SCC | SC 2023 | CCF A | Related Work; SCC Methodology; Limitations | Shows why SCC needs GPU-friendly propagation/edge-centric designs, not DFS-style CPU logic.  It motivates our SCC trim, frontier, pivot, and epoch/tag optimizations, while also explaining remaining SCC hard cases. |
| Accelerating k-Core Decomposition by a GPU | ICDE 2023 | CCF A | Related Work; KCore Methodology | Shows that high-performance GPU KCore depends on optimized peeling.  It supports our decision to disable unsafe CPU fallback and keep KCore on the GPU path, while acknowledging small-graph CPU constant-factor losses. |
| Gunrock | PPoPP 2016 / TOPC lineage | CCF A-related systems lineage | Baselines; Related Work | Provides the frontier/advance/filter design background for traversal and path workloads, and explains why Gunrock is a strong baseline for BFS/SSSP/KCore/BC. |
| GraphBLAST | ACM TOMS | CCF A journal | Related Work; Methodology graph representation | Supports CSR/sparse-linear-algebra graph processing, load balancing, and sparsity-aware graph primitives. |

Source links checked:

- HyTGraph DBLP: https://dblp.org/rec/conf/icde/Wang0ZC023
- ECL-SCC SC 2023: https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap326.html
- ECL-SCC project page: https://userweb.cs.txstate.edu/~burtscher/research/ECL-SCC/
- GPU k-core ORNL page: https://impact.ornl.gov/en/publications/accelerating-k-core-decomposition-by-a-gpu
- GraphBLAST DBLP: https://dblp.org/rec/journals/toms/YangBO22
- GraphBLAST GitHub: https://github.com/gunrock/graphblast

## Next Required Run

Regenerate the main and ablation artifacts after the ablation runner and structural-hole kernel-timing fixes.  Use one idle GPU sequentially:

```bash
cd EGGPU/EG_Evaluation && \
RUN_LOG="benchmarking/results/main_then_ablation_$(date +%Y%m%d_%H%M%S).console.log" && \
MAIN_GPU=<IDLE_GPU> ABL_GPU=<IDLE_GPU> LIBRARY_TIMEOUT=100 ABLATION_TIMEOUT=300 \
bash run_main_and_ablation.sh |& tee "${RUN_LOG}"
```

Use two idle GPUs if desired:

```bash
cd EGGPU/EG_Evaluation && \
RUN_LOG="benchmarking/results/main_then_ablation_$(date +%Y%m%d_%H%M%S).console.log" && \
MAIN_GPU=<IDLE_GPU_A> ABL_GPU=<IDLE_GPU_B> RUN_PARALLEL=1 LIBRARY_TIMEOUT=100 ABLATION_TIMEOUT=300 \
bash run_main_and_ablation.sh |& tee "${RUN_LOG}"
```

Do not set `EGGPU_ALLOW_BUSY_GPU=1` for paper-quality results.
