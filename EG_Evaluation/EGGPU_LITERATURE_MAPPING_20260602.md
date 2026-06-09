# EGGPU Literature and Method Mapping, 2026-06-02

This note records the papers and systems that should be cited when writing the
EGGPU related-work and methodology sections.  It separates direct engineering
influence from baseline/context references.  The current benchmark status is
still not final because the 2026-06-02 full run has four EGGPU timeout rows;
use this document as method support, not as proof of final performance.

## Source Verification

Checked on 2026-06-02 and refreshed on 2026-06-03/2026-06-04 with primary or
near-primary sources where possible:

- ECL-SCC, SC 2023 technical paper page:
  https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap326.html
- ECL-SCC project/code page:
  https://userweb.cs.txstate.edu/~burtscher/research/ECL-SCC/
- Accelerating k-Core Decomposition by a GPU, ICDE 2023:
  https://impact.ornl.gov/en/publications/accelerating-k-core-decomposition-by-a-gpu/
- HyTGraph, ICDE 2023:
  https://arxiv.org/abs/2208.14935
- Gunrock documentation and paper metadata:
  https://gunrock.github.io/gunrock/gunrock.wiki/Overview.html
  https://dblp.org/rec/conf/ppopp/WangDPWRO16
- GraphBLAST repository and GraphBLAST article/news page:
  https://github.com/gunrock/graphblast
  https://cs.lbl.gov/news-and-events/news/2022/graphblast-targets-gpu-graph-analytics-performance-issues/
- RAPIDS cuGraph algorithm coverage:
  https://docs.rapids.ai/api/cugraph/stable/graph_support/algorithms/
- RAPIDS nx-cugraph supported algorithm list:
  https://docs.rapids.ai/api/cugraph/nightly/nx_cugraph/supported-algorithms/
- RAPIDS cuGraph Python API reference:
  https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph/
- RAPIDS cuGraph 26.04 stable API and nx-cugraph 26.06 nightly pages were
  rechecked on 2026-06-03.  The nx-cugraph Centrality support list includes
  betweenness, degree, edge betweenness, eigenvector, in-degree, Katz, and
  out-degree centrality, but not Closeness.  The cuGraph 26.04 Python API
  Centrality section lists betweenness, edge betweenness, Katz, degree, and
  eigenvector centrality, but not Closeness:
  https://docs.rapids.ai/api/cugraph/nightly/nx_cugraph/supported-algorithms/
  https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph/
- NetworkX closeness centrality semantics:
  https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
- python-igraph closeness API semantics:
  https://igraph.org/python/versions/0.10.0/api/igraph.GraphBase.html
- iBFS: Concurrent Breadth-First Search on GPUs, SIGMOD 2016:
  https://www2.seas.gwu.edu/~howie/publications/iBFS-SIGMOD16.pdf
  https://www.researchwithrutgers.com/en/publications/ibfs-concurrent-breadth-first-search-on-gpus
- Regularizing graph centrality computations, JPDC 2015:
  https://sariyuce.com/papers/jpdc15.pdf
- TigerGraph Closeness Centrality documentation, useful only as production
  context for multi-source BFS framing:
  https://docs.tigergraph.com/graph-ml/3.10/centrality-algorithms/closeness-centrality
- The parallel computing of node centrality based on GPU, useful only as older
  non-CCF-A context:
  https://www.aimspress.com/article/doi/10.3934/mbe.2022123
- NetworkX core-number backend caveat:
  https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.core.core_number.html
- PPoPP 2024 graph-processing session page for INFINEL and GraphCube:
  https://ppopp24.sigplan.org/details/PPoPP-2024-papers/23/GraphCube-Interconnection-Hierarchy-aware-Graph-Processing
- INFINEL PPoPP 2024 paper page:
  https://ppopp24.sigplan.org/details/PPoPP-2024-papers/17/INFINEL-An-efficient-GPU-based-processing-method-for-unpredictable-large-output-grap
- PICO, 2024 arXiv:
  https://arxiv.org/abs/2402.15253
- Node Centrality Approximation for Large Networks Based on Inductive Graph
  Neural Networks, 2024 arXiv:
  https://arxiv.org/abs/2403.04977
- Using Time-Aware Graph Neural Networks to Predict Temporal Centralities in
  Dynamic Graphs, NeurIPS 2024 poster:
  https://openreview.net/forum?id=6n709MszkP

For CCF class, use the current official CCF catalogue used by the target venue
or school before final submission.  The classifications below follow the common
systems/database convention: SC, ICDE, PPoPP, SIGMOD, VLDB/PVLDB are treated as
CCF-A venues.  arXiv-only work is not counted as CCF-A unless it later appears
in a ranked venue.

## Core Papers and How EGGPU Uses Them

| Work | Venue / Status | CCF-A role | EGGPU use | Paper section |
|---|---|---|---|---|
| HyTGraph: GPU-Accelerated Graph Processing with Hybrid Transfer Management | ICDE 2023 | Yes | Motivates the middle-layer idea that CPU-GPU transfer strategy matters as much as kernel speed.  EGGPU does not copy HyTGraph's per-iteration hybrid scheduler; instead it applies the system insight to GraphContext reuse, device CSR cache, workspace reuse, pinned host buffers, and workflow-reuse ablation. | Related Work: GPU graph systems; Methodology: middle layer and data movement; Ablation: graph/context reuse. |
| A GPU Algorithm for Detecting Strongly Connected Components / ECL-SCC | SC 2023 | Yes | Motivates GPU-friendly SCC design: avoid recursive DFS, use data-driven parallel propagation, keep CSR graph representation, and expose why SCC is graph-structure sensitive.  EGGPU's SCC uses active trimming, degree-aware pivoting, frontier expansion, epoch-tagged visited arrays, and conservative return policy.  It does not implement ECL-SCC's maximum-ID propagation exactly. | Related Work: GPU SCC; Methodology: SCC implementation; Limitations: sparse directed graphs with many small SCCs. |
| Accelerating k-Core Decomposition by a GPU | ICDE 2023 | Yes | Confirms that high-performance GPU KCore needs specialized peeling rather than a generic graph-parallel layer.  EGGPU keeps KCore on the GPU path, disables accidental host fallback, uses graph-aware single-block/queue policy, and reports small-graph low-constant CPU losses as a real graph-regime issue. | Related Work: KCore; Methodology: KCore peeling; Discussion: why small graphs and low average degree are hard for GPU. |
| Gunrock | PPoPP 2016 plus system lineage | Yes | Provides the frontier/advance/filter mental model for BFS, SSSP, BC, WCC-style propagation, and source batching.  Gunrock is also a GPU baseline where local binaries are available. | Related Work: GPU graph frameworks; Baselines; Methodology: traversal/path primitives. |
| GraphBLAST | TOMS / GraphBLAS GPU framework | Not counted as CCF-A unless local catalogue says so | Supports CSR/sparse-linear-algebra framing, push-pull/direction-aware sparse operations, load balancing, and automatic GPU memory management as background.  EGGPU uses CSR kernels, not GraphBLAS semirings, so cite it as related context rather than a direct algorithm template. | Related Work: sparse linear algebra graph processing; Methodology motivation for CSR. |
| RAPIDS cuGraph and nx-cugraph | Production GPU graph library | Baseline / ecosystem | Defines an important Python GPU baseline and confirms coverage for PageRank, WCC/SCC, KCore/core number, BFS, SSSP, MST, BC, etc.  cuGraph does not cover EasyGraph structural-hole functions directly, so structural-hole comparisons rely on EasyGraph/igraph semantics. | Baseline section; Experimental setup; Coverage matrix. |
| NetworkX backend documentation | API/reference | Baseline semantics | Documents backend dispatch and caveats such as cuGraph core-number not supporting directed graphs in NetworkX's backend page.  This supports why the benchmark distinguishes NetworkX, nx-cugraph, native cuGraph fallback, and EGGPU. | Baseline support and fairness notes. |
| NetworkX and python-igraph closeness documentation | API/reference | Baseline semantics | Establishes the exact Closeness semantics used in the benchmark.  EasyGraph computes outward distance on directed graphs and uses the Wasserman-Faust disconnected-graph correction.  NetworkX computes inward distance by default, so the benchmark applies a reverse graph view for directed inputs.  igraph supports `mode=OUT`, but its normalized value lacks the NetworkX/EasyGraph disconnected correction, so the benchmark multiplies by the reachable-fraction factor to align semantics. | Baseline support and fairness notes; Methodology: Closeness integration. |
| iBFS: Concurrent Breadth-First Search on GPUs | SIGMOD 2016 | Yes | Provides the strongest CCF-A traversal reference for exact Closeness-like work: Closeness is a repeated shortest-path traversal workload, and iBFS shows that grouping multiple BFS instances in one GPU traversal can exploit shared frontiers and bit-level parallelism.  EGGPU's current Closeness kernel is simpler and exact, using source-parallel CSR traversal rather than iBFS's full joint-frontier design; cite iBFS as a principled direction and future optimization path, not as implemented code. | Related Work: GPU traversal; Methodology/Future Work: Closeness source batching and joint traversal opportunity. |
| Regularizing graph centrality computations | JPDC 2015 | Non-CCF-A journal in common CCF lists | Directly studies exact GPU/accelerator Closeness and argues that one-BFS-at-a-time kernels underutilize GPUs, while simultaneous traversals and vectorization improve centrality workloads.  EGGPU uses the same high-level lesson for all-source Closeness and result/materialization design, but not the paper's exact vectorized formulation. | Related Work: centrality acceleration; Methodology: why Closeness needs source batching and why small graphs remain hard. |
| INFINEL | PPoPP 2024 | Yes | Useful for the "large output graph queries" story: GPU kernels can be fast while output size and materialization dominate end-to-end time.  EGGPU's return-path slimming and lazy/materialization ablation address the same systems pressure, but for EasyGraph API outputs. | Related Work: output-heavy GPU graph processing; Methodology: return path. |
| GraphCube | PPoPP 2024 | Yes | Large-scale distributed graph processing context.  It is not a direct EGGPU design source because EGGPU is single-machine Python/API integration, but it helps position graph-structure-aware load balancing and communication as a broader issue. | Related Work only, not method contribution. |
| PICO: Accelerating All k-Core Paradigms on GPU | arXiv 2024 | No, arXiv-only at time checked | Useful as a forward-looking KCore reference: Peel and Index2core paradigms, reducing atomics/redundant edge work, and explaining why KCore has remaining algorithmic space.  Do not count it as a CCF-A citation unless accepted in a ranked venue. | Future work / KCore limitations. |

## 2026-06-03 Implementation Mapping Update

The latest Closeness implementation should be described as an engineering
application of established traversal and transfer-management principles, not as
a new Closeness-specific algorithm from recent CCF-A literature:

- For unweighted graphs, EGGPU now dispatches Closeness to an exact
  source-parallel BFS kernel over cached device CSR.  This is consistent with
  the all-source traversal framing in Gunrock/iBFS-style GPU graph processing.
- For weighted graphs, EGGPU keeps the weighted Dijkstra-style CUDA path.
- The optimization target is E2E integration: avoiding unnecessary weight
  transfer for unweighted inputs, reusing device CSR/workspaces, and keeping
  kernel timing separate from output copy and Python result wrapping.
- The relevant CCF-A citations are still general traversal/systems references:
  Gunrock for frontier-centric GPU graph processing, iBFS for concurrent BFS,
  HyTGraph for transfer-management pressure, and INFINEL for output/materialize
  pressure.  Do not claim that EGGPU implements iBFS, HyTGraph, or INFINEL.
  They motivate design choices and limitations.

## 2026-06-04 Optimization Mapping Update

The latest optimization pass adds two implementation choices and one reporting
view that should be reflected in the paper:

- BC now uses a BC-specific warp-size policy for large low-average-degree
  directed graphs.  The old shared centrality heuristic chose `warp=8` on
  `web-NotreDame`; targeted sweeps showed `warp=2` is faster on both
  `web-NotreDame` and `wiki-Talk`.  This remains a local load-balancing
  heuristic, not a new BC algorithm.  Cite Gunrock/iBFS for traversal
  parallelism context, and report `EASYGRAPH_GPU_BC_WARP_SIZE` only as an
  implementation knob.
- KCore keeps the ICDE 2023 peeling-paper framing.  EGGPU's practical change is
  narrower: `1024` single-block threads, int32 dense return, and a graph-aware
  single-block gate reduce launch and Python return costs on small/medium KCore
  pairs while avoiding the high-max-degree/low-average-degree slowdown observed
  on `ca-CondMat`.  Do not claim parity with the full ICDE 2023 optimized KCore
  design; `ca-CondMat` remains a hard case.
- The `paper-core` view excludes component-output-dominated WCC/SCC pairs where
  EasyGraph's public API must create Python sets for many components.  INFINEL
  (PPoPP 2024) is the best CCF-A systems reference for the broader lesson that
  GPU graph kernels with unpredictable or large outputs need separate output
  handling.  EGGPU uses this as motivation for return-path slimming and for a
  transparent limitation table, not as an implemented INFINEL-style engine.

## Mapping by EGGPU Function Category

### Ranking and Centrality

Functions: `PageRank`, `BC`, `Closeness`.

- Use Gunrock for the frontier-centric GPU graph-processing lineage.
- Use GraphBLAST for sparse matrix/vector and load-balancing background.
- Use cuGraph/nx-cugraph as production GPU baseline coverage.
- Use NetworkX and igraph documentation to pin down Closeness direction and
  disconnected-graph normalization semantics.
- EGGPU-specific contribution: keep EasyGraph API unchanged while using
  reverse CSR, device cache, source batching, all-source shortest-path
  Closeness, and compact Python-compatible return objects.
- RAPIDS documentation checked on 2026-06-02 does not list
  `closeness_centrality` in the nx-cugraph supported Centrality algorithms, and
  the cuGraph Python API Centrality section lists betweenness, Katz, degree,
  and eigenvector centrality but not closeness.  Therefore Closeness should be
  reported as an EGGPU coverage extension, with NetworkX/igraph as semantic
  CPU references and nx-cugraph/cuGraph marked unsupported where appropriate.
- The CPU-only Closeness preflight in
  `benchmarking/preflight_closeness_semantics.py` should be cited internally as
  the implementation check for baseline semantics: NetworkX is evaluated on a
  reverse directed view to match EasyGraph's outward distance, and igraph's
  normalized output is multiplied by the reachable-fraction
  Wasserman-Faust correction.

Recommended paper placement:

- Related Work: Gunrock, GraphBLAST, cuGraph.
- Methodology: reverse CSR for PageRank; source-batched workspace for BC;
  all-source CSR traversal for Closeness with outward directed semantics.
- Experiments: E2E first, kernel second; BC source count and Closeness
  disconnected-graph formula must be explicitly reported.

Closeness-specific literature finding:

- No strong recent 2024-2026 CCF-A paper was found that directly targets exact
  GPU Closeness Centrality for Python graph APIs.  Most recent CCF-A GPU graph
  systems focus on traversal, transfer management, dynamic graph processing,
  or output-heavy graph queries rather than exact all-node Closeness itself.
- A fresh search on 2026-06-02 using queries such as `2024 GPU closeness
  centrality parallel algorithm paper`, `2025 GPU closeness centrality graph
  centrality algorithm`, and `CCF A GPU closeness centrality parallel graph
  processing 2024` found useful context but not a direct CCF-A Closeness
  algorithm to claim as the template.  The most relevant hits were production
  or API context such as TigerGraph's multi-source BFS description for
  Closeness, RAPIDS/nx-cugraph coverage pages, and an older GPU node-centrality
  paper outside the 2024-2026 CCF-A window.
- A follow-up search on the same date with broader terms such as `2024 GPU
  closeness centrality parallel graph processing exact all nodes closeness
  centrality GPU`, `2025 GPU closeness centrality graph processing paper`, and
  `CCF A GPU closeness centrality ICDE SIGMOD VLDB SC 2024 2025` again did not
  identify a direct recent CCF-A exact-GPU-Closeness paper.  Representative
  hits were GNN/centrality-learning papers, distributed graph-system papers,
  or the older non-CCF-A GPU node-centrality work already listed above.
- A 2026-06-03 refresh with targeted searches over ACM/IEEE/OpenReview/arXiv
  still did not find a recent CCF-A exact GPU Closeness implementation paper.
  The closest current literature around Closeness is approximation or
  prediction oriented, e.g., inductive GNN ranking for CC/BC on arXiv 2024 and
  temporal-centrality prediction in a NeurIPS 2024 graph-learning poster.  These
  are useful related-work context for why exact Closeness remains expensive, but
  they are not direct algorithmic templates for EGGPU's exact EasyGraph API
  implementation.
- The strongest older exact/near-exact GPU Closeness evidence remains the
  pre-2024 centrality line, especially Sariyuce et al.'s JPDC 2015
  "Regularizing graph centrality computations", which reports GPU/vectorized
  Closeness acceleration, and iBFS/SIGMOD 2016, which is not a Closeness paper
  but is a CCF-A repeated-BFS GPU traversal reference.  These support the
  engineering choice of source batching and CSR traversal, but only iBFS should
  be counted as CCF-A and neither should be presented as a recent 2024-2026
  Closeness-specific contribution.
- Therefore, cite Closeness integration as an application of the same
  all-source shortest-path/frontier and data-transfer principles used by
  Gunrock/GraphBLAST/HyTGraph, and cite NetworkX/igraph/cuGraph documentation
  for baseline support and semantic fairness.  Do not overclaim a direct
  Closeness-specific CCF-A algorithmic inheritance.

### Connectivity and Core Structure

Functions: `LCC`, `WCC`, `SCC`, `KCore`.

- Use ECL-SCC for SCC-specific GPU design and limitations.
- Use ICDE 2023 KCore GPU paper for KCore peeling and synchronization costs.
- Use Gunrock as GPU baseline where local executable semantics are aligned.
- Use NetworkX/cuGraph documentation to explain backend coverage and directed
  KCore caveats.

Recommended paper placement:

- Related Work: SCC and KCore as specialized GPU graph kernels.
- Methodology: WCC label propagation, SCC active trimming/frontier/pivot/epoch
  tags, KCore peeling policy.
- Discussion: small graphs, many singleton SCCs, low average degree, and
  Python set/dict result materialization are real E2E barriers.

### Path and Spanning Structure

Functions: `MST`, `BFS`, `Dijkstra`, `BellmanFord`, `SSSP`.

- Use Gunrock for BFS/SSSP frontier execution and comparison.
- Use cuGraph/nx-cugraph for Python GPU ecosystem coverage.
- Use GraphBLAST for SSSP as a sparse-algebra example; its repository notes
  concise SSSP expression and direction-optimization influence.

Recommended paper placement:

- Related Work: GPU traversal/path frameworks.
- Methodology: unweighted BFS branch, weighted CSR for Dijkstra/Bellman-Ford,
  multi-source batching, MST return compacting.
- Experiments: report source counts for multi-source path algorithms and
  distinguish kernel time from repeated per-source CLI runs in Gunrock.

### Structural Holes

Functions: `EffectiveSize`, `Efficiency`, `Constraint`, `Hierarchy`.

- There is no direct cuGraph/nx-cugraph baseline for Burt structural-hole APIs.
- Use HyTGraph for transfer/cache rationale and INFINEL for output-heavy GPU
  query motivation.
- Use GraphBLAST/Gunrock only as general GPU graph-processing background, not
  as direct structural-hole baselines.
- EGGPU-specific contribution: move Python ego-network/set loops into CSR/CUDA,
  use sorted-neighbor intersection and smaller-neighbor scan for Constraint,
  and keep output semantics aligned with EasyGraph.

Recommended paper placement:

- Related Work: structural-hole functions are under-covered by existing GPU
  Python graph libraries.
- Methodology: local ego-neighborhood scans, CSR intersection, compact return.
- Experiments: structural-hole functions should be emphasized because they are
  a strong E2E story and broaden beyond standard cuGraph coverage.

## What Not to Overclaim

1. Do not say EGGPU implements ECL-SCC.  It borrows the GPU-friendly lesson
   of data-driven propagation and non-recursive parallel SCC, but the kernel
   strategy is different.
2. Do not count PICO as CCF-A unless it has a later accepted venue in the
   citation list used by the thesis/paper.
3. Do not claim GraphBLAST is a direct baseline in this benchmark unless it is
   built and measured.  It is currently related work only.
4. Do not use structural-hole `igraph` rows as a universal correctness oracle.
   EasyGraph semantics are the primary reference for those functions.
5. Do not claim final performance based on the 2026-06-02 full run until the
   four timeout rows are cleared or explicitly reported after the direct-child
   runner fix.

## Current Evidence Gaps

The current workflow is ready for a clean rerun, but the goal is not complete
until an idle-GPU full benchmark proves:

- EGGPU has no runtime timeout/failure rows.
- EGGPU correctness validation remains clean.
- backend separation audit still shows no EGGPU/easygraph-cpp leakage.
- the final summary reaches the desired SOTA target either by most
  function-dataset pairs or by the four category-level aggregates.
