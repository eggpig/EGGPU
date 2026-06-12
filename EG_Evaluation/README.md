# EG_Evaluation

This directory contains the reproducible benchmark, audit, ablation, dataset,
and reporting workflow for EGGPU.

Current benchmark scope:

```text
PageRank, MST, LCC, WCC, SCC, BFS, Dijkstra, BellmanFord, SSSP,
KCore, BC, Closeness, EffectiveSize, Efficiency, Constraint, Hierarchy
```

Primary entry point:

```bash
bash run_main_and_ablation.sh
```

The official workflow runs preflight first, then the full baseline benchmark,
full-result audit, backend-separation audit, pair-level SOTA summaries, the
large-graph Closeness supplement, final summary generation, and ablations.  Use
an idle GPU and keep
`EGGPU_ALLOW_BUSY_GPU`, `RUN_PREFLIGHT=FALSE`, and `EGGPU_USE_CONDA_RUN=TRUE`
for debug-only runs.

Current official defaults include EGGPU warmup of two calls, baseline warmup of
zero, `EGGPU_GPU_VISIBILITY_MARKER=FALSE`,
`EGGPU_GPU_VISIBILITY_MARKER_MB=0`,
`EGGPU_CLOSENESS_EXACT_MAX_NODES=1000000`,
`EGGPU_CLOSENESS_EXACT_MAX_WORK=50000000000`,
`RUN_CLOSENESS_LARGE_SUPPLEMENT=TRUE`,
`CLOSENESS_LARGE_SOURCES=16`,
`CLOSENESS_LARGE_TIMEOUT=1800`,
`EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS=1024`,
`EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE=AUTO`, and
`EASYGRAPH_GPU_BC_WARP_SIZE=AUTO`.  Unweighted `BC` and unweighted
`Closeness` use BFS-based CUDA paths by default; set
`EASYGRAPH_GPU_BC_UNWEIGHTED_BFS=FALSE` or
`EASYGRAPH_GPU_CLOSENESS_UNWEIGHTED_BFS=FALSE` only for regression A/B runs.
KCore's default single-block gate is
graph-aware and does not select the single-block path from high maximum degree
alone.  Exact all-node Closeness remains the main 270-pair semantic; datasets
skipped by the symmetric exact node/work scale guard are filled by an
automatically generated `sampled_target_exact` supplement and merged matrix
with explicit `semantic`, `skip_reason`, `source_policy`, and
`source_nodes_sha` columns.  Supplement rows are exact only for the
deterministic sampled target vertices and must not be counted as all-node exact
Closeness.  The final summary reports `full`, `nodes>=10000`,
`gpu-friendly`, and reproducible filtered SOTA views, with a documented
`0.05%` timing-tie tolerance and near-miss diagnostics for targeted reruns.

For shared-server reservation visibility, you may set
`EGGPU_GPU_VISIBILITY_MARKER=TRUE` and
`EGGPU_GPU_VISIBILITY_MARKER_MB=<N>` to make the long-lived runner visible in
`nvidia-smi`/`nvtop` with a fixed allocation.  If enabled and no marker size is
provided, the runner uses `256` MiB.  The default
`EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB=AUTO` measures the whole-device increment
created by the marker process, including CUDA context overhead, records it in
`run_metadata.json`, and subtracts it only from whole-device absolute memory
metrics; process-tree and delta memory metrics remain unadjusted.

Public reference files:

- `datasets/DATASETS.md` and `datasets/MANIFEST_20260609.tsv`: processed
  dataset policy, sizes, checksums, and source hints.
- `environment_eggpu_cuda128.yml`: conda environment specification.
- `benchmarking/*.py`: benchmark runners, correctness validation, audit, and
  summary generation scripts.

Generated result directories under `benchmarking/results/` are intentionally
not versioned.  A paper-quality result must include `run_metadata.json`,
correctness validation, backend-separation evidence, and final summaries from a
clean idle-GPU run.
