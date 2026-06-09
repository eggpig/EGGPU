# EG_Evaluation

This directory contains the paper-facing benchmark, audit, ablation, dataset,
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
full-result audit, backend-separation audit, pair-level SOTA summaries, final
summary generation, and ablations.  Use an idle GPU and keep
`EGGPU_ALLOW_BUSY_GPU`, `RUN_PREFLIGHT=FALSE`, and `EGGPU_USE_CONDA_RUN=TRUE`
for debug-only runs.

Current official defaults include EGGPU warmup of two calls, baseline warmup of
zero, `EGGPU_GPU_VISIBILITY_MARKER=FALSE`,
`EGGPU_GPU_VISIBILITY_MARKER_MB=0`,
`EGGPU_CLOSENESS_EXACT_MAX_NODES=1000000`,
`EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS=1024`,
`EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE=AUTO`, and
`EASYGRAPH_GPU_BC_WARP_SIZE=AUTO`.  KCore's default single-block gate is
graph-aware and does not select the single-block path from high maximum degree
alone.  The final summary reports `full`, `nodes>=10000`, `gpu-friendly`, and
reproducible `paper-core` SOTA views, with a documented `0.05%` timing-tie
tolerance and a `2%` near-miss table for targeted reruns.

For shared-server reservation visibility, you may set
`EGGPU_GPU_VISIBILITY_MARKER=TRUE` and
`EGGPU_GPU_VISIBILITY_MARKER_MB=<N>` to make the long-lived runner visible in
`nvidia-smi`/`nvtop` with a fixed allocation.  If enabled and no marker size is
provided, the runner uses `256` MiB.  The default
`EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB=AUTO` measures the whole-device increment
created by the marker process, including CUDA context overhead, records it in
`run_metadata.json`, and subtracts it only from whole-device absolute memory
metrics; process-tree and delta memory metrics remain unadjusted.

Current reference notes:

- `BASELINE_SUPPORT_MATRIX_20260525.md`: supported baselines, unsupported
  functions, Closeness semantic alignment, and timing/memory definitions.
- `EGGPU_SECURITY_EVAL_AUDIT_20260602.md`: strict-error, backend separation,
  metadata/correctness gates, GPU-idle policy, and remaining required evidence.
- `EGGPU_LITERATURE_MAPPING_20260602.md`: related-work mapping and claims that
  are safe or unsafe for the paper.
- `BASELINE_VERSIONS_20260609.md`: pinned Python/RAPIDS/baseline versions and
  CUDA compatibility boundary for full paper reproduction.
- `datasets/DATASETS.md` and `datasets/MANIFEST_20260609.tsv`: processed
  dataset policy, sizes, checksums, and source hints.
- `REPRODUCE_ON_NEW_GPU_20260531.md`: clone/build/run instructions for another
  GPU machine.
- `EGGPU_CCFA_REPORT_20260603.md`: current chapter-3/chapter-4 style summary
  and next-run checklist.

Older dated notes in this directory are archival debugging/history records.
They may mention local absolute paths or superseded result directories; do not
use them as clone-from-Git reproduction instructions unless they are explicitly
referenced by one of the current notes above.

Generated result directories under `benchmarking/results/` are intentionally
not versioned.  A paper-quality result must include `run_metadata.json`,
correctness validation, backend-separation evidence, and final summaries from a
clean idle-GPU run.
