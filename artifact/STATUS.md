# Validation and upstream integration

Checked on 2026-09-22. The implementation is the published `79d74a5` source;
workspace consolidation does not add or change algorithms.

| Check | Result |
| --- | --- |
| Clean CUDA build, A100, sm_80 | 47 compilation/link targets completed |
| EGGPU contract tests | 39 passed; no skips |
| Strict CPU/GPU small-graph comparisons | 16/16 passed |
| Selected CPU graph and algorithm tests | 425 passed, 4 failed |
| Read/write tests | 61 passed, 36 failed, 2 errors, 21 skipped |
| Evaluation protocol tests | 324 passed, 10 failed; 25 subtests passed |
| Entire EasyGraph suite | 853 tests collected; not all executed |

The environment used Python 3.10.20, CUDA 13.1.115, NumPy 2.2.6 and pytest
9.1.1. Read/write skips came from missing `lxml` and `pygraphviz`. No new
large-graph performance measurements were made during consolidation.

## Known integration work

The source is based on EasyGraph's `pybind11` development line around
`be370954` (2026-02-06). The upstream default branch was still `pybind11`, at
`af235e7852dc5bf361d568fc2c22ab90e17f4057`, when checked. That baseline is
44 commits behind. EGGPU uses its own snapshot history, so port changes onto
the current upstream tree instead of replacing it with this snapshot.

Required fixes and decisions:

- Restore graph serialization. Edge-attribute wrappers contain weak references
  that break pickle, including CPU multiprocessing in BC and Closeness.
- Preserve `edge_attr_dict_factory` overrides when wrapping edge attributes.
- Invalidate cached weighted views for every MST output-construction path.
  The alternate fast tree constructor still creates ordinary attribute dicts.
- Preserve MST `algorithm` and `ignore_nan` semantics. The GPU dispatch omits
  those parameters, and its kernel currently maps non-finite weights to 1.0.
- Include `gpu_easygraph` in the source distribution. The current sdist
  contains no GPU sources even though a repository checkout builds successfully.
- Replace removed NumPy aliases in GEXF/GraphML. Thirty-six read/write failures
  under NumPy 2 were caused by `np.float_`. Two additional CPU tests expose
  NOBE empty-graph exception mismatches.
- Resolve upstream C++ binding, MST/path and setup conflicts; run current
  upstream CPU, optional-CUDA and platform CI before proposing a merge.

The ten remaining evaluation test failures are separate from GPU numeric
correctness: nine Gunrock fixtures/assertions predate the strict three-phase
timing contract, and one GraphScope test omits a required cleanup-grace
argument. The support-table test now uses the versioned CSV instead of an
untracked manuscript export, so a source checkout supplies that input.

Ordinary Graph/DiGraph inputs retain EasyGraph node labels and adjacency
structures. `EGGPUBulkGraph` is a separate immutable adapter with contiguous
integer labels and int32 CSR storage, not a replacement for all Graph APIs.
Some GPU results are read-only `Mapping` objects; use `dict(result)` when a
mutable dictionary is required. Return types, projected directed clustering,
GraphC dispatch and fallback behavior require explicit upstream agreement.

## Experimental evidence

Local retention keeps five complete evidence versions: RTX 3080 Ti (August
1–2), A100 V15, V14, V13 (July 30), and V10 (July 29), plus their shared raw
dependencies. Some versions reassemble earlier runs or revise timing
protocols; they are not five independent full reruns.

The final A100 V15 matrix records 208/208 EGGPU cells as executed successfully,
with 180 full passes and 28 sampled passes. Its portable raw table has 15,600
samples in 3,120 groups of five. The RTX evidence has 176 EGGPU cells, but 121
are marked `inconclusive_self_reference`, 44 `sampled_pass`, and 11 `pass`;
complete measurement coverage is not independent correctness certification.

The later Katz work is a review of [upstream PR176](https://github.com/easy-graph/Easy-Graph/pull/176),
not a completed EGGPU main update. The reviewed commit `674edbc` is retained
locally on `archive/easygraph-pr176-katz-20260818` and is not merged into main.
