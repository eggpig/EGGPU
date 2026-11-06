# EGGPU artifact

This artifact contains the graph runtime, input tools, author-baseline adapters, experiment runners, and figure exporters. Its source inventory is verified by `tools/verify.py`.

## Build

Use an existing Python environment with NumPy, SciPy, pybind11, networkx, and CMake. Tests also require pytest; figure export requires Matplotlib. Supply a compatible CUDA toolkit and select its compute architecture explicitly.

```bash
python -B artifact/icde/tools/verify.py
python -B artifact/icde/tools/build.py \
  --build-dir /path/to/new-build --cuda-root /path/to/cuda \
  --architecture 89 --arena --execute
```

The build receipt records the source, command, and generated libraries. Build output belongs outside the checkout. `--arena` includes the libraries used by the single-GPU path. Add `--nccl-include /path/to/include --nccl-library /path/to/libnccl.so` when using NCCL.

After building, use the explicit extension and receipt paths:

```bash
python -B artifact/icde/tools/reproduce.py run-native-example \
  --native-dso /path/to/build/cpp_easygraph.so \
  --build-receipt /path/to/build/portable_build_receipt.json \
  --manifest /path/to/graph-manifest.json \
  --devices 0,1 --function sssp --source 0 \
  --communication host_staging --output /path/to/new-result
```

Use the exact extension filename produced by the build. Device IDs are logical IDs after `CUDA_VISIBLE_DEVICES` mapping. The example returns a complete host vector, retains its ownership through session close, and saves the timing and execution metrics.

## Inputs

The seven fixed graphs are US (DIMACS10/road_usa), EP (SNAP soc-Epinions1), LJ (SNAP soc-LiveJournal1), OR (Network Repository soc-orkut), IN (LAW indochina-2004), UK (LAW uk-2002), and TW (LAW twitter-2010). [datasets.json](datasets.json) records exact source URLs, direction, normalization, weights, and fixed-source selection.

Normalization preserves the vertex universe, removes self-loops and duplicate ordered arcs, sorts CSR rows, and stores both orientations for undirected graphs. SSSP uses deterministic integer weights in [1,16]. Keep the normalization receipt and identifier map with each prepared input.

```bash
python -B artifact/icde/tools/reproduce.py build-input-tools --help
python -B artifact/icde/tools/reproduce.py fetch-inputs --help
python -B artifact/icde/tools/reproduce.py prepare-inputs --help
python -B artifact/icde/tools/reproduce.py capture-local-setup --help
```

## Experiments

| Experiment | Workload | Entry points |
|---|---|---|
| E1 | Seven graphs, BFS/SSSP/WCC/PageRank, one GPU | `run-sg`, `run-sg-authors`, `run-c2graph-p` |
| E2 | Same-resource multi-GPU comparison and fixed-backend 1/2/4/8-GPU scaling | `run-native`, `run-gum`, `run-cugraph-mg` |
| E3 | Weight updates, function dependencies, and old-version holders | `state-inputs`, `state-grid`, `state-plan`, `state-worker` |
| E4 | Graph switching under a per-device memory budget; multi-source SSSP | `state-plan`, `state-worker`, `run-e4s` |
| E5 | Matched D-Maintain/C-Retain four-arm intervention | `state-plan`, `state-worker` |
| E6 | Structural-hole screening, Closeness ranking, BFS and WCC | `e6-protocol-plan`, `e6-protocol-worker` |

Invoke an entry with `python -B artifact/icde/tools/reproduce.py ENTRY --help`. Dataset paths, built libraries, local environment, CPU affinity, GPU placement, and output paths are explicit runner arguments. [campaign/latest.json](campaign/latest.json) identifies the selected implementations. The [baseline guide](baselines/README_CN.md) records author revisions, supported algorithms, and compatibility changes.

Partitioned multi-GPU execution supports BFS, SSSP, WCC, and PageRank. Extended application functions use replicated graph state with ego-vertex or source work distributed across devices. D-Maintain's native sparse weight-update and copy-on-write path uses FP64 weights. C-Retain scores historical demand against actually releasable physical storage.

For E3–E5, `state-deployment-template` creates the local resource template. `state-grid` builds shared immutable graph bases and predetermined updates. `state-plan` binds these inputs to the runtime, numerical checks, budgets, and policy settings. All policies in a matched experiment use the same source, input sequence, resource assignment, and clock.

For E6, prepare the independent reference for each exact graph before timed runs. `e6-protocol-worker --help` exposes the reference and workflow modes. The workflow computes all structural scores, selects 32 candidates, computes exact Wasserman–Faust Closeness, selects the BFS source, and returns complete BFS/WCC results. CPU igraph and a cuGraph/CuPy composition provide full-workflow comparisons and operator replacements.

## Measurement

Execution time starts with prepared graph state and includes device synchronization and a complete, independently owned host result. First-task time additionally includes backend-specific preparation from the common host CSR. Continuous-workload timing includes the updates, queries, preparation, and state management inside the specified request sequence. Independent reference calculation and output archival are outside the measured interval. Job wall time is recorded separately.

BFS/SSSP compare complete distance vectors; WCC compares complete partitions. PageRank checks fixed-point residual, probability mass, and agreement with an independent reference. Static BFS/SSSP observations average three fixed sources. The first pass records one observation per configuration, followed by a second repetition. Preserve raw observations and failure states when exporting tables.

Use `first-pass-collect`, `first-pass-build`, and `first-pass-render` for complete available observations. The `paper-current-*` and `paper-e2-e6` commands implement the explicitly selected repeated-result protocols; their input selection must match the requested repetition count. [first_pass/README_CN.md](first_pass/README_CN.md) describes the collector schema.

## Checks

```bash
python -B artifact/icde/tools/test_cpu.py --work /path/to/new-cpu-check
python -B artifact/icde/tools/test_projection_cpu.py \
  --groups native_load_p1,sg_current,state_worker \
  --work /path/to/new-runner-check
```

After a local build, pass `--native-dso` to `test_cpu.py` to include the compiled CPU cases. `test-native-arrays-cpu` checks file and array imports against the local build receipt. Run the initial small GPU check once for each new source/build/resource configuration, then retain complete numerical output checks during measurements.
