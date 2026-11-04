# ICDE artifact guide

This checkpoint separates the historical source3 base, the current source candidate, and measurements. `provenance/origin.json` preserves the full 1150-file source3 manifest; `omitted.json` records six nontext upstream illustrations/sample assets excluded from publication. `candidate_delta.json` records seven modified and three added Python/test files. `current_source.json` binds all current source bytes; `manifest.json` binds the complete publication.

The source3 original binary has historical qualification. The capability-dispatch patch passed 46 CPU checks; the closure-LRU control passed a separate 10-test CPU suite. Neither CPU suite loaded CUDA or a DSO. A new build, its GPU qualification, and the new performance campaign need separate source-bound receipts. Do not transfer the base binary's qualification to the patched source or claim new results from packaging.

## Implemented execution scope

| Entry | Functions and execution scope |
|---|---|
| Public SINGLE / REQUEST_PARALLEL | pagerank, mst, lcc, wcc, scc, bfs, dijkstra, bellman_ford, sssp, kcore, bc, closeness, effective_size, efficiency, constraint, hierarchy |
| REPLICATED | bc/closeness partition source vertices; four structural-hole functions partition target vertices; every GPU holds a full graph |
| PARTITIONED | BFS, SSSP, WCC, PR; one graph is divided among GPUs |
| Session.native | BFS, SSSP, WCC, PR, weighted PR; immutable snapshots and declared D-Maintain/C-Retain policies |
| Session.arena | Single-GPU BFS, SSSP, WCC, PR, weighted PR; explicit arena budget and journal updates |
| Session.create | capability_v1: one-device static/journal → arena; immutable snapshots or multi-device static → native; multi-device journal rejected |

This is an implementation map, not a claim that every function/resource/input combination has completed current-release qualification. Replication, request parallelism and graph partitioning have different memory and communication costs. The arena takes owned COO inputs and a backing-allocation budget; native execution takes partition manifests and optional per-device retained-state budgets. Session.create does not hide the required input conversion or make these budget definitions equivalent.

D-Maintain reuses valid representations and maintains supported fixed-topology weight versions. Native sparse device patch/COW/lease is FP64-specific. It is not a general topology-update algorithm or incremental maintenance of graph-algorithm answers. C-Retain uses shared ownership and actual releasable allocations; its score is a historical-demand heuristic rather than a measured recovery-time optimizer.

## Explicit reproduction commands

Start with `python -B artifact/icde/tools/reproduce.py status` and `verify`. Commands forwarded by reproduce.py retain their own --help and explicit paths; there is no fabricated run-all command.

| Command | Actual behavior |
|---|---|
| build | Print an external clean-build plan; --execute compiles and records current source/DSO hashes |
| build-input-tools | Print or execute C++17/OpenMP/OpenSSL builds of CSR and deterministic-weight tools |
| test-input-tools | CPU fixtures for normalization, isolated vertices, invalid endpoints, and SHA-defined weights |
| fetch-inputs | Fetch the three missing official sources LJ/US/TW; never run GPU code |
| prepare-inputs | Prepare declared datasets using explicit roots/tools/registry; see inherited input/exporter prerequisites below |
| qualify-sg | Tiny capability-route semantic check; GPU execution requires its explicit --execute-gpu flag and a built runtime |
| run-native-example | Fixed-native complete-output example with explicit build/DSO/input receipts; not the E1 optimal single-GPU comparison |

For example, build CPU input tools outside the checkout:

```sh
python -B artifact/icde/tools/reproduce.py build-input-tools --build-dir /outside/input-tools --cxx c++ --execute
python -B artifact/icde/tools/reproduce.py test-input-tools --tools /outside/input-tools --work /outside/new-input-test
python -B artifact/icde/tools/reproduce.py fetch-inputs --root /outside/data/source
```

prepare-inputs additionally needs --root, --legacy, --tools, --registry artifact/icde/datasets.json and optionally --datasets. EP/OR/IN/UK currently consume the previously prepared canonical layout under the explicit legacy root. TW needs the existing WebGraph Java exporter and jars under that layout. These inherited prerequisites are not bundled in this source checkpoint. The published downloader and preprocessing code are useful components, not yet a clean-machine seven-dataset downloader/converter. The full fresh-input path remains a reproduction task.

## E1–E6 status and comparison boundaries

All six new experiment groups have frozen designs; formal current-release runs, baseline qualification and portable campaign assembly remain pending unless an explicit source-bound result receipt is published in a later release. campaign_status.json contains the machine-readable state.

| Group | Question and comparison | Current publication |
|---|---|---|
| E1 | Same-resource complete tasks; SG and 8-GPU tables separately, strongest applicable qualified external systems | Protocol/capability map; no new timing table |
| E2 | Fixed-backend 1/2/4/8-GPU strong scaling; graph/work/parameters fixed | Protocol; portable formal runner pending |
| E3 | D-Maintain versus Cold/Reuse+Full and compatible persistent/update baselines | Controlled design; new trace runner/results pending |
| E4 | C-Retain under repeated graph visits and budgets; allow native external reuse | Controlled design; KGraph SSSP memoization has a separate matched-task role, not a GPU budget-policy label |
| E5 | Paired D×C four-arm intervention with strong internal alternatives | New-campaign runner/results pending; historical byte interaction is not a new timing result |
| E6 | Complete dependent application and extended-function coverage ablation | Workflow/control design; new application runner/results pending |

Each formal configuration retains ten independent process repetitions. Unsupported/OOM/timeout/quality failure are separate statuses; no fastest-cell source splicing or performance-dependent dataset removal is allowed. The 90% single-GPU target guides optimization and is not a measured conclusion in this checkpoint.

Public [setup summary](setup_public.json) records the observed GPU/CPU/memory/software fields without UUIDs, connection information, private paths or process lists. Every actual run still needs its own resource and numerical checks. Mechanism and framework artwork has [reserved slots](paper/figures/README.md); no figure placeholder carries a performance claim.
