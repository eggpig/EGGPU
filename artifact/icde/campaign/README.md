# Source-bound campaign tools

All GPU commands require explicit execution flags, the caller's actual GPU UUIDs, a shared exclusive lock, resource checks, and prepared inputs. `run_*` controllers print plans unless explicitly asked to execute. Use a fresh process for every measured configuration and retain every repetition/failure. The owner must also provide the documented process-group deadline.

[`latest.json`](latest.json) points to versioned manifests without copying shared helpers. A source or runner change invalidates earlier qualification. Publicly relocated copies need their own exact source/input/runtime proofs; the historical scope summaries do not bypass that gate.

| Experiment | Entry | Status/boundary |
|---|---|---|
| E1 SG calls | `run_sg.py` / `sg_worker.py` | Complete first and ready outputs; tiny and same-input target proof before ten formal repeats |
| E1 dependent sequence | `run_sequence16.py` / `sequence16_worker.py` | One continuous `[BFS,SSSP,WCC,PR] × 4` trace, source indices `[0,1,2,0]`, zero warmups; all 16 independent full host outputs retained |
| E1 MG / E2 | `run_native.py` / `native_worker.py` | Fixed native backend at 1/2/4/8 GPUs; explicit host-staging communication on the measured hardware |
| External MG | `run_gum.py`, `run_cugraph_mg.py` | Author algorithms retained; complete-output/timing/resource observation still under qualification; no formal claims |
| E3–E5 | `state_worker.py`, `state_inputs.py` | v4 candidate: exact tiny/target/formal gates; native old-reader axis for E3; frozen physical-byte pilot; all full output vectors stored losslessly |
| E6 | `run_e6.py` / `e6_worker.py` | Full ES+Constraint → fixed 32 candidates → exact WF closeness → winner → BFS+WCC; coverage controls and external compositions |

See [state protocol](STATE_V4_PROTOCOL_CN.md), [sequence protocol](SEQUENCE16_PROTOCOL_CN.md), [E6 protocol](E6_PROTOCOL_CN.md), and [external MG protocol](MG_EXTERNAL_PROTOCOL_CN.md). Retain explicit SG/MG extension semantics: replicated ego work and source-sharded closeness are not partitioned native-session scaling.

Use `python artifact/icde/tools/reproduce.py test-campaign-cpu --work /outside/absent-cpu-dir` after installing NumPy/SciPy/pytest. CUDA must not execute in CPU contracts. Use the public `Easy-Graph` source and rebuilt DSO/library paths, a machine-local setup receipt, canonical input root and output/reference stores as explicit arguments. Setup capture output can contain private UUIDs/paths and is not included in publication.

Current qualification is separate from the new formal matrix. State-v4 has 87 CPU checks but needs its new GPU proofs. GUM and cuGraph-MG retain native behavior, including library CPU-affinity narrowing inside the assigned domain, and are not marked completed merely because an earlier observer failed or a CPU test passed. External SG author timing changes remain pending; the full external build/run bundle is not complete.

The figure and table pipeline must consume source-bound formal records only. No timing table or claim of 90% wins is generated from tiny/target qualification. Failure/timeouts and raw outputs are not filtered by performance.

**PR protocol correction:** the packaged SG cuGraph adapter uses global L1 epsilon `1e-9` directly, matching the installed cuGraph implementation. Earlier SG cuGraph PR and cuGraph 16-request timing records remain historical and are not comparable under this corrected protocol. New exact runner qualification is required; EGGPU source checks are unchanged. See [the correction record](../provenance/pr_protocol_correction.json).

## Planning, storage admission, and presentation

[Queue v4](QUEUE_V4_CN.md) enforces explicit phases, capacity gates and SHA-bound maintenance without silently retrying or selecting results. Its outer timeout uses process groups; workers can create separate child sessions, so operators must verify no child GPU process remains after an outer timeout. A descendant-aware successor is pending. [State planning](state_final_plan/README_CN.md) verifies exact frozen-source and input gates. [E1 presentation](../presentation/README_CN.md) validates full provenance and outputs before aggregation; synthetic CPU fixtures are never paper measurements. These tools do not turn pending experiments into completed results.
