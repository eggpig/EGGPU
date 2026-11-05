# EGGPU

EGGPU provides single-GPU and multi-GPU graph analysis with D-Maintain for version-aware maintenance and C-Retain for memory-budget-aware retention. The implementation is in `Easy-Graph/`.

This development checkpoint uses exact source SHA-256 `d84ff388e1e40d1fd6ed83fcaceecd0491fc0451adffaff657a95940b1803f84`. It includes safe native CSR host-registration lifetimes, a strict in-memory CSR entry, fused weight admission, the optimized P1 preparation path, and the E6 dependent workflow without an intermediate nine-array file round trip. The research source is unchanged by packaging. Benchmark relocations and optional CPU-affinity mapping are recorded separately.

```sh
python -B artifact/icde/tools/verify.py
python -B artifact/icde/tools/reproduce.py status
python -B artifact/icde/tools/reproduce.py protocol-grid --kind state-configurations --output state-configurations.json
python -B artifact/icde/tools/build.py --build-dir /outside/new-build --cuda-root /your/cuda --architecture 89 --jobs 2
```

The build command prints a plan unless `--execute` is supplied. Build artifacts, input payloads and results belong outside this checkout. Rebuilt binaries and relocated runners require new matching qualifications before formal timing.

The fixed graphs are US, EP, LJ, OR, IN, UK and TW. Exact source variants and normalization are documented in `artifact/icde/datasets.json` and `artifact/icde/paper/DATASET_AND_EXPERIMENT_FREEZE_20261104_CN.md`. The formal queues have started; results and qualification coverage remain incomplete. This checkpoint contains no completed formal result matrix or claim of a measured SOTA percentage.

See `artifact/icde/README.md` for current entry points, timing boundaries, baseline precision, reproduction checks and pending work. Mechanism and framework artwork remain reserved for the collaborator.

The followup includes recoverable E4-S output archival, a read-only per-cell disposition inspector, watcher binding-schema fixes, the new E3/E4 and E6 protocol candidates, twenty primary Related Work sources, and the Chinese introduction with a reproducible first-page counterexample. The independent WCC int64-offset candidate is an overlay under `artifact/icde/experimental/native_wcc_wide_v2`; main `Easy-Graph/` remains d84. See `artifact/icde/FOLLOWUP_CN.md`.

Latest protocol and paper update: strict 561-cell export, incremental admission gates, qualification capacity guards, a separate arena diagnostic and WCC runner v3. Main d84 source is unchanged. See [the current runbook](artifact/icde/PROTOCOLS_UPDATE_CN.md) and [paper navigation](artifact/icde/paper/README_CN.md). This checkpoint does not contain a complete formal result matrix.
