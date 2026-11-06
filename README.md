# EGGPU

EGGPU provides single-GPU and multi-GPU graph analysis with D-Maintain for version-aware maintenance and C-Retain for memory-budget-aware retention. The implementation is in `Easy-Graph/`.

This development checkpoint freezes exact source SHA-256 `71ea757d372f8c1b969ea074caaa27d3a77d4cea6d48cd9c6ae9c4b02936d7ce`. It integrates immutable SSSP host payload sharing, retained resident owners, Arena WCC path halving, and registered owning host output. The native build source is `959cb6bac81b36c9c56d7ec02427bf9c7c1dae6d6812e7fc381ae42be59b0d65`; the unified source differs from it only in two Python runtime files. Public builds must compile the complete source. No binaries or machine-specific qualification proofs are shipped.

```sh
python -B artifact/icde/tools/verify.py
python -B artifact/icde/tools/reproduce.py status
python -B artifact/icde/tools/reproduce.py protocol-grid --kind state-configurations --output state-configurations.json
python -B artifact/icde/tools/build.py --build-dir /outside/new-build --cuda-root /your/cuda --architecture 89 --jobs 2
```

The build command prints a plan unless `--execute` is supplied. Build artifacts, input payloads and results belong outside this checkout. Rebuilt binaries and relocated runners require a focused matching qualification; unchanged inputs and already complete current-source observations are reused explicitly.

The fixed graphs are US, EP, LJ, OR, IN, UK and TW. Exact source variants and normalization are documented in `artifact/icde/datasets.json` and `artifact/icde/paper/DATASET_AND_EXPERIMENT_FREEZE_20261104_CN.md`. The formal queues have started; results and qualification coverage remain incomplete. This checkpoint contains no completed formal result matrix or claim of a measured SOTA percentage.

See `artifact/icde/README.md` for current entry points, timing boundaries, baseline precision, reproduction checks and pending work. Mechanism and framework artwork remain reserved for the collaborator.

The current first-pass protocol presents one complete observation per configuration before a second repetition, retaining original target/formal labels, all missing cells and failures. First/ready query times and job wall time remain separate. No current-source result matrix or performance percentage is included. See [the current source and first-pass update](artifact/icde/FIRST_PASS_UPDATE_CN.md). Earlier protocol documents and qualification evidence remain explicitly historical.
