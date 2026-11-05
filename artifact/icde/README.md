# ICDE artifact guide

This is the current source and reproduction entry. The first public commit and source3 evidence remain historical; current source is `arena_output_ownership_v2`, digest `cb14360e2dfb934b8a28f9875fb80639ce4503bc6ed27956201b18e9edec6869`. [manifest.json](manifest.json), [current source](provenance/current_source.json), and [candidate delta](provenance/candidate_delta.json) explicitly separate the current path-to-SHA256-and-byte-size digest from the historical source3 digest. Verify every declared file before use.

## Current evidence

Current-source checks include 178 public CPU contracts (the earlier candidate receipt records its original 153-test scope), 33 real GPU semantic/owned-output checks, and 16 corrected CSR/global-L1 tiny configurations. Real-graph targets and other workflows are qualifying separately. Earlier native FP64 lease qualification covered one and eight GPUs on the integrated parent. The broader E6/16-request/state/baseline checks have their own source/runner identities. [Current GPU scope](provenance/current_gpu_qualification.json) and [campaign status](campaign_status.json) distinguish these; qualification does not establish formal performance. The full new E1–E6 formal matrix and 90% single-GPU target remain unmeasured.

## Build and test

Run from the repository root:

```sh
python -B artifact/icde/tools/reproduce.py verify
python -B artifact/icde/tools/reproduce.py test-cpu --work /outside/new-source-cpu-checks
python -B artifact/icde/tools/reproduce.py test-campaign-cpu --work /outside/new-campaign-cpu-checks
python -B artifact/icde/tools/reproduce.py build --arena --build-dir /outside/new-build --cuda-root /your/cuda --architecture 89 --jobs 2
```

The build wrapper is plan-only unless `--execute` is supplied. Provide compatible Python development libraries, C++ compiler, CMake, CUDA, and NCCL include/library paths. The measured machine used RTX 4090 D devices and host staging; [setup_public.json](setup_public.json) records the nonprivate environment. Actual host UUIDs, free memory, CPU affinity, loaded binary hashes, and lock paths must be captured locally. Build output and GPU qualification are different statuses. The public SG route needs the arena libraries built with `--arena`; fixed-backend native scaling uses the native extension.

Normal public imports and owned CSR admission are tested without replacing the EasyGraph root namespace. CPU tests need NumPy/SciPy/pytest; some baseline helpers additionally need their explicit dependencies. `capture-local-setup` only records the current host and never authorizes a GPU run by itself.

## Inputs and experiments

Use [inputs/README_CN.md](inputs/README_CN.md) for the seven frozen original sources, exact normalized variants, weight/source rules, dependency locks and preflight budgets. Six graphs have matched complete regenerated payload hashes; TW has since completed its original-format preparation and full qualification. [Current input completion](inputs/completion.json) supersedes the earlier frozen TW-pending status. The portable tool sequence is `preflight.py`, `setup_tools.py`, `fetch.py`, then `prepare_inputs.py`. Legacy compressed caches may be imported only after full original-file verification; old prepared CSR is not a substitute for original-source regeneration.

[Campaign documentation](campaign/README.md) gives each worker/controller and [latest.json](campaign/latest.json) identifies the precise versions. Match semantics, resources, weights, sources, complete host outputs, and continuous time boundaries before comparing systems. Rebuilt or relocated sources require their own tiny and target proofs. SG single-call and the continuous 16-request sequence are separate workloads. State-v4 records real FP64 old-reader lifetimes, exact proof chains, and lossless content-addressed full outputs; it requires its new GPU gates. GUM/cuGraph-MG/author SG clock corrections remain explicitly scoped candidates until measured qualification passes.

No controller silently runs all paper experiments. Inputs, author baseline builds, machine admission, explicit GPU execution, qualification and formal repetitions remain separate. The owner must supervise process groups and enforce the global GPU lock. Ten formal repetitions are planned per configuration; do not discard failures or retain only winning cases.

## Paper and results

[Related Work](paper/RELATED_WORK_NARRATIVE_20261104_CN.md), [experiment/presentation design](paper/PAPER_EXPERIMENT_DESIGN_20261104_CN.md), [baseline matrix](baselines/README_CN.md), and [figure placeholders](paper/figures/README.md) are included. The [E1 aggregation tools](presentation/README_CN.md) include complete proof and output-hash checks and have CPU qualification; a real formal export and the full formal measurements remain pending. Qualification raw vectors remain in the operator's evidence store and are not shipped as graph/binary payloads in this source checkpoint.

**PR protocol correction:** the packaged SG cuGraph adapter uses global L1 epsilon `1e-9` directly, matching the installed cuGraph implementation. Earlier SG cuGraph PR and cuGraph 16-request timing records remain historical and are not comparable under this corrected protocol. The current source and corrected CSR runner have passed all 16 SG tiny configurations; other exact runner/real-graph gates remain separate. See [the correction record](provenance/pr_protocol_correction.json).

The state planner v3 and unweighted-PR control v2 preserve the selected Python environment and bind its runtime identity. They supersede the earlier planners that resolved the interpreter symlink incorrectly. Their CPU admission checks passed; this does not replace GPU or formal qualification.
