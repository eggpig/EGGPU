# EGGPU

EGGPU supports single-GPU and multi-GPU graph analysis with shared execution representations, dependency-aware maintenance (D-Maintain), and closure-aware retention (C-Retain). The current implementation is [`Easy-Graph/`](Easy-Graph/).

This checkpoint integrates the public owned-CSR route, fixed capability dispatch, deferred unrelated imports, the closure-LRU control, and explicit readonly FP64 native snapshot leases. `Session.create` selects by requested devices and state semantics; it never selects an implementation by graph name or measured speed. Use `Session.native` at every GPU count for fixed-backend scaling. Extension workflows use their separately documented dense native APIs.

## Reproduction status

```sh
python -B artifact/icde/tools/reproduce.py status
python -B artifact/icde/tools/reproduce.py verify
python -B artifact/icde/tools/reproduce.py test-cpu --work /outside/cpu-checks
python -B artifact/icde/tools/reproduce.py build --arena --build-dir /outside/new-build --cuda-root /your/cuda --architecture 89 --jobs 2
```

The build command prints a plan; add `--execute` and the required NCCL paths to compile. `--arena` supplies the libraries needed by the public single-GPU route. Rebuilt binaries require independent qualification on the actual host.

The current `arena_output_ownership_v2` source digest is `cb14360e2dfb934b8a28f9875fb80639ce4503bc6ed27956201b18e9edec6869`. Its current scope includes 178 public CPU contracts (the earlier candidate receipt records its original 153-test scope), 33 actual GPU semantic/owned-output checks, and 16 EGGPU/cuGraph directed/undirected tiny configurations using the corrected CSR/global-L1 runner. Current real-graph targets and other workflows are qualifying separately. These are qualification calls, **not formal performance repetitions or a completed 90% leading-rate claim**. Earlier source identities and their evidence remain explicitly separate in provenance.

[All seven frozen input datasets](artifact/icde/inputs/completion.json) have qualified canonical payloads. Six were independently regenerated from original compressed sources with complete matching hashes; TW was subsequently prepared from its verified original format. The [input tools](artifact/icde/inputs/README_CN.md) describe sources, dependencies and space checks; [current status](artifact/icde/inputs/CURRENT_STATUS.md) explains the remaining fresh-machine qualification limits.

The [campaign guide](artifact/icde/campaign/README.md) and [version pointers](artifact/icde/campaign/latest.json) distinguish available SG/native/sequence/E6 tools, the state-v4 candidate, and pending external-baseline timing qualification. E1–E6 formal results, the complete baseline build bundle, remain unfinished. Source-bound E1 aggregation tools are included; no real formal table is supplied. This release does not provide an unconditional run-all command.

## Paper organization

- [Related Work narrative](artifact/icde/paper/RELATED_WORK_NARRATIVE_20261104_CN.md): two parts organized around assumptions.
- [Experiment and presentation design](artifact/icde/paper/PAPER_EXPERIMENT_DESIGN_20261104_CN.md): E1–E6, comparable baselines, tables, figures, captions and setup.
- [Baseline capabilities](artifact/icde/baselines/README_CN.md) and [frozen seven datasets](artifact/icde/datasets.json).
- [Mechanism/framework artwork slots](artifact/icde/paper/figures/README.md), reserved for the supplied drawings.

Formal repetitions and failures remain individually traceable. Identical immutable full output vectors may share one content-addressed object; this never removes measurements. Historical evaluation code elsewhere in the repository describes its original source and results. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`CITATION.cff`](CITATION.cff). Credentials, private host configuration, compiled libraries, and graph payloads are not included.

**PR protocol correction:** the packaged SG cuGraph adapter uses global L1 epsilon `1e-9` directly, matching the installed cuGraph implementation. Earlier SG cuGraph PR and cuGraph 16-request timing records remain historical and are not comparable under this corrected protocol. The current source and corrected CSR runner have passed all 16 SG tiny configurations; other exact runner/real-graph gates remain separate. See [the correction record](artifact/icde/provenance/pr_protocol_correction.json).

The supervised [queue v4](artifact/icde/campaign/QUEUE_V4_CN.md) records actual phase counts and pauses on insufficient storage. Its descendant-session timeout limitation is tracked; use worker watchdogs and verify process cleanup before resuming.

The state planner v3 and unweighted-PR control v2 preserve the selected Python environment and bind its runtime identity. They supersede the earlier planners that resolved the interpreter symlink incorrectly. Their CPU admission checks passed; this does not replace GPU or formal qualification.
