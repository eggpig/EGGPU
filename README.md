# EGGPU

EGGPU integrates single-GPU and multi-GPU graph analysis with reusable execution state, version-aware maintenance, and memory-budget-aware retention. The maintained implementation is [`Easy-Graph/`](Easy-Graph/).

The current candidate adds a fixed capability policy to the source3 implementation. `Session.create` selects the single-GPU arena for static or journal workloads and native partition execution for immutable snapshots or multiple GPUs. Selection uses requested devices and state semantics; it does not use graph names, measured timings, or a fallback after failure. Use `Session.native` at every GPU count for fixed-backend scaling.

## Status and reproduction

```sh
python -B artifact/icde/tools/reproduce.py status
python -B artifact/icde/tools/reproduce.py verify
python -B artifact/icde/tools/reproduce.py build --build-dir /outside/new-build --cuda-root /your/cuda --architecture 89 --jobs 2
```

The build command prints a plan. Add `--execute` to compile, with the target-machine NCCL include/library paths as required. Rebuilt binaries need their own correctness and numerical qualification. The capability-dispatch patch has passed 46 CPU checks, and the closure-LRU control passed a separate 10-test CPU suite; GPU qualification and the new seven-dataset performance campaign are recorded separately and are not inferred from the older source3 results.

The [artifact guide](artifact/icde/README.md) covers input preparation, supported execution modes, qualification, and available commands. E1–E6 are designed and registered; complete portable campaign runners and the new measurements remain pending. This checkpoint does not claim a completed paper reproducer or a measured 90% leading rate.

## Paper and experiment organization

- [Related Work narrative](artifact/icde/paper/RELATED_WORK_NARRATIVE_20261104_CN.md): assumptions and the two-part argument, ready for author editing.
- [Experiment and presentation design](artifact/icde/paper/PAPER_EXPERIMENT_DESIGN_20261104_CN.md): E1–E6, matched baselines, visual roles, captions and setup.
- [Baseline capability matrix](artifact/icde/baselines/README_CN.md): author implementations, applicable functions/resources, and qualification gaps.
- [Seven frozen datasets](artifact/icde/datasets.json): US, EP, LJ, OR, IN, UK, TW, with source variants and deterministic normalization/weight/source rules.
- [Figure slots](artifact/icde/paper/figures/README.md): reserved mechanism and framework artwork; final drawings will be supplied separately.

Only current and previous successful presentation generations should be exposed as results. Formal repetitions remain individually traceable; identical immutable output arrays may share storage without deleting measurements. Historical VLDB evaluation source and notes retained elsewhere in this repository describe their original releases.

See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`CITATION.cff`](CITATION.cff). The source publication includes no credentials, private host configuration, compiled libraries, or graph payloads.
