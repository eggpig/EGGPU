# Evaluation source

This directory retains the EGGPU paper workspace's benchmark, correctness,
preprocessing and evidence-assembly source. `../Easy-Graph` is the only EGGPU
implementation used for ongoing development.

- `benchmarking/run_full_baselines.py`: function and baseline measurements.
- `benchmarking/run_eggpu_correctness_gate.py`: evaluation correctness gates.
- `benchmarking/prepare_*`: graph conversion and CSR preparation.
- `benchmarking/run_v15_*` and associated audit/assembly tools: the last A100
  campaign and its protocol corrections.
- `environment_eggpu_cuda128.yml`: historical evaluation dependencies.

For a current installation check, run `../scripts/run_smoke.sh` from the
repository root. Performance runners also require their declared baseline
libraries and data. Historical orchestration scripts record machine-specific
paths, GPU assignments and frozen build locations; review these before rerunning
on a different machine. A fresh clone alone is not a complete performance setup.

Raw `datasets/`, `benchmarking/results/`, `build_artifacts/` and third-party
runtime environments remain local and are ignored by Git. Dataset provenance
and the published download are described in `../artifact/datasets/`.
The local `../.maintenance/consolidation_20260922/` manifest records the five
retained evidence packages, their raw dependencies and cleanup operations.
See `../artifact/STATUS.md` for measured coverage and known compatibility issues.
