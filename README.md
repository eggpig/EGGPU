# EGGPU

EGGPU runs graph analytics on one or multiple GPUs and manages graph state across successive requests. D-Maintain updates the representations required by a request. C-Retain manages retained state using shared physical allocations and observed reuse demand.

The implementation is in `Easy-Graph/`. The [artifact guide](artifact/icde/README.md) describes the fixed datasets, baseline adapters, six experiments, and measurement protocol.

## Build and run

Use a Python environment with NumPy, SciPy, pybind11, and CMake, together with a compatible CUDA toolkit. The build uses the caller's environment and writes outside this checkout.

```bash
python -B artifact/icde/tools/verify.py
python -B artifact/icde/tools/build.py \
  --build-dir /path/to/eggpu-build \
  --cuda-root /path/to/cuda --architecture 89 --arena --execute
python -B artifact/icde/tools/reproduce.py run-native-example --help
```

Architecture `89` targets the RTX 4090 D used in the main experiments; use `80` for A100. For NCCL support, supply both `--nccl-include` and `--nccl-library` to the build command. Record the locally selected communication backend with the experiment.

The [reproduction commands](artifact/icde/README.md#experiments) cover input preparation, single- and multi-GPU execution, maintenance and retention workloads, and the dependent application. All results are written to caller-selected directories.

## Source and checks

This release contains source `5af9876b8596b3c8b769fe1865890eb3fe6afc91cd8bf60c9c9bf76dce9ffdeb`, including immutable host-input sharing, retained device owners, canonical WCC output handoff, and independently owned host results. The source inventory and checksums are in `artifact/icde/provenance/`. Performance receipts bind revision `308f80a5f866`; this release adds NumPy boolean-input rejection, a checked fallback for older CPU extensions, and an updated CPU download fixture. CUDA/C++ sources are unchanged.

```bash
python -B artifact/icde/tools/test_cpu.py --work /path/to/new-cpu-check
python -B artifact/icde/tools/reproduce.py status
```

See [LICENSE](LICENSE), [NOTICE](NOTICE), and [CITATION.cff](CITATION.cff) for licensing and attribution.
