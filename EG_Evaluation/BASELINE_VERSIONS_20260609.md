# EGGPU Baseline Version Lock, 2026-06-09

This file records the package boundary used for reproducible EGGPU paper
evaluation.  The benchmark runner still records per-run metadata under each
`benchmarking/results/<run_id>/metadata.json`; this document is the repository
level installation target.

## Recommended Reproduction Environment

Use `environment_eggpu_cuda128.yml` for a clean new machine:

```bash
conda env create -f EG_Evaluation/environment_eggpu_cuda128.yml
conda activate EGGPU
```

Pinned baseline package versions:

| Baseline or dependency | Version |
| --- | --- |
| Python | 3.10 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| NetworkX | 3.4.2 |
| python-igraph | 1.0.0 |
| CuPy | 14.0.1 |
| RAPIDS cuDF | 26.02.01 |
| RAPIDS cuGraph | 26.02.00 |
| nx-cugraph | 26.02.00 |
| psutil | 7.2.2 |
| nvidia-ml-py / pynvml | 13.595.45 |
| matplotlib | 3.10.9 |

Gunrock is treated as an optional external executable baseline.  When the
configured Gunrock binary is absent or a function is semantically unsupported,
the runner emits unavailable rows instead of timing a fallback.

## CUDA Compatibility Boundary

There are two different compatibility targets:

1. EGGPU source build: the native EasyGraph/EGGPU extension is built from source
   through CMake and a CUDA toolkit selected by `EGGPU_CUDA_ROOT`, `CUDA_PATH`,
   `CUDA_HOME`, `CUDAToolkit_ROOT`, or `CONDA_PREFIX`.  Set
   `EGGPU_CUDA_ARCHITECTURES` to the target GPU compute capability, for example
   `80` for A100 and `89` for RTX 4090.
2. Full paper baseline reproduction: RAPIDS/cuGraph/nx-cugraph must also be
   available.  RAPIDS releases are built and tested against specific CUDA and
   Python versions, and current RAPIDS documentation requires NVIDIA Volta or
   newer GPUs with compute capability 7.0+.  Use the RAPIDS install selector if
   the exact conda solve differs on a new machine:
   https://docs.rapids.ai/install and https://docs.rapids.ai/platform-support/.

CUDA 11.x is not the recommended target for full paper reproduction.  It may be
possible to experiment with an EGGPU-only source build on a CUDA 11.8 class
toolchain, but the full benchmark matrix depends on RAPIDS/cuGraph/nx-cugraph
versions that are not locked here for CUDA 11.x.  Treat CUDA 12.x or newer with
RAPIDS 26.02 as the paper reproduction path.

NVIDIA compute capability references:

- NVIDIA CUDA GPU table: https://developer.nvidia.com/cuda-gpus
- RAPIDS install requirements: https://docs.rapids.ai/install
- RAPIDS platform support: https://docs.rapids.ai/platform-support/

## Current Machine Observation

The local server environment inspected on 2026-06-09 is not used as a hard
requirement for every machine, but it is useful for audit trails:

| Component | Observed value |
| --- | --- |
| Python | 3.10.20 |
| nvcc | CUDA compilation tools 13.2, V13.2.78 |
| CUDA conda metadata | `cuda-version 13.2`, RAPIDS `cuda13` packages |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| NetworkX | 3.4.2 |
| python-igraph | 1.0.0 |
| CuPy | 14.0.1, CUDA runtime 13.1 |
| cuDF | 26.02.01 |
| cuGraph | 26.02.00 |
| nx-cugraph | 26.02.00 |
| psutil | 7.2.2 |
| nvidia-ml-py | 13.595.45 |
| matplotlib | 3.10.9 |

This difference is why the README describes a recommended pinned environment
and also provides `scripts/check_eggpu_compat.py` for machine-specific checks.
