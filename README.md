# EGGPU

EGGPU is an end-to-end GPU backend for EasyGraph network analysis. Existing
EasyGraph calls enter native C++/CUDA implementations while reusable host and
device graph state avoids rebuilding the same representations across compatible
calls. The current implementation exposes 16 analysis functions in four
families and preserves their documented EasyGraph return forms.

This repository is the **framework and correctness artifact**. It contains the
implementation, build helpers, code-to-design map, dataset provenance, and a
small GPU correctness smoke test. It deliberately contains no paper source,
benchmark results, raw datasets, or performance harness. Dataset files will be
published separately; [`artifact/datasets/`](artifact/datasets/) records their
provenance and planned archive metadata.

## Repository layout

- [`Easy-Graph/`](Easy-Graph/) — EasyGraph, EGGPU dispatch, native bindings,
  reusable graph state, and CUDA operators.
- [`artifact/`](artifact/) — architecture, function coverage, correctness
  protocol, paper-to-code map, and dataset metadata.
- [`scripts/`](scripts/) — environment check, native build, correctness smoke,
  and release-content verifier.

## Requirements

The reference environment is Linux x86-64, Python 3.10, CUDA Toolkit 12.8, and
an NVIDIA GPU. The build has been exercised on compute capabilities 8.0 and
8.6. A compatible NVIDIA driver, `nvidia-smi`, CMake 3.23 or newer, and a C++
compiler are required.

Create the supplied environment:

```bash
conda env create -f environment.yml
conda activate eggpu-artifact
export EGGPU_CUDA_ROOT="$CONDA_PREFIX"
python scripts/check_eggpu_compat.py --strict
```

If CUDA is installed outside Conda, set `EGGPU_CUDA_ROOT` to the toolkit root
containing `bin/nvcc`.

## Build and verify

```bash
bash scripts/build_eggpu.sh
python scripts/check_eggpu_compat.py --strict --require-extension
GPU=0 bash scripts/run_smoke.sh
python scripts/verify_release.py
```

The build helper detects the visible GPU architecture. Override it when
cross-compiling, for example:

```bash
EGGPU_CUDA_ARCHITECTURES=86 bash scripts/build_eggpu.sh
```

The smoke test performs no benchmarking. It compares the public GPU calls with
the corresponding EasyGraph CPU results on small deterministic graphs and
fails if strict GPU dispatch or a result contract is violated.

## Use EGGPU

One environment setting enables EGGPU for supported calls:

```bash
export EASYGRAPH_ENABLE_GPU=TRUE
```

Then use the normal EasyGraph API:

```python
import easygraph as eg

graph = eg.DiGraph()
graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
rank = eg.pagerank(graph)
components = list(eg.strongly_connected_components(graph))
```

For validation and deployment, strict mode prevents an unsupported or failed
GPU path from silently continuing on the CPU:

```bash
export EASYGRAPH_GPU_STRICT_ERRORS=TRUE
```

See [`artifact/FUNCTION_SUPPORT.md`](artifact/FUNCTION_SUPPORT.md) for the 16
public calls and their input boundaries.

## Artifact boundary

This code release supports implementation inspection and functional
verification. It is not, by itself, the complete performance-reproduction
package for the paper. The latter also requires the archived datasets,
experiment protocol, baseline environments, and result provenance.

## License and citation

EGGPU is distributed under the BSD 3-Clause License. See [`LICENSE`](LICENSE),
[`NOTICE`](NOTICE), and [`CITATION.cff`](CITATION.cff).
