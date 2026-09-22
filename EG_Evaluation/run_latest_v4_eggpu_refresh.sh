#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# < 2 )); then
  echo "usage: $0 MODE GPU [OUTPUT_DIR]" >&2
  echo "MODE must be one of: main11, com-orkut, rmat4" >&2
  exit 2
fi
MODE="$1"
GPU="$2"
PYTHON="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
CUDA_ROOT="${EGGPU_CUDA_ROOT:-/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang}"
RUNTIME_ROOT="${ROOT}/build_artifacts/eggpu_candidate_20260726_v4"
PYTHON_SOURCE_ROOT="$(cd "${ROOT}/../Easy-Graph" && pwd)"
EXPECTED_EXTENSION_SHA256="4f27133874dedd64a2f81043fd3d76939b10999e97ceb2030b0cad1a842c43cf"
STAMP="20260726"

export PYTHONPATH="${RUNTIME_ROOT}:${PYTHON_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export EASYGRAPH_ENABLE_GPU=TRUE
export EASYGRAPH_GPU_RESULT_CACHE=FALSE
export EASYGRAPH_GPU_STRICT_ERRORS=TRUE
export EGGPU_USE_CONDA_RUN=FALSE
export EGGPU_CUDA_ROOT="${CUDA_ROOT}"
export EGGPU_SKIP_PLOTS=TRUE
export EGGPU_GPU_VISIBILITY_MARKER=FALSE
export CUDA_VISIBLE_DEVICES="${GPU}"

cd "${ROOT}"

LOADED_EASYGRAPH="$("${PYTHON}" -c 'import easygraph; print(easygraph.__file__)')"
LOADED_EXTENSION="$("${PYTHON}" -c 'import cpp_easygraph; print(cpp_easygraph.__file__)')"
LOADED_EXTENSION_SHA256="$("${PYTHON}" -c 'import hashlib, pathlib, cpp_easygraph; print(hashlib.sha256(pathlib.Path(cpp_easygraph.__file__).read_bytes()).hexdigest())')"
case "${LOADED_EASYGRAPH}" in
  "${PYTHON_SOURCE_ROOT}"/*) ;;
  *)
    echo "fatal: easygraph loaded from unexpected path: ${LOADED_EASYGRAPH}" >&2
    exit 3
    ;;
esac
case "${LOADED_EXTENSION}" in
  "${RUNTIME_ROOT}"/*) ;;
  *)
    echo "fatal: cpp_easygraph loaded from unexpected path: ${LOADED_EXTENSION}" >&2
    exit 3
    ;;
esac
if [[ "${LOADED_EXTENSION_SHA256}" != "${EXPECTED_EXTENSION_SHA256}" ]]; then
  echo "fatal: cpp_easygraph SHA-256 mismatch: ${LOADED_EXTENSION_SHA256}" >&2
  exit 3
fi
printf '[artifact-gate] easygraph=%s\n' "${LOADED_EASYGRAPH}"
printf '[artifact-gate] cpp_easygraph=%s sha256=%s\n' \
  "${LOADED_EXTENSION}" "${LOADED_EXTENSION_SHA256}"

case "${MODE}" in
  main11)
    OUT="${3:-${ROOT}/benchmarking/results/latest_v4_main11_gpu${GPU}_${STAMP}_repeat5_memory3}"
    exec "${PYTHON}" benchmarking/run_split_full_baselines.py \
      --gpu "${GPU}" \
      --out-dir "${OUT}" \
      --repeat 5 \
      --memory-repeat 3 \
      --easygraph-warmup 2 \
      --library-timeout 100 \
      --inter-run-cooldown 0.2 \
      --easygraph-repo "${RUNTIME_ROOT}" \
      --datasets ca-HepTh,LastFM,p2p-Gnutella04,ca-HepPh,email-Enron,ca-CondMat,soc-Epinions1,soc-Slashdot0811,ER-100k,web-NotreDame,com-youtube \
      --functions all \
      --baselines EGGPU
    ;;
  com-orkut)
    OUT="${3:-${ROOT}/benchmarking/results/latest_v4_com_orkut_gpu${GPU}_${STAMP}_repeat5_memory3}"
    exec "${PYTHON}" benchmarking/run_eggpu_scaling.py \
      --manifests datasets/scaling/csr/com-Orkut/com-Orkut.json \
      --functions all \
      --output-dir "${OUT}" \
      --repeat 5 \
      --warmup 2 \
      --memory-repeat 3 \
      --source-count 8 \
      --bc-source-count 16 \
      --closeness-source-count 16 \
      --memory-poll-ms 2 \
      --timeout 100 \
      --load-timeout 900 \
      --hard-call-timeout \
      --continue-on-failure \
      --no-resume
    ;;
  rmat4)
    OUT="${3:-${ROOT}/benchmarking/results/latest_v4_rmat4_gpu${GPU}_${STAMP}_repeat5_memory3}"
    exec "${PYTHON}" benchmarking/run_eggpu_scaling.py \
      --manifests \
        datasets/scaling/csr/R-MAT-S20-EF16/R-MAT-S20-EF16.json \
        datasets/scaling/csr/R-MAT-S22-EF16/R-MAT-S22-EF16.json \
        datasets/scaling/csr/R-MAT-S24-EF16/R-MAT-S24-EF16.json \
        datasets/scaling/csr/R-MAT-S26-EF16/R-MAT-S26-EF16.json \
      --functions PageRank,WCC,BFS,SSSP \
      --output-dir "${OUT}" \
      --repeat 5 \
      --warmup 2 \
      --memory-repeat 3 \
      --source-count 8 \
      --bc-source-count 16 \
      --closeness-source-count 16 \
      --memory-poll-ms 2 \
      --timeout 100 \
      --load-timeout 900 \
      --hard-call-timeout \
      --continue-on-failure \
      --no-resume
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
