#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
CUDA_ROOT="${EGGPU_CUDA_ROOT:-/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang}"
GPU="${TARGET_GPU:-7}"
STAMP="${TARGET_TS:-$(date +%Y%m%d_%H%M%S)}"
RUNTIME="${ROOT}/build_artifacts/eggpu_candidate_20260726_v4"
SOURCE="${ROOT}/../Easy-Graph"
EXPECTED_SHA="4f27133874dedd64a2f81043fd3d76939b10999e97ceb2030b0cad1a842c43cf"
OUT="${TARGET_OUT:-${ROOT}/benchmarking/results/v4_targeted_large_scaling_gpu${GPU}_${STAMP}}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export EGGPU_MONITOR_GPU_INDEX="${GPU}"
export EGGPU_CUDA_ROOT="${CUDA_ROOT}"
export EASYGRAPH_ENABLE_GPU=TRUE
export EASYGRAPH_GPU_STRICT_ERRORS=TRUE
export EASYGRAPH_GPU_RESULT_CACHE=FALSE
export EGGPU_GPU_VISIBILITY_MARKER=FALSE
export PYTHONPATH="${RUNTIME}:${SOURCE}:${ROOT}/benchmarking${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib:${CUDA_ROOT}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${OUT}"
cd "${ROOT}"

EXTENSION="$("${PYTHON}" -c 'import cpp_easygraph; print(cpp_easygraph.__file__)')"
ACTUAL_SHA="$("${PYTHON}" -c 'import hashlib,pathlib,cpp_easygraph; print(hashlib.sha256(pathlib.Path(cpp_easygraph.__file__).read_bytes()).hexdigest())')"
if [[ "${EXTENSION}" != "${RUNTIME}"/* || "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
  printf 'artifact gate failed: extension=%s sha256=%s\n' "${EXTENSION}" "${ACTUAL_SHA}" >&2
  exit 3
fi

cat >"${OUT}/protocol.json" <<EOF
{
  "gpu": ${GPU},
  "extension": "${EXTENSION}",
  "extension_sha256": "${ACTUAL_SHA}",
  "timing_repeat": 5,
  "memory_repeat": 3,
  "warmup": 2,
  "stages": [
    "GAP-twitter BC with 240-second call budget",
    "GAP-twitter SCC with 600-second call budget",
    "GAP-twitter selected-node structural-hole queries",
    "R-MAT S20/S22/S24/S26 Dijkstra",
    "device-cache mutation correctness"
  ],
  "note": "Each stage writes an independent result directory. A nonzero stage exit is recorded and does not erase later stages."
}
EOF

run_stage() {
  local name="$1"
  shift
  printf '[stage] %s\n' "${name}" | tee -a "${OUT}/launcher.log"
  "$@" |& tee "${OUT}/${name}.log"
  local rc="${PIPESTATUS[0]}"
  printf '%s,%s\n' "${name}" "${rc}" >>"${OUT}/stage_exit_codes.csv"
  return 0
}

printf 'stage,exit_code\n' >"${OUT}/stage_exit_codes.csv"

GAP="${ROOT}/datasets/scaling/csr/GAP-twitter/GAP-twitter.json"
run_stage gap_bc \
  "${PYTHON}" benchmarking/run_eggpu_scaling.py \
  --manifests "${GAP}" --functions BC \
  --output-dir "${OUT}/gap_bc" \
  --repeat 5 --warmup 2 --memory-repeat 3 \
  --source-count 8 --bc-source-count 16 --closeness-source-count 16 \
  --memory-poll-ms 2 --timeout 240 --first-use-call-timeout 240 \
  --load-timeout 900 --hard-call-timeout --continue-on-failure --no-resume

run_stage gap_scc \
  "${PYTHON}" benchmarking/run_eggpu_scaling.py \
  --manifests "${GAP}" --functions SCC \
  --output-dir "${OUT}/gap_scc" \
  --repeat 5 --warmup 2 --memory-repeat 3 \
  --source-count 8 --bc-source-count 16 --closeness-source-count 16 \
  --memory-poll-ms 2 --timeout 600 --first-use-call-timeout 600 \
  --load-timeout 900 --hard-call-timeout --continue-on-failure --no-resume

run_stage gap_structural_subset16 \
  "${PYTHON}" benchmarking/run_eggpu_scaling.py \
  --manifests "${GAP}" \
  --functions EffectiveSize,Efficiency,Constraint,Hierarchy \
  --structural-node-count 16 \
  --output-dir "${OUT}/gap_structural_subset16" \
  --repeat 5 --warmup 2 --memory-repeat 3 \
  --source-count 8 --bc-source-count 16 --closeness-source-count 16 \
  --memory-poll-ms 2 --timeout 240 --first-use-call-timeout 240 \
  --load-timeout 900 --hard-call-timeout --continue-on-failure --no-resume

run_stage rmat_dijkstra \
  "${PYTHON}" benchmarking/run_eggpu_scaling.py \
  --manifests \
    datasets/scaling/csr/R-MAT-S20-EF16/R-MAT-S20-EF16.json \
    datasets/scaling/csr/R-MAT-S22-EF16/R-MAT-S22-EF16.json \
    datasets/scaling/csr/R-MAT-S24-EF16/R-MAT-S24-EF16.json \
    datasets/scaling/csr/R-MAT-S26-EF16/R-MAT-S26-EF16.json \
  --functions Dijkstra \
  --output-dir "${OUT}/rmat_dijkstra" \
  --repeat 5 --warmup 2 --memory-repeat 3 \
  --source-count 8 --bc-source-count 16 --closeness-source-count 16 \
  --memory-poll-ms 2 --timeout 100 --first-use-call-timeout 100 \
  --load-timeout 900 --hard-call-timeout --continue-on-failure --no-resume

run_stage cache_mutation \
  "${PYTHON}" benchmarking/validate_device_graph_cache_identity.py \
  --output "${OUT}/cache_mutation.json"

printf 'completed\n' >"${OUT}/COMPLETED"
printf 'Targeted v4 supplement finished: %s\n' "${OUT}" | tee -a "${OUT}/launcher.log"
