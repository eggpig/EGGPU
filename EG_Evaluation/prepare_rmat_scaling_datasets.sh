#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
CUDA_ROOT="${EGGPU_CUDA_ROOT:-/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang}"
NVCC="${CUDA_ROOT}/bin/nvcc"
GENERATOR_SRC="${ROOT}/benchmarking/tools/generate_rmat_csr.cu"
GENERATOR_BIN="${ROOT}/benchmarking/tools/generate_rmat_csr"
OUTPUT_ROOT="${ROOT}/datasets/scaling/csr"
GPU="${RMAT_GPU:-0}"
EDGE_FACTOR="${RMAT_EDGE_FACTOR:-16}"
SEED="${RMAT_SEED:-20260716}"
SCALES="${RMAT_SCALES:-20 22 24 26}"
CUDA_ARCH="${RMAT_CUDA_ARCH:-80}"
COMPILE_COMMAND="${NVCC} -O3 -std=c++17 -DNDEBUG -arch=sm_${CUDA_ARCH} ${GENERATOR_SRC} -o ${GENERATOR_BIN}"

if [[ ! -x "${NVCC}" ]]; then
  echo "local nvcc not found: ${NVCC}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
if [[ ! -x "${GENERATOR_BIN}" || "${GENERATOR_SRC}" -nt "${GENERATOR_BIN}" ]]; then
  "${NVCC}" -O3 -std=c++17 -DNDEBUG -arch="sm_${CUDA_ARCH}" \
    "${GENERATOR_SRC}" -o "${GENERATOR_BIN}"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
manifests=()
all_scales="10 ${SCALES}"
for scale in ${all_scales}; do
  name="R-MAT-S${scale}-EF${EDGE_FACTOR}"
  output_dir="${OUTPUT_ROOT}/${name}"
  manifest="${output_dir}/${name}.json"
  mkdir -p "${output_dir}"
  if [[ ! -s "${manifest}" ]]; then
    /usr/bin/time -v -o "${output_dir}/generation_resource_usage.txt" \
      "${GENERATOR_BIN}" \
        --scale "${scale}" --edge-factor "${EDGE_FACTOR}" --seed "${SEED}" \
        --name "${name}" --output-dir "${output_dir}"
  else
    echo "Reusing controlled R-MAT artifact: ${manifest}"
  fi
  "${PYTHON_BIN}" "${ROOT}/benchmarking/finalize_rmat_manifest.py" \
    "${manifest}" \
    --generator-source "${GENERATOR_SRC}" \
    --generator-binary "${GENERATOR_BIN}" \
    --compile-command "${COMPILE_COMMAND}"
  if [[ "${scale}" == "10" ]]; then
    reference_manifest="${manifest}"
  else
    manifests+=("${manifest}")
  fi
done

PYTHONPATH="${ROOT}/../Easy-Graph${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${ROOT}/benchmarking/validate_bulk_csr_manifests.py" \
  "${reference_manifest}" "${manifests[@]}"
PYTHONPATH="${ROOT}/../Easy-Graph${PYTHONPATH:+:${PYTHONPATH}}" \
EASYGRAPH_ENABLE_GPU=TRUE EASYGRAPH_GPU_STRICT_ERRORS=TRUE \
EGGPU_ALLOW_CUDA_SYNC=TRUE \
LD_LIBRARY_PATH="${CUDA_ROOT}/lib:${CUDA_ROOT}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${PYTHON_BIN}" "${ROOT}/benchmarking/validate_rmat_reference_gate.py" \
  "${reference_manifest}" \
  --output "${OUTPUT_ROOT}/R-MAT-S10-EF${EDGE_FACTOR}/reference_validation.json"

printf 'Prepared controlled R-MAT manifests:\n'
printf '  %s (CPU reference gate)\n' "${reference_manifest}"
printf '  %s\n' "${manifests[@]}"
