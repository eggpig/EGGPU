#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-4}"
OUT="${2:-${ROOT}/benchmarking/results/latest_v3_main11_gpu${GPU}_20260725_repeat5_memory3}"
PYTHON="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
CUDA_ROOT="${EGGPU_CUDA_ROOT:-/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang}"
RUNTIME_ROOT="${ROOT}/build_artifacts/eggpu_candidate_20260725_v3"
LEGACY_EVAL="/home/dataset-assist-0/einwang/workspace/haorandu/EG_Evaluation"

export PYTHONPATH="${RUNTIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export EASYGRAPH_ENABLE_GPU=TRUE
export EGGPU_USE_CONDA_RUN=FALSE
export EGGPU_CUDA_ROOT="${CUDA_ROOT}"
export EGGPU_SKIP_PLOTS=TRUE
export EGGPU_GPU_VISIBILITY_MARKER=FALSE
export EG_GUNROCK_BIN_PATHS="${LEGACY_EVAL}/gunrock_latest/build_cuda132_a100_migrated/bin:${LEGACY_EVAL}/gunrock_legacy_master/build_lcc_result_cuda128_clean/bin"

cd "${ROOT}"
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
