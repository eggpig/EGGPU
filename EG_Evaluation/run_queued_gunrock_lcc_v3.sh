#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/EG_Evaluation"
PY="/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python"
CANDIDATE="${ROOT}/build_artifacts/eggpu_candidate_20260725_v3"
OUT="${ROOT}/benchmarking/results/latest_gunrock_lcc_aligned_gpu5_20260725_repeat5_memory3"
LOG="${OUT}.launch.log"

while pgrep -f "run_split_full_baselines.py --gpu 4 --out-dir ${ROOT}/benchmarking/results/latest_v3_main11_gpu4_20260725_repeat5_memory3" >/dev/null; do
    sleep 30
done

cd "${ROOT}"
export EGGPU_CUDA_ROOT="/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang"
export EG_GUNROCK_BIN_PATHS="/home/dataset-assist-0/einwang/workspace/haorandu/EG_Evaluation/gunrock_latest/build_cuda132_a100_migrated/bin:/home/dataset-assist-0/einwang/workspace/haorandu/EG_Evaluation/gunrock_legacy_master/build_lcc_result_cuda128_clean/bin"

"${PY}" benchmarking/run_split_full_baselines.py \
    --gpu 5 \
    --out-dir "${OUT}" \
    --repeat 5 \
    --memory-repeat 3 \
    --easygraph-warmup 2 \
    --library-timeout 100 \
    --inter-run-cooldown 0.2 \
    --easygraph-repo "${CANDIDATE}" \
    --datasets "ca-HepTh,LastFM,p2p-Gnutella04,ca-HepPh,email-Enron,ca-CondMat,soc-Epinions1,soc-Slashdot0811,ER-100k,web-NotreDame,com-youtube" \
    --functions "LCC" \
    --baselines "Gunrock" >"${LOG}" 2>&1
