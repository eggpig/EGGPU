#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/EG_Evaluation"
PYTHON_BIN="/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python"
RESULT_ROOT="${ROOT}/benchmarking/results/nxcugraph_strict_main99_20260730"
FUNCTIONS="PageRank,LCC,WCC,BFS,Dijkstra,BellmanFord,SSSP,KCore"

launch_group() {
  local gpu="$1"
  local label="$2"
  local datasets="$3"
  local out_dir="${RESULT_ROOT}/${label}"
  local session="eggpu_nxcg99_${label}_0730"
  mkdir -p "${out_dir}/cupy_cache"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "session already exists: ${session}"
    return
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && \
     CUDA_VISIBLE_DEVICES='${gpu}' \
     CUPY_CACHE_DIR='${out_dir}/cupy_cache' \
     EGGPU_EXTERNAL_VISIBILITY_MARKER=TRUE \
     '${PYTHON_BIN}' benchmarking/run_full_baselines.py \
       --gpu '${gpu}' \
       --out-dir '${out_dir}' \
       --repeat 5 \
       --datasets '${datasets}' \
       --functions '${FUNCTIONS}' \
       --baselines nx-cugraph \
       --measurement-mode timing \
       --nx-cugraph-warmup 3 \
       --library-timeout 600 \
       --inter-run-cooldown 0.1 \
       --sssp-sources 8 \
       >'${out_dir}/run.log' 2>&1"
  echo "launched ${session}: GPU ${gpu}, ${datasets}"
}

launch_group 0 g0_small_a "ca-HepTh,LastFM"
launch_group 1 g1_small_b "p2p-Gnutella04,ca-HepPh"
launch_group 2 g2_medium_a "email-Enron,ca-CondMat"
launch_group 4 g4_medium_b "soc-Epinions1,soc-Slashdot0811"
launch_group 5 g5_large_a "ER-100k,web-NotreDame"
launch_group 6 g6_youtube "com-youtube"

echo "Moderate 88-cell nx-cugraph run launched under ${RESULT_ROOT}"
