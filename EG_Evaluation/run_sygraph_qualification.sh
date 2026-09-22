#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
GPU="${SYGRAPH_GPU:-7}"
REPEAT="${PAPER_REPEAT:-5}"
MEMORY_REPEAT="${PAPER_MEMORY_REPEAT:-3}"
TIMEOUT="${PAPER_TIMEOUT:-100}"
LOAD_TIMEOUT="${PAPER_LOAD_TIMEOUT:-900}"
SOURCE_COMMIT="${SYGRAPH_SOURCE_COMMIT:-71dcb56aed0cd43fefa5752a552e0a10cb69a758}"

CORE_ROOT="${FINAL13_RESULT_ROOT:-${ROOT}/benchmarking/results/final13_missing_gpu7_20260716_212844}"
FOLLOWUP_ROOT="${FINAL13_FOLLOWUP_ROOT:-${ROOT}/benchmarking/results/final13_followup_gpu7_20260717_complete}"
OUTPUT_ROOT="${SYGRAPH_RESULT_ROOT:-${CORE_ROOT}/sygraph_bfs}"
MAIN_CSR_ROOT="${SYGRAPH_MAIN_CSR_ROOT:-${ROOT}/datasets/scaling/csr/sygraph-main}"
REFERENCE_JSON="${OUTPUT_ROOT}/sygraph_bfs_references.json"
RUNNER="${ROOT}/third_party/sygraph_build/bin/sygraph_bfs_runner"

ORKUT="${ROOT}/datasets/scaling/csr/com-Orkut/com-Orkut.json"
TWITTER="${ROOT}/datasets/scaling/csr/GAP-twitter/GAP-twitter.json"
RMAT_ROOT="${ROOT}/datasets/scaling/csr"
RMAT_RESULT="${FOLLOWUP_ROOT}/eggpu_rmat_four_functions"

mkdir -p "${OUTPUT_ROOT}"

echo "[SYgraph 1/5] prepare the 11 normalized main-dataset CSR inputs"
"${PYTHON_BIN}" "${ROOT}/benchmarking/prepare_sygraph_main_csr.py" \
  --eval-root "${ROOT}" \
  --output-root "${MAIN_CSR_ROOT}"

echo "[SYgraph 2/5] generate exact EGGPU BFS references for 13 main and 4 R-MAT graphs"
"${PYTHON_BIN}" "${ROOT}/benchmarking/generate_sygraph_bfs_references.py" \
  --main-result "${ROOT}/benchmarking/results/full_eval_gpu0_20260712_144427_repeat5_splitmem_exact_mst_strict_nxcg" \
  --core-result "${CORE_ROOT}" \
  --com-orkut-manifest "${ORKUT}" \
  --gap-twitter-manifest "${TWITTER}" \
  --rmat-result "${RMAT_RESULT}" \
  --rmat-manifest "${RMAT_ROOT}/R-MAT-S20-EF16/R-MAT-S20-EF16.json" \
  --rmat-manifest "${RMAT_ROOT}/R-MAT-S22-EF16/R-MAT-S22-EF16.json" \
  --rmat-manifest "${RMAT_ROOT}/R-MAT-S24-EF16/R-MAT-S24-EF16.json" \
  --rmat-manifest "${RMAT_ROOT}/R-MAT-S26-EF16/R-MAT-S26-EF16.json" \
  --output "${REFERENCE_JSON}"

echo "[SYgraph 3/5] build the unmodified upstream SYgraph BFS through project-local AdaptiveCpp"
SYGRAPH_SOURCE_COMMIT="${SOURCE_COMMIT}" \
bash "${ROOT}/benchmarking/build_sygraph_local.sh"

main_manifests=()
for dataset in \
  ca-HepTh LastFM p2p-Gnutella04 ca-HepPh email-Enron ca-CondMat \
  soc-Epinions1 soc-Slashdot0811 ER-100k web-NotreDame com-youtube; do
  main_manifests+=(--manifest "${MAIN_CSR_ROOT}/${dataset}/${dataset}.json")
done

scale_manifests=(
  --manifest "${ORKUT}"
  --manifest "${TWITTER}"
  --manifest "${RMAT_ROOT}/R-MAT-S20-EF16/R-MAT-S20-EF16.json"
  --manifest "${RMAT_ROOT}/R-MAT-S22-EF16/R-MAT-S22-EF16.json"
  --manifest "${RMAT_ROOT}/R-MAT-S24-EF16/R-MAT-S24-EF16.json"
  --manifest "${RMAT_ROOT}/R-MAT-S26-EF16/R-MAT-S26-EF16.json"
)

echo "[SYgraph 4/5] run native BFS with fresh-process timing and memory isolation"
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_sygraph_bfs_baseline.py" \
  "${main_manifests[@]}" \
  "${scale_manifests[@]}" \
  --binary "${RUNNER}" \
  --reference-json "${REFERENCE_JSON}" \
  --output-dir "${OUTPUT_ROOT}" \
  --gpu "${GPU}" \
  --repeat "${REPEAT}" \
  --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 8 \
  --timeout "${TIMEOUT}" \
  --load-timeout "${LOAD_TIMEOUT}" \
  --source-commit "${SOURCE_COMMIT}"

echo "[SYgraph 5/5] require every reported success to pass exact-result validation"
"${PYTHON_BIN}" -c '
import json, pathlib, sys
import pandas as pd
root = pathlib.Path(sys.argv[1])
data = pd.read_csv(root / "sygraph_bfs.csv")
invalid = data[data["status"].eq("ok") & ~data["validation"].eq("pass")]
if not invalid.empty:
    raise SystemExit(f"SYgraph contains {len(invalid)} unvalidated successful rows")
summary = {
    "datasets_attempted": int(len(data)),
    "validated_successes": int((data["status"].eq("ok") & data["validation"].eq("pass")).sum()),
    "status_counts": data["status"].value_counts().to_dict(),
    "failure_kinds": data.loc[~data["status"].eq("ok"), "failure_kind"].value_counts().to_dict(),
}
(root / "qualification_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
' "${OUTPUT_ROOT}"

echo "SYgraph qualification result: ${OUTPUT_ROOT}"
