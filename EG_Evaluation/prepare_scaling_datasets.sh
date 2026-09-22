#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
DOWNLOAD_DIR="${ROOT}/datasets/scaling/downloads"
CSR_DIR="${ROOT}/datasets/scaling/csr"
CONVERTER_SRC="${ROOT}/benchmarking/tools/edge_list_to_eggpu_csr.cpp"
CONVERTER_BIN="${ROOT}/benchmarking/tools/edge_list_to_eggpu_csr"

mkdir -p "${DOWNLOAD_DIR}" "${CSR_DIR}"

if [[ ! -x "${CONVERTER_BIN}" || "${CONVERTER_SRC}" -nt "${CONVERTER_BIN}" ]]; then
  g++ -O3 -std=c++17 -DNDEBUG "${CONVERTER_SRC}" -o "${CONVERTER_BIN}"
fi

GATE_TXT="${ROOT}/datasets/undirected/com-youtube.ungraph.txt"
GATE_OUT="${CSR_DIR}/com-youtube"
mkdir -p "${GATE_OUT}"
if [[ ! -s "${GATE_TXT}" ]]; then
  echo "missing frozen performance-gate dataset: ${GATE_TXT}" >&2
  exit 2
fi
if [[ ! -s "${GATE_OUT}/com-youtube.json" ]]; then
  /usr/bin/time -v -o "${GATE_OUT}/conversion_resource_usage.txt" \
    "${CONVERTER_BIN}" \
    --input "${GATE_TXT}" --output-dir "${GATE_OUT}" \
    --name com-youtube --format edge-list --num-nodes 1134890 \
    --expected-edges 2987624 --input-base 0 --mirror-undirected --relabel \
    --source-url "https://snap.stanford.edu/data/com-Youtube.html"
fi
"${PYTHON_BIN}" "${ROOT}/benchmarking/finalize_scaling_manifest.py" \
  "${GATE_OUT}/com-youtube.json" \
  --plain-source "${GATE_TXT}" \
  --download-url "https://snap.stanford.edu/data/com-Youtube.html" \
  --catalog-url "https://snap.stanford.edu/data/com-Youtube.html" \
  --raw-edge-records 2987624 --unique-undirected-edges 2987624 \
  --notes "Formal positive-performance gate derived from the frozen main-benchmark simple graph."

download_exact() {
  local url="$1"
  local output="$2"
  local expected_bytes="$3"
  if [[ -s "${output}" ]]; then
    local existing_bytes
    existing_bytes="$(stat -c %s "${output}")"
    if [[ "${existing_bytes}" == "${expected_bytes}" ]]; then
      echo "Reusing byte-exact download: ${output} (${existing_bytes} bytes)"
      return 0
    fi
  fi
  aria2c --continue=true --allow-overwrite=true --auto-file-renaming=false \
    --file-allocation=none --max-connection-per-server=16 --split=16 \
    --min-split-size=1M --dir="$(dirname "${output}")" \
    --out="$(basename "${output}")" "${url}"
  local actual_bytes
  actual_bytes="$(stat -c %s "${output}")"
  if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
    echo "size mismatch for ${output}: expected ${expected_bytes}, got ${actual_bytes}" >&2
    exit 2
  fi
}

ORKUT_GZ="${DOWNLOAD_DIR}/com-orkut.ungraph.txt.gz"
ORKUT_TXT="${DOWNLOAD_DIR}/com-orkut.ungraph.txt"
ORKUT_OUT="${CSR_DIR}/com-Orkut"
mkdir -p "${ORKUT_OUT}"
download_exact \
  "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz" \
  "${ORKUT_GZ}" 447251958
gzip -t "${ORKUT_GZ}"
if [[ ! -s "${ORKUT_TXT}" ]]; then
  gzip -dc "${ORKUT_GZ}" > "${ORKUT_TXT}.tmp"
  mv "${ORKUT_TXT}.tmp" "${ORKUT_TXT}"
fi
if [[ ! -s "${ORKUT_OUT}/com-Orkut.json" ]]; then
  /usr/bin/time -v -o "${ORKUT_OUT}/conversion_resource_usage.txt" \
    "${CONVERTER_BIN}" \
    --input "${ORKUT_TXT}" --output-dir "${ORKUT_OUT}" \
    --name com-Orkut --format edge-list --num-nodes 3072441 \
    --expected-edges 117185083 --input-base 0 --mirror-undirected --relabel \
    --source-url "https://snap.stanford.edu/data/com-Orkut.html"
fi
"${PYTHON_BIN}" "${ROOT}/benchmarking/finalize_scaling_manifest.py" \
  "${ORKUT_OUT}/com-Orkut.json" \
  --compressed-source "${ORKUT_GZ}" --plain-source "${ORKUT_TXT}" \
  --download-url "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz" \
  --catalog-url "https://snap.stanford.edu/data/com-Orkut.html" \
  --raw-edge-records 117185083 --unique-undirected-edges 117185083 \
  --notes "SNAP largest connected component; source labels are deterministically sorted and remapped to 0..N-1."

TWITTER_TAR="${DOWNLOAD_DIR}/GAP-twitter.tar.gz"
TWITTER_DIR="${DOWNLOAD_DIR}/GAP-twitter"
TWITTER_MTX="${TWITTER_DIR}/GAP-twitter.mtx"
TWITTER_OUT="${CSR_DIR}/GAP-twitter"
mkdir -p "${TWITTER_OUT}"
download_exact \
  "http://sparse-files.engr.tamu.edu/MM/GAP/GAP-twitter.tar.gz" \
  "${TWITTER_TAR}" 9469180207
gzip -t "${TWITTER_TAR}"
mkdir -p "${TWITTER_DIR}"
if [[ ! -s "${TWITTER_MTX}" ]]; then
  # The archive also contains GAP-twitter_sources.mtx.  Selecting the first
  # *.mtx member silently extracts the source list instead of the graph.
  TWITTER_MEMBER="GAP-twitter/GAP-twitter.mtx"
  tar -xOzf "${TWITTER_TAR}" "${TWITTER_MEMBER}" > "${TWITTER_MTX}.tmp"
  TWITTER_DIMS="$(awk '!/^%/ {print $1, $2, $3; exit}' "${TWITTER_MTX}.tmp")"
  if [[ "${TWITTER_DIMS}" != "61578415 61578415 1468364884" ]]; then
    echo "unexpected GAP-twitter Matrix Market dimensions: ${TWITTER_DIMS}" >&2
    rm -f "${TWITTER_MTX}.tmp"
    exit 2
  fi
  mv "${TWITTER_MTX}.tmp" "${TWITTER_MTX}"
fi
if [[ ! -s "${TWITTER_OUT}/GAP-twitter.json" ]]; then
  /usr/bin/time -v -o "${TWITTER_OUT}/conversion_resource_usage.txt" \
    "${CONVERTER_BIN}" \
    --input "${TWITTER_MTX}" --output-dir "${TWITTER_OUT}" \
    --name GAP-twitter --format matrix-market --num-nodes 61578415 \
    --expected-edges 1468364884 --input-base 1 --directed \
    --source-url "https://sparse.tamu.edu/GAP/GAP-twitter"
fi

GAP_SOURCES="12441073,54488258,25451916,57714474,14839495,32081105,52957358,50444381,49590702,20127817,34939334,48251002,19524254,43676727,33055509,15244688,24946739,6479473,26077683,22023876,22081916,40034163,49496015,42847508,52409558,55445389,22028098,48766649,44521242,60135543,28528672,9678013,40020307,31625736,37446893,51788953,52584256,20346697,48387910,37337428,50501085,30130062,41185894,56495704,45663306,33359461,48143059,33291514,53461446,29340611,34148499,49171807,35550697,14521508,51633219,46823383,19396274,19871751,36862678,49539127,34016453,36567396,55487794,14391371"
"${PYTHON_BIN}" "${ROOT}/benchmarking/finalize_scaling_manifest.py" \
  "${TWITTER_OUT}/GAP-twitter.json" \
  --compressed-source "${TWITTER_TAR}" --plain-source "${TWITTER_MTX}" \
  --download-url "http://sparse-files.engr.tamu.edu/MM/GAP/GAP-twitter.tar.gz" \
  --catalog-url "https://sparse.tamu.edu/GAP/GAP-twitter" \
  --raw-edge-records 1468364884 --simple-directed-edges 1468364884 \
  --benchmark-sources-one-based "${GAP_SOURCES}" \
  --notes "All 61,578,415 matrix rows are retained, including 19,926,185 isolates. The scaling study uses the directed topology with implicit unit weights; supplied random weights are outside this topology-scaling claim."

echo "Prepared ${GATE_OUT}/com-youtube.json"
echo "Prepared ${ORKUT_OUT}/com-Orkut.json"
echo "Prepared ${TWITTER_OUT}/GAP-twitter.json"
