#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${SYGRAPH_ENV_PREFIX:-${EVAL_ROOT}/third_party/sygraph_env}"
PKG_CACHE="${SYGRAPH_CONDA_PKGS_DIRS:-${EVAL_ROOT}/third_party/.conda_pkgs}"
SOURCE_ROOT="${EVAL_ROOT}/third_party/SYgraph"
BUILD_ROOT="${EVAL_ROOT}/third_party/sygraph_build"
OVERLAY_INCLUDE="${BUILD_ROOT}/overlay/include"
ACCESSOR_PATCH="${EVAL_ROOT}/benchmarking/sygraph_result_accessors.patch"
CUDA_ROOT="${EGGPU_CUDA_ROOT:-/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang}"
CUDA_VIEW="${BUILD_ROOT}/cuda_toolkit_view"
ACPP_SPEC="${SYGRAPH_ADAPTIVECPP_SPEC:-adaptivecpp=25.10.0=cuda129_llvm17_h985edc5_6}"
CLANGXX_SPEC="${SYGRAPH_CLANGXX_SPEC:-clangxx=17}"

mkdir -p "${PKG_CACHE}" "${BUILD_ROOT}/bin" "${OVERLAY_INCLUDE}"
if [[ ! -x "${PREFIX}/bin/acpp" ]]; then
  CONDA_PKGS_DIRS="${PKG_CACHE}" /opt/conda/bin/mamba create -y -p "${PREFIX}" \
    -c conda-forge "${ACPP_SPEC}" "${CLANGXX_SPEC}" openssl
elif ! compgen -G "${PREFIX}/conda-meta/adaptivecpp-25.10.0-*.json" >/dev/null; then
  CONDA_PKGS_DIRS="${PKG_CACHE}" /opt/conda/bin/mamba install -y -p "${PREFIX}" \
    -c conda-forge "${ACPP_SPEC}" "${CLANGXX_SPEC}"
elif [[ ! -x "${PREFIX}/bin/clang++" ]]; then
  CONDA_PKGS_DIRS="${PKG_CACHE}" /opt/conda/bin/mamba install -y -p "${PREFIX}" \
    -c conda-forge "${CLANGXX_SPEC}"
fi

# Conda CUDA keeps headers and libraries below targets/x86_64-linux, while
# clang's CUDA discovery expects the conventional toolkit layout.
mkdir -p "${CUDA_VIEW}"
ln -sfn "${CUDA_ROOT}/bin" "${CUDA_VIEW}/bin"
ln -sfn "${CUDA_ROOT}/targets/x86_64-linux/include" "${CUDA_VIEW}/include"
ln -sfn "${CUDA_ROOT}/targets/x86_64-linux/lib" "${CUDA_VIEW}/lib64"
ln -sfn "${CUDA_ROOT}/nvvm" "${CUDA_VIEW}/nvvm"

# Keep the upstream checkout byte-for-byte clean.  The overlay exposes bulk
# SSSP/BC results and fixes SSSP's finite-distance sentinel for arbitrary
# nonnegative weights; it does not replace either algorithm.
cp -a "${SOURCE_ROOT}/include/." "${OVERLAY_INCLUDE}/"
patch --batch --forward -p1 -d "${BUILD_ROOT}/overlay" < "${ACCESSOR_PATCH}"

"${PREFIX}/bin/acpp" \
  --acpp-targets="cuda:sm_80" \
  --acpp-cuda-path="${CUDA_VIEW}" \
  -O3 -DNDEBUG -std=c++20 -DSYCL_EXTERNAL= -DBITMAP_SIZE=32 -DCU_SIZE=512 \
  -I"${OVERLAY_INCLUDE}" -I"${PREFIX}/include" \
  "${EVAL_ROOT}/benchmarking/sygraph_bfs_runner.cpp" \
  -L"${PREFIX}/lib" -Wl,-rpath,"${PREFIX}/lib" -lcrypto \
  -o "${BUILD_ROOT}/bin/sygraph_bfs_runner"

"${PREFIX}/bin/acpp" \
  --acpp-targets="cuda:sm_80" \
  --acpp-cuda-path="${CUDA_VIEW}" \
  -O3 -DNDEBUG -std=c++20 -DSYCL_EXTERNAL= -DSYGRAPH_RUNNER_SSSP=1 -DBITMAP_SIZE=32 -DCU_SIZE=512 \
  -I"${OVERLAY_INCLUDE}" -I"${PREFIX}/include" \
  "${EVAL_ROOT}/benchmarking/sygraph_multi_runner.cpp" \
  -L"${PREFIX}/lib" -Wl,-rpath,"${PREFIX}/lib" -lcrypto \
  -o "${BUILD_ROOT}/bin/sygraph_sssp_runner"

"${PREFIX}/bin/acpp" \
  --acpp-targets="cuda:sm_80" \
  --acpp-cuda-path="${CUDA_VIEW}" \
  -O3 -DNDEBUG -std=c++20 -DSYCL_EXTERNAL= -DSYGRAPH_RUNNER_BC=1 -DBITMAP_SIZE=32 -DCU_SIZE=512 \
  -I"${OVERLAY_INCLUDE}" -I"${PREFIX}/include" \
  "${EVAL_ROOT}/benchmarking/sygraph_multi_runner.cpp" \
  -L"${PREFIX}/lib" -Wl,-rpath,"${PREFIX}/lib" -lcrypto \
  -o "${BUILD_ROOT}/bin/sygraph_bc_runner"

SOURCE_COMMIT="${SYGRAPH_SOURCE_COMMIT:-71dcb56aed0cd43fefa5752a552e0a10cb69a758}"
PATCH_SHA256="$(sha256sum "${ACCESSOR_PATCH}" | awk '{print $1}')"
cat > "${BUILD_ROOT}/build_manifest.json" <<EOF
{
  "adaptivecpp_spec": "${ACPP_SPEC}",
  "clangxx_spec": "${CLANGXX_SPEC}",
  "compiler": "${PREFIX}/bin/acpp",
  "cuda_root": "${CUDA_ROOT}",
  "cuda_toolkit_view": "${CUDA_VIEW}",
  "source_commit": "${SOURCE_COMMIT}",
  "source_remote": "https://github.com/unisa-hpc/SYgraph",
  "target": "cuda:sm_80",
  "upstream_checkout_modified": false,
  "overlay_patch": "benchmarking/sygraph_result_accessors.patch",
  "overlay_patch_sha256": "${PATCH_SHA256}",
  "overlay_scope": ["SSSP bulk result accessor", "SSSP infinity sentinel", "BC bulk result accessor", "AdaptiveCpp standard device query compatibility", "AdaptiveCpp SYCL 2020 reduction compatibility", "MLB frontier counter reset before multi-work-group compaction"],
  "wrappers": ["benchmarking/sygraph_bfs_runner.cpp", "benchmarking/sygraph_multi_runner.cpp"],
  "binaries": ["sygraph_bfs_runner", "sygraph_sssp_runner", "sygraph_bc_runner"]
}
EOF

echo "Built ${BUILD_ROOT}/bin/sygraph_bfs_runner"
echo "Built ${BUILD_ROOT}/bin/sygraph_sssp_runner"
echo "Built ${BUILD_ROOT}/bin/sygraph_bc_runner"
