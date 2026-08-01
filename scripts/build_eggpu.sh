#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PYTHON_BINDIR="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
PYTHON_PREFIX="$(cd "${PYTHON_BINDIR}/.." && pwd)"
export PATH="${PYTHON_BINDIR}:${PATH}"

CUDA_ROOT="${EGGPU_CUDA_ROOT:-${CUDA_PATH:-${CUDA_HOME:-${CUDAToolkit_ROOT:-${CONDA_PREFIX:-}}}}}"
if [[ -z "${CUDA_ROOT}" ]]; then
  echo "No CUDA root found. Set EGGPU_CUDA_ROOT or activate a CUDA-enabled conda environment." >&2
  exit 2
fi
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
  echo "nvcc not found at ${CUDA_ROOT}/bin/nvcc. Set EGGPU_CUDA_ROOT to a CUDA toolkit path." >&2
  exit 2
fi

export EASYGRAPH_ENABLE_GPU=TRUE
export EGGPU_CUDA_ROOT="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"
export CUDA_HOME="${CUDA_ROOT}"
export CUDAToolkit_ROOT="${CUDA_ROOT}"

if [[ -z "${CMAKE_GENERATOR:-}" && -x "${PYTHON_BINDIR}/ninja" ]]; then
  export CMAKE_GENERATOR=Ninja
fi

OPENMP_GOMP_LIBRARY=""
for candidate in \
    "${CUDA_ROOT}/lib/libgomp.so" \
    "${CUDA_ROOT}/targets/x86_64-linux/lib/libgomp.so" \
    "${PYTHON_PREFIX}/lib/libgomp.so"
do
  if [[ -f "${candidate}" ]]; then
    OPENMP_GOMP_LIBRARY="${candidate}"
    break
  fi
done

BUILD_ROOT="${ROOT}/Easy-Graph/build"
EXPECTED_BUILD_ROOT="${ROOT}/Easy-Graph/build"
if [[ "$(cd "$(dirname "${BUILD_ROOT}")" && pwd)/$(basename "${BUILD_ROOT}")" != "${EXPECTED_BUILD_ROOT}" ]]; then
  echo "Refusing to clean unexpected build path: ${BUILD_ROOT}" >&2
  exit 2
fi

clock_skew_truthy() {
  [[ "${1:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]
}

future_mtimes() {
  local scan_root="$1"
  local mode="$2"
  "${PYTHON_BIN}" - "${scan_root}" "${mode}" <<'PY'
from pathlib import Path
import sys
import time

root = Path(sys.argv[1])
mode = sys.argv[2]
now = time.time()
future = []

skip_dirs = {".git", "__pycache__"}
if mode == "source":
    skip_dirs.update({"build", ".eggs"})

for path in root.rglob("*"):
    parts = set(path.parts)
    if skip_dirs & parts:
        continue
    if mode == "source" and any(part.endswith(".egg-info") for part in path.parts):
        continue
    try:
        st = path.stat()
    except OSError:
        continue
    delta = st.st_mtime - now
    if delta > 60:
        future.append((delta, path))

for delta, path in sorted(future, reverse=True)[:20]:
    print(f"{delta:.0f}s {path}")
if len(future) > 20:
    print(f"... and {len(future) - 20} more")
PY
}

SOURCE_FUTURE_MTIMES="$(future_mtimes "${ROOT}/Easy-Graph" source)"
if [[ -n "${SOURCE_FUTURE_MTIMES}" ]]; then
  echo "Clock skew detected in Easy-Graph source files; build may be incomplete:" >&2
  echo "${SOURCE_FUTURE_MTIMES}" >&2
  if ! clock_skew_truthy "${EGGPU_ALLOW_CLOCK_SKEW:-}"; then
    echo "Refusing to build. Fix the machine clock or source mtimes, or set EGGPU_ALLOW_CLOCK_SKEW=1 for a debug-only build." >&2
    exit 2
  fi
  echo "Continuing despite source clock skew because EGGPU_ALLOW_CLOCK_SKEW=1." >&2
fi

if [[ -d "${BUILD_ROOT}" ]]; then
  BUILD_FUTURE_MTIMES="$(future_mtimes "${BUILD_ROOT}" build)"
  if [[ -n "${BUILD_FUTURE_MTIMES}" ]]; then
    echo "Removing CMake build cache with future timestamps:" >&2
    echo "${BUILD_FUTURE_MTIMES}" >&2
    rm -rf "${BUILD_ROOT}"
  fi
fi

if [[ "${EGGPU_CLEAN_BUILD:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  rm -rf "${BUILD_ROOT}"
elif [[ -d "${BUILD_ROOT}" ]]; then
  while IFS= read -r cache; do
    gomp_path="$(sed -n 's/^OpenMP_gomp_LIBRARY:FILEPATH=//p' "${cache}" | head -n 1)"
    make_program="$(sed -n 's/^CMAKE_MAKE_PROGRAM:FILEPATH=//p' "${cache}" | head -n 1)"
    if [[ -n "${gomp_path}" && ! -e "${gomp_path}" ]]; then
      echo "Removing stale CMake build cache: ${gomp_path} no longer exists"
      rm -rf "${BUILD_ROOT}"
      break
    fi
    if [[ -n "${make_program}" && ! -x "${make_program}" ]]; then
      echo "Removing stale CMake build cache: ${make_program} no longer exists or is not executable"
      rm -rf "${BUILD_ROOT}"
      break
    fi
  done < <(find "${BUILD_ROOT}" -name CMakeCache.txt -print)
fi

detect_cuda_architectures() {
  local requested="${EGGPU_CUDA_ARCHITECTURES:-${CMAKE_CUDA_ARCHITECTURES:-AUTO}}"
  if [[ -n "${requested}" && ! "${requested}" =~ ^([Aa][Uu][Tt][Oo])$ ]]; then
    printf '%s' "${requested}"
    return 0
  fi

  local detected=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    detected="$(
      nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
        | "${PYTHON_BIN}" -c '
import sys

seen = []
for raw in sys.stdin:
    cap = raw.strip().replace(" ", "")
    if not cap:
        continue
    digits = cap.replace(".", "")
    if digits.isdigit() and digits not in seen:
        seen.append(digits)
print(";".join(seen))
'
    )"
  fi

  if [[ -z "${detected}" ]]; then
    echo "Could not auto-detect CUDA architecture with nvidia-smi; defaulting to sm_80. Set EGGPU_CUDA_ARCHITECTURES explicitly for other GPUs." >&2
    detected="80"
  fi
  printf '%s' "${detected}"
}

CUDA_ARCHITECTURES="$(detect_cuda_architectures)"
echo "Using CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"

CMAKE_ARG_LIST=(
  "-DCUDAToolkit_ROOT=${CUDA_ROOT}"
  "-DCMAKE_CUDA_COMPILER=${CUDA_ROOT}/bin/nvcc"
  "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
)
if [[ -n "${OPENMP_GOMP_LIBRARY}" ]]; then
  CMAKE_ARG_LIST+=("-DOpenMP_gomp_LIBRARY=${OPENMP_GOMP_LIBRARY}")
fi
CMAKE_ARGS_JOINED="${CMAKE_ARG_LIST[*]}"

if [[ "${EGGPU_BUILD_DRY_RUN:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  printf 'CMAKE_ARGS=%s\n' "${CMAKE_ARGS_JOINED}"
  exit 0
fi

cd "${ROOT}/Easy-Graph"
env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
    -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LD_LIBRARY_PATH \
    EASYGRAPH_ENABLE_GPU=TRUE \
    EGGPU_CUDA_ROOT="${CUDA_ROOT}" \
    CUDA_PATH="${CUDA_ROOT}" \
    CUDA_HOME="${CUDA_ROOT}" \
    CUDAToolkit_ROOT="${CUDA_ROOT}" \
    CMAKE_ARGS="${CMAKE_ARGS_JOINED}" \
    "${PYTHON_BIN}" setup.py build_ext --inplace
