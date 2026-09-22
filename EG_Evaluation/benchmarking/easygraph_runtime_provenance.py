"""Freeze and fingerprint the EasyGraph runtime used by auxiliary benchmarks."""

from __future__ import annotations

import hashlib
import importlib.machinery
import os
import sys
from pathlib import Path

from run_full_baselines import collect_runtime_python_snapshot


RUNTIME_MODULES = ("easygraph", "cpp_easygraph")
EXPLICIT_ENV_KEYS = {
    "BLIS_NUM_THREADS",
    "CONDA_PREFIX",
    "LD_LIBRARY_PATH",
    "MALLOC_ARENA_MAX",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "OMP_PLACES",
    "OMP_PROC_BIND",
    "OPENBLAS_NUM_THREADS",
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "VECLIB_MAXIMUM_THREADS",
}
ENV_PREFIXES = ("CUDA_", "EASYGRAPH_", "EGGPU_")


def _resolved_repo(path):
    requested = Path(path).expanduser().absolute()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"EasyGraph runtime does not exist: {requested}") from exc
    if not resolved.is_dir():
        raise RuntimeError(f"EasyGraph runtime is not a directory: {resolved}")
    return requested, resolved


def _assert_within_runtime(origin, runtime_root, module_name):
    resolved = Path(origin).resolve()
    try:
        resolved.relative_to(runtime_root)
    except ValueError as exc:
        raise RuntimeError(
            f"{module_name} resolved outside --easygraph-repo: "
            f"{resolved} (runtime root: {runtime_root})"
        ) from exc
    return resolved


def _artifact(origin, runtime_root, module_name):
    path = Path(origin)
    resolved = _assert_within_runtime(path, runtime_root, module_name)
    if not resolved.is_file():
        raise RuntimeError(f"{module_name} origin is not a file: {resolved}")
    payload = resolved.read_bytes()
    return {
        "module_origin": str(path.absolute()),
        "module_origin_resolved": str(resolved),
        "relative_to_runtime": str(resolved.relative_to(runtime_root)),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "is_native_extension": any(
            str(resolved).endswith(suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        ),
    }


def _repository_module_artifact(runtime_root, module_name):
    spec = importlib.machinery.PathFinder.find_spec(
        module_name, [str(runtime_root)]
    )
    origin = getattr(spec, "origin", "") if spec is not None else ""
    if not origin:
        raise RuntimeError(
            f"{module_name} is not importable directly from --easygraph-repo "
            f"{runtime_root}"
        )
    return _artifact(origin, runtime_root, module_name)


def collect_runtime_repository_provenance(easygraph_repo):
    """Fingerprint the requested package tree and directly discoverable modules."""

    requested, runtime_root = _resolved_repo(easygraph_repo)
    modules = {
        name: _repository_module_artifact(runtime_root, name)
        for name in RUNTIME_MODULES
    }
    if not modules["cpp_easygraph"]["is_native_extension"]:
        raise RuntimeError(
            "cpp_easygraph from --easygraph-repo is not a native extension: "
            f"{modules['cpp_easygraph']['module_origin_resolved']}"
        )
    return {
        "requested_root": str(requested),
        "resolved_root": str(runtime_root),
        "runtime_python_snapshot": collect_runtime_python_snapshot(requested),
        "modules": modules,
        "native_sha256": modules["cpp_easygraph"]["sha256"],
    }


def install_runtime_import_root(easygraph_repo):
    """Make the requested runtime first in sys.path and reject prior drift."""

    _requested, runtime_root = _resolved_repo(easygraph_repo)
    for module_name in RUNTIME_MODULES:
        loaded = sys.modules.get(module_name)
        origin = getattr(loaded, "__file__", "") if loaded is not None else ""
        if origin:
            _assert_within_runtime(origin, runtime_root, module_name)

    retained = []
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == runtime_root:
                continue
        except OSError:
            pass
        retained.append(entry)
    sys.path[:] = [str(runtime_root), *retained]
    return runtime_root


def collect_loaded_runtime_provenance(easygraph_repo):
    """Fingerprint the modules actually imported by the running benchmark."""

    _requested, runtime_root = _resolved_repo(easygraph_repo)
    modules = {}
    for module_name in RUNTIME_MODULES:
        loaded = sys.modules.get(module_name)
        origin = getattr(loaded, "__file__", "") if loaded is not None else ""
        if not origin:
            raise RuntimeError(
                f"{module_name} was not loaded while collecting runtime provenance"
            )
        modules[module_name] = _artifact(origin, runtime_root, module_name)
    return {
        "resolved_root": str(runtime_root),
        "modules": modules,
        "native_sha256": modules["cpp_easygraph"]["sha256"],
    }


def collect_relevant_environment(environment=None):
    """Record runtime controls without copying unrelated process environment."""

    source = os.environ if environment is None else environment
    return {
        key: source.get(key, "")
        for key in sorted(source)
        if key in EXPLICIT_ENV_KEYS or key.startswith(ENV_PREFIXES)
    }
