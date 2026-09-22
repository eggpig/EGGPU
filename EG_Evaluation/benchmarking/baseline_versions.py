"""Version provenance shared by benchmark parents and child runners."""

import hashlib
from importlib import metadata
from importlib import util as importlib_util
import platform
import sys
from pathlib import Path


def _module_location(module_name):
    try:
        spec = importlib_util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is None:
        return {"module": module_name, "module_origin": "not-found"}
    locations = [str(path) for path in (spec.submodule_search_locations or [])]
    return {
        "module": module_name,
        "module_origin": str(spec.origin or "namespace-package"),
        "module_search_locations": locations,
    }


def _distribution_entry(distribution_names, role, module_name):
    for name in distribution_names:
        try:
            return {
                "distribution": name,
                "version": metadata.version(name),
                "role": role,
                **_module_location(module_name),
            }
        except metadata.PackageNotFoundError:
            continue
    return {
        "distribution": distribution_names[0],
        "version": "not-installed",
        "role": role,
        **_module_location(module_name),
    }


def _cpp_easygraph_location():
    resolved = _module_location("cpp_easygraph")
    if str(resolved.get("module_origin", "")).endswith(".so"):
        path = Path(resolved["module_origin"])
        if path.is_file():
            resolved.update(
                {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        resolved["resolution"] = "python-import-spec"
        return resolved

    easygraph_root = Path(__file__).resolve().parents[2] / "Easy-Graph"
    candidates = []
    for pattern in ("cpp_easygraph*.so", "build/lib*/cpp_easygraph*.so"):
        candidates.extend(easygraph_root.glob(pattern))
    candidates = sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=lambda path: ("/build/" in str(path), str(path)),
    )
    if not candidates:
        return resolved
    selected = candidates[0]
    return {
        "module": "cpp_easygraph",
        "module_origin": str(selected),
        "module_search_locations": [],
        "resolution": "workspace-artifact-discovery",
        "import_spec_origin": resolved.get("module_origin", "not-found"),
        "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "size_bytes": selected.stat().st_size,
    }


def collect_python_baseline_versions():
    versions = {
        "easygraph": _distribution_entry(
            ("Python-EasyGraph", "easygraph"),
            "workspace EasyGraph source used by EGGPU, easygraph-cpu, and easygraph-cpp",
            "easygraph",
        ),
        "networkx": _distribution_entry(
            ("networkx",), "evaluated CPU baseline", "networkx"
        ),
        "igraph": _distribution_entry(
            ("igraph", "python-igraph"), "evaluated CPU baseline", "igraph"
        ),
        "nx-cugraph": _distribution_entry(
            ("nx-cugraph", "nx_cugraph"),
            "evaluated NetworkX backend baseline with fallback forbidden",
            "nx_cugraph",
        ),
        "cpp_easygraph": {
            "version": "workspace-extension",
            "role": "compiled extension used by EGGPU and easygraph-cpp",
            **_cpp_easygraph_location(),
        },
    }
    versions["evaluated_baselines"] = [
        "EGGPU",
        "easygraph-cpu",
        "easygraph-cpp",
        "networkx",
        "igraph",
        "nx-cugraph",
        "Gunrock",
    ]
    versions["runtime_dependencies"] = {
        "cugraph": _distribution_entry(
            ("cugraph",),
            "nx-cugraph runtime dependency; not an evaluated baseline",
            "cugraph",
        ),
        "cudf": _distribution_entry(
            ("cudf",),
            "nx-cugraph runtime dependency; not an evaluated baseline",
            "cudf",
        ),
        "numpy": _distribution_entry(("numpy",), "benchmark dependency", "numpy"),
        "pandas": _distribution_entry(("pandas",), "benchmark dependency", "pandas"),
        "scipy": _distribution_entry(("scipy",), "benchmark dependency", "scipy"),
    }
    versions["python"] = {
        "version": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    return versions
