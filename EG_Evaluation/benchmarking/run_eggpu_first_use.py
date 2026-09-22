#!/usr/bin/env python3
"""Run and compare EGGPU first-use and steady-state protocols."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from easygraph_runtime_provenance import (
    collect_runtime_repository_provenance,
    install_runtime_import_root,
)


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_METRICS = {"e2e", "kernel"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def plus_minus(mean, std) -> str:
    mean_value = as_float(mean)
    std_value = as_float(std)
    if mean_value is None:
        return ""
    if std_value is None:
        return f"{mean_value:.6g}"
    return f"{mean_value:.6g} +/- {std_value:.3g}"


def _required_digest(value, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise SystemExit(
            f"{label} is missing or is not a SHA-256 digest: "
            f"{digest or 'missing'}"
        )
    return digest


def _absolute_resolved_path(value, label: str) -> Path:
    path = Path(str(value or "")).expanduser()
    if not str(value or "") or not path.is_absolute():
        raise SystemExit(f"{label} must be an absolute path: {value or 'missing'}")
    return path.resolve()


def _module_origin(value, runtime_root: Path, label: str) -> str:
    origin = _absolute_resolved_path(value, label)
    try:
        origin.relative_to(runtime_root)
    except ValueError as exc:
        raise SystemExit(
            f"{label} resolved outside frozen --easygraph-repo: "
            f"{origin} (runtime root: {runtime_root})"
        ) from exc
    return str(origin)


def prepare_frozen_runtime(easygraph_repo) -> tuple[Path, dict]:
    """Install and fingerprint one explicit, non-symlinked runtime tree.

    This runs before any EasyGraph import in ``main``.  The helper also rejects
    an EasyGraph/cpp_easygraph module that was already imported from elsewhere.
    """

    requested = Path(str(easygraph_repo)).expanduser()
    if not requested.is_absolute():
        raise SystemExit(
            "--easygraph-repo must be an absolute frozen runtime path: "
            f"{easygraph_repo}"
        )
    try:
        runtime_root = install_runtime_import_root(requested)
        provenance = collect_runtime_repository_provenance(runtime_root)
    except RuntimeError as exc:
        raise SystemExit(f"invalid frozen --easygraph-repo: {exc}") from exc

    snapshot = provenance.get("runtime_python_snapshot") or {}
    if snapshot.get("package_is_symlink") is not False:
        raise SystemExit(
            "invalid frozen --easygraph-repo: easygraph package_is_symlink "
            "must be false"
        )
    return runtime_root, provenance


def prepend_runtime_pythonpath(environment: dict[str, str], runtime_root: Path) -> None:
    """Prioritize the frozen runtime in every benchmark subprocess."""

    root = str(runtime_root)
    retained = [
        entry
        for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
        if entry and str(Path(entry).expanduser().resolve()) != root
    ]
    environment["PYTHONPATH"] = os.pathsep.join([root, *retained])


def _metadata_runtime_identity(metadata: dict, label: str) -> dict:
    runtime_root = _absolute_resolved_path(
        metadata.get("easygraph_repo"), f"{label} easygraph_repo"
    )
    implementation_digest = _required_digest(
        (metadata.get("implementation_source_snapshot") or {}).get("digest"),
        f"{label} implementation_source_snapshot.digest",
    )

    runtime_snapshot = metadata.get("runtime_python_snapshot") or {}
    runtime_python_digest = _required_digest(
        runtime_snapshot.get("digest"),
        f"{label} runtime_python_snapshot.digest",
    )
    if runtime_snapshot.get("package_is_symlink") is not False:
        raise SystemExit(
            f"{label} runtime_python_snapshot.package_is_symlink must be false"
        )
    recorded_runtime_root = _absolute_resolved_path(
        runtime_snapshot.get("runtime_root"),
        f"{label} runtime_python_snapshot.runtime_root",
    )
    if recorded_runtime_root != runtime_root:
        raise SystemExit(
            f"{label} runtime root mismatch: metadata={runtime_root} "
            f"snapshot={recorded_runtime_root}"
        )

    package_path = _absolute_resolved_path(
        runtime_snapshot.get("package_path"),
        f"{label} runtime_python_snapshot.package_path",
    )
    package_resolved_path = _absolute_resolved_path(
        runtime_snapshot.get("package_resolved_path"),
        f"{label} runtime_python_snapshot.package_resolved_path",
    )
    expected_package_path = runtime_root / "easygraph"
    if (
        package_path != expected_package_path
        or package_resolved_path != expected_package_path
    ):
        raise SystemExit(
            f"{label} EasyGraph package does not belong to the frozen runtime: "
            f"package={package_path} resolved={package_resolved_path} "
            f"expected={expected_package_path}"
        )

    build_artifacts = metadata.get("build_artifacts") or {}
    active_native = build_artifacts.get("active_cpp_easygraph") or {}
    native_sha = _required_digest(
        active_native.get("sha256"),
        f"{label} build_artifacts.active_cpp_easygraph.sha256",
    )
    active_native_origin = _module_origin(
        active_native.get("path"),
        runtime_root,
        f"{label} build_artifacts.active_cpp_easygraph.path",
    )

    versions = metadata.get("baseline_versions") or {}
    easygraph_version = versions.get("easygraph") or {}
    cpp_version = versions.get("cpp_easygraph") or {}
    easygraph_origin = _module_origin(
        easygraph_version.get("module_origin"),
        runtime_root,
        f"{label} baseline_versions.easygraph.module_origin",
    )
    cpp_origin = _module_origin(
        cpp_version.get("module_origin"),
        runtime_root,
        f"{label} baseline_versions.cpp_easygraph.module_origin",
    )
    version_native_sha = _required_digest(
        cpp_version.get("sha256"),
        f"{label} baseline_versions.cpp_easygraph.sha256",
    )
    if version_native_sha != native_sha:
        raise SystemExit(
            f"{label} active native SHA differs from the imported module SHA: "
            f"active={native_sha} module={version_native_sha}"
        )
    if cpp_origin != active_native_origin:
        raise SystemExit(
            f"{label} active native path differs from cpp_easygraph module origin: "
            f"active={active_native_origin} module={cpp_origin}"
        )
    try:
        Path(easygraph_origin).relative_to(expected_package_path)
    except ValueError as exc:
        raise SystemExit(
            f"{label} easygraph module origin is outside the frozen package: "
            f"{easygraph_origin}"
        ) from exc

    return {
        "runtime_root": str(runtime_root),
        "implementation_source_digest": implementation_digest,
        "runtime_python_digest": runtime_python_digest,
        "package_is_symlink": False,
        "active_native_sha256": native_sha,
        "module_origins": {
            "easygraph": easygraph_origin,
            "cpp_easygraph": cpp_origin,
        },
    }


def _requested_runtime_identity(provenance: dict) -> dict:
    snapshot = provenance.get("runtime_python_snapshot") or {}
    modules = provenance.get("modules") or {}
    return {
        "runtime_root": str(
            _absolute_resolved_path(
                provenance.get("resolved_root"), "requested runtime resolved_root"
            )
        ),
        "runtime_python_digest": _required_digest(
            snapshot.get("digest"), "requested runtime Python digest"
        ),
        "package_is_symlink": snapshot.get("package_is_symlink"),
        "active_native_sha256": _required_digest(
            provenance.get("native_sha256"), "requested runtime native SHA"
        ),
        "module_origins": {
            module_name: _module_origin(
                (modules.get(module_name) or {}).get("module_origin_resolved"),
                _absolute_resolved_path(
                    provenance.get("resolved_root"), "requested runtime resolved_root"
                ),
                f"requested runtime {module_name} module origin",
            )
            for module_name in ("easygraph", "cpp_easygraph")
        },
    }


def verify_same_implementation(
    steady_dir: Path,
    first_use_dir: Path,
    expected_runtime_root: Path | None = None,
    requested_runtime_provenance: dict | None = None,
) -> tuple[str, list[str]]:
    steady = json.loads((steady_dir / "run_metadata.json").read_text())
    first_use = json.loads((first_use_dir / "run_metadata.json").read_text())
    steady_snapshot = (steady.get("source_snapshot") or {}).get("digest", "")
    first_snapshot = (first_use.get("source_snapshot") or {}).get("digest", "")
    steady_identity = _metadata_runtime_identity(steady, "steady")
    first_identity = _metadata_runtime_identity(first_use, "first-use")
    full_snapshot_match = bool(steady_snapshot and steady_snapshot == first_snapshot)
    compared_fields = (
        "runtime_root",
        "implementation_source_digest",
        "runtime_python_digest",
        "package_is_symlink",
        "active_native_sha256",
        "module_origins",
    )
    mismatches = {
        field: {
            "steady": steady_identity[field],
            "first_use": first_identity[field],
        }
        for field in compared_fields
        if steady_identity[field] != first_identity[field]
    }
    if mismatches:
        raise SystemExit(
            "steady/first-use frozen runtime identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    if expected_runtime_root is not None:
        expected_root = _absolute_resolved_path(
            expected_runtime_root, "expected frozen runtime"
        )
        if steady_identity["runtime_root"] != str(expected_root):
            raise SystemExit(
                "steady/first-use metadata does not use requested frozen runtime: "
                f"recorded={steady_identity['runtime_root']} requested={expected_root}"
            )

    requested_identity = None
    if requested_runtime_provenance is not None:
        requested_identity = _requested_runtime_identity(
            requested_runtime_provenance
        )
        for field in (
            "runtime_root",
            "runtime_python_digest",
            "package_is_symlink",
            "active_native_sha256",
            "module_origins",
        ):
            if steady_identity[field] != requested_identity[field]:
                raise SystemExit(
                    "recorded metadata differs from the active frozen runtime: "
                    f"field={field} recorded={steady_identity[field]} "
                    f"active={requested_identity[field]}"
                )

    verification = {
        "status": "pass",
        "mode": "v10_frozen_runtime_identity_match",
        "steady_full_source_snapshot": steady_snapshot,
        "first_use_full_source_snapshot": first_snapshot,
        "full_source_snapshot_match": full_snapshot_match,
        "steady_implementation_source_snapshot": steady.get(
            "implementation_source_snapshot"
        ),
        "first_use_implementation_source_snapshot": first_use.get(
            "implementation_source_snapshot"
        ),
        "frozen_runtime_identity": first_identity,
        "active_requested_runtime_identity": requested_identity,
        "note": (
            "The full source snapshot may differ after benchmark-harness-only "
            "changes. The implementation source, active extension, Python runtime, "
            "non-symlink package, and both module origins are identical."
        ),
    }
    (first_use_dir / "implementation_compatibility.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )
    return steady_snapshot, [steady_identity["active_native_sha256"]]


def eligible(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    selected = {}
    for row in rows:
        if row.get("baseline") != "EGGPU" or row.get("metric") not in COMPARISON_METRICS:
            continue
        if row.get("status") != "ok" or row.get("publishable", "true").lower() != "true":
            continue
        selected[(row.get("dataset", ""), row.get("function", ""), row.get("metric", ""))] = row
    return selected


def comparison_rows(steady_dir: Path, first_use_dir: Path) -> list[dict[str, object]]:
    steady = eligible(read_csv(steady_dir / "results_long.csv"))
    first_use = eligible(read_csv(first_use_dir / "results_long.csv"))
    rows: list[dict[str, object]] = []
    for key in sorted(set(steady) & set(first_use)):
        dataset, function, metric = key
        steady_row = steady[key]
        first_row = first_use[key]
        steady_mean = as_float(steady_row.get("mean_seconds") or steady_row.get("seconds"))
        first_mean = as_float(first_row.get("mean_seconds") or first_row.get("seconds"))
        ratio = (
            first_mean / steady_mean
            if first_mean is not None and steady_mean not in (None, 0.0)
            else None
        )
        rows.append(
            {
                "dataset": dataset,
                "graph_type": first_row.get("graph_type", ""),
                "dataset_size": first_row.get("dataset_size", ""),
                "function": function,
                "metric": metric,
                "steady_mean_seconds": steady_mean,
                "steady_std_seconds": as_float(steady_row.get("std_seconds")),
                "steady_mean_plus_minus_sd": plus_minus(
                    steady_mean, steady_row.get("std_seconds")
                ),
                "steady_relative_std_percent": steady_row.get("relative_std_percent", ""),
                "steady_ci95_half_width_seconds": steady_row.get("ci95_half_width_value", ""),
                "first_use_mean_seconds": first_mean,
                "first_use_std_seconds": as_float(first_row.get("std_seconds")),
                "first_use_mean_plus_minus_sd": plus_minus(
                    first_mean, first_row.get("std_seconds")
                ),
                "first_use_relative_std_percent": first_row.get("relative_std_percent", ""),
                "first_use_ci95_half_width_seconds": first_row.get("ci95_half_width_value", ""),
                "first_use_over_steady": ratio,
                "sample_count": first_row.get("sample_count", ""),
            }
        )
    return rows


def add_host_overhead(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_pair = {}
    for row in rows:
        by_pair.setdefault((row["dataset"], row["function"]), {})[row["metric"]] = row
    output = []
    for (dataset, function), metrics in sorted(by_pair.items()):
        if not {"e2e", "kernel"} <= set(metrics):
            continue
        e2e = metrics["e2e"]
        kernel = metrics["kernel"]
        steady_host = max(
            0.0,
            float(e2e["steady_mean_seconds"]) - float(kernel["steady_mean_seconds"]),
        )
        first_host = max(
            0.0,
            float(e2e["first_use_mean_seconds"]) - float(kernel["first_use_mean_seconds"]),
        )
        output.append(
            {
                "dataset": dataset,
                "graph_type": e2e["graph_type"],
                "dataset_size": e2e["dataset_size"],
                "function": function,
                "steady_host_overhead_seconds": steady_host,
                "first_use_host_overhead_seconds": first_host,
                "first_use_extra_host_overhead_seconds": first_host - steady_host,
                "first_use_over_steady_e2e": e2e["first_use_over_steady"],
                "first_use_over_steady_kernel": kernel["first_use_over_steady"],
            }
        )
    return output


def compare_with_unwarmed_baselines(
    steady_dir: Path, first_use_dir: Path
) -> list[dict[str, object]]:
    """Reuse validated non-EGGPU rows from the main run as the cold-call control."""

    first = eligible(read_csv(first_use_dir / "results_long.csv"))
    output: list[dict[str, object]] = []
    for metric in sorted(COMPARISON_METRICS):
        validated_path = steady_dir / f"results_{metric}.csv"
        if not validated_path.exists():
            continue
        competitors: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in read_csv(validated_path):
            if row.get("baseline") == "EGGPU" or row.get("status") != "ok":
                continue
            value = as_float(row.get("mean_seconds") or row.get("seconds") or row.get("value"))
            if value is None:
                continue
            competitors.setdefault(
                (row.get("dataset", ""), row.get("function", "")), []
            ).append(row)
        for (dataset, function), candidates in sorted(competitors.items()):
            first_row = first.get((dataset, function, metric))
            if first_row is None:
                continue
            best = min(
                candidates,
                key=lambda row: float(
                    row.get("mean_seconds") or row.get("seconds") or row.get("value")
                ),
            )
            first_mean = as_float(
                first_row.get("mean_seconds") or first_row.get("seconds")
            )
            best_mean = as_float(
                best.get("mean_seconds") or best.get("seconds") or best.get("value")
            )
            output.append(
                {
                    "dataset": dataset,
                    "graph_type": first_row.get("graph_type", ""),
                    "dataset_size": first_row.get("dataset_size", ""),
                    "function": function,
                    "metric": metric,
                    "eggpu_first_use_mean_seconds": first_mean,
                    "eggpu_first_use_std_seconds": as_float(first_row.get("std_seconds")),
                    "eggpu_first_use_mean_plus_minus_sd": plus_minus(
                        first_mean, first_row.get("std_seconds")
                    ),
                    "best_unwarmed_baseline": best.get("baseline", ""),
                    "best_unwarmed_baseline_mean_seconds": best_mean,
                    "best_baseline_over_eggpu_first_use": (
                        best_mean / first_mean
                        if best_mean is not None and first_mean not in (None, 0.0)
                        else None
                    ),
                    "eggpu_first_use_is_fastest": (
                        best_mean is not None
                        and first_mean is not None
                        and first_mean <= best_mean
                    ),
                }
            )
    return output


def geomean(values) -> float | None:
    valid = [float(value) for value in values if value is not None and float(value) > 0]
    if not valid:
        return None
    return math.exp(sum(math.log(value) for value in valid) / len(valid))


def write_report(
    out_dir: Path,
    steady_dir: Path,
    rows: list[dict[str, object]],
    cold_baseline_rows: list[dict[str, object]],
    source_snapshot: str,
) -> None:
    lines = [
        "# EGGPU First-Use versus Steady-State\n\n",
        "## Protocol\n\n",
        "- First-use: fresh subprocess, Python graph only, no GraphContext/C++ prewarm, zero function warmup.\n",
        "- Steady-state: main experiment GraphContext/C++ prewarm plus two untimed full function calls.\n",
        "- Both E2E values exclude raw edge-list parsing and graph-object construction.\n",
        "- Every cell is the arithmetic mean of independent subprocess samples; `+/-` denotes sample standard deviation.\n",
        f"- Steady-state source: `{steady_dir}`.\n",
        f"- Executed source snapshot: `{source_snapshot}`.\n\n",
    ]
    for metric in ("e2e", "kernel"):
        subset = [row for row in rows if row["metric"] == metric]
        ratio = geomean(row["first_use_over_steady"] for row in subset)
        lines.extend(
            [
                f"## {metric.upper()}\n\n",
                f"- Comparable pairs: **{len(subset)}**.\n",
                f"- Geometric mean First-use / Steady-state: **{ratio:.3f}x**.\n\n"
                if ratio is not None
                else "- Geometric mean ratio: unavailable.\n\n",
            ]
        )
    lines.append("## First-use versus unwarmed baselines\n\n")
    for metric in ("e2e", "kernel"):
        subset = [row for row in cold_baseline_rows if row["metric"] == metric]
        wins = sum(bool(row["eggpu_first_use_is_fastest"]) for row in subset)
        lines.append(f"- {metric.upper()}: **{wins}/{len(subset)}** fastest pairs.\n")
    lines.append("\n")
    lines.extend(
        [
            "## Artifacts\n\n",
            "- `results_samples.csv`: untouched First-use samples.\n",
            "- `results_long.csv`: First-use aggregate statistics.\n",
            "- `first_use_vs_steady.csv`: aligned E2E/kernel comparison.\n",
            "- `first_use_host_overhead.csv`: E2E minus device-kernel decomposition.\n",
            "- `first_use_vs_unwarmed_baselines.csv`: cold-call fairness comparison using validated main-run competitors.\n",
        ]
    )
    (out_dir / "FIRST_USE_VS_STEADY.md").write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--steady-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--memory-repeat", type=int, default=1)
    parser.add_argument("--library-timeout", type=int, default=100)
    parser.add_argument("--inter-run-cooldown", type=float, default=1.0)
    parser.add_argument("--easygraph-repo", required=True)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--functions", default="all")
    parser.add_argument("--sssp-sources", type=int, default=8)
    parser.add_argument("--bc-sources", type=int, default=16)
    parser.add_argument("--pr-alpha", type=float, default=0.75)
    parser.add_argument("--pr-eps", type=float, default=1.0e-6)
    parser.add_argument("--pr-max-iter", type=int, default=200)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Reuse a completed split-measurement directory and run only the "
            "implementation checks and comparison postprocessing."
        ),
    )
    args = parser.parse_args()

    runtime_root, requested_runtime_provenance = prepare_frozen_runtime(
        args.easygraph_repo
    )
    args.easygraph_repo = str(runtime_root)
    steady_dir = Path(args.steady_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not (steady_dir / "results_long.csv").exists():
        raise SystemExit(f"missing steady-state results: {steady_dir / 'results_long.csv'}")
    required_existing = [
        out_dir / "results_samples.csv",
        out_dir / "results_long.csv",
        out_dir / "run_metadata.json",
    ]
    if args.reuse_existing:
        missing = [str(path) for path in required_existing if not path.exists()]
        if missing:
            raise SystemExit(
                "cannot reuse incomplete First-use directory; missing: "
                + ", ".join(missing)
            )
        print(f"Reusing completed First-use measurements: {out_dir}", flush=True)
    else:
        if out_dir.exists() and any(out_dir.iterdir()):
            raise SystemExit(
                f"refusing to overwrite non-empty First-use directory: {out_dir}"
            )
        out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(ROOT / "benchmarking" / "run_split_full_baselines.py"),
        "--gpu",
        str(args.gpu),
        "--out-dir",
        str(out_dir),
        "--repeat",
        str(args.repeat),
        "--memory-repeat",
        str(args.memory_repeat),
        "--warmup",
        "0",
        "--easygraph-warmup",
        "0",
        "--eggpu-execution-protocol",
        "first-use",
        "--baselines",
        "EGGPU",
        "--library-timeout",
        str(args.library_timeout),
        "--inter-run-cooldown",
        str(args.inter_run_cooldown),
        "--pr-alpha",
        str(args.pr_alpha),
        "--pr-eps",
        str(args.pr_eps),
        "--pr-max-iter",
        str(args.pr_max_iter),
        "--easygraph-repo",
        str(runtime_root),
        "--sssp-sources",
        str(args.sssp_sources),
        "--bc-sources",
        str(args.bc_sources),
        "--datasets",
        str(args.datasets),
        "--functions",
        str(args.functions),
    ]
    env = dict(os.environ)
    prepend_runtime_pythonpath(env, runtime_root)
    env["EGGPU_EXECUTION_PROTOCOL"] = "first-use"
    env["EGGPU_SKIP_PLOTS"] = "TRUE"
    if not args.reuse_existing:
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"First-use benchmark failed with exit code {completed.returncode}")

    snapshot, _ = verify_same_implementation(
        steady_dir,
        out_dir,
        expected_runtime_root=runtime_root,
        requested_runtime_provenance=requested_runtime_provenance,
    )
    rows = comparison_rows(steady_dir, out_dir)
    if not rows:
        raise SystemExit("no comparable EGGPU First-use/Steady-state rows")
    write_csv(out_dir / "first_use_vs_steady.csv", rows)
    write_csv(out_dir / "first_use_host_overhead.csv", add_host_overhead(rows))
    cold_baseline_rows = compare_with_unwarmed_baselines(steady_dir, out_dir)
    write_csv(
        out_dir / "first_use_vs_unwarmed_baselines.csv", cold_baseline_rows
    )
    write_report(out_dir, steady_dir, rows, cold_baseline_rows, snapshot)
    print(f"Done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
