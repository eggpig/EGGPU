#!/usr/bin/env python3
"""Run the large-graph sampled-target exact Closeness supplement.

The main benchmark keeps Closeness as exact all-node semantics.  This helper
only fills datasets that were symmetrically skipped by the exact scale guard,
using a separate sampled-target exact semantic with deterministic sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from gpu_device_profile import collect_gpu_device_profile, collect_host_profile
from gpu_visibility_marker import GpuVisibilityMarker
from measurement_schema import write_measurement_schema
from run_full_baselines import (
    check_eggpu_child_gpu_idle,
    collect_implementation_source_snapshot,
    collect_source_snapshot,
    cpp_easygraph_artifacts,
)


BASELINES = ["igraph", "networkx", "EGGPU", "easygraph-cpu", "easygraph-cpp", "nx-cugraph", "Gunrock"]
RUN_BASELINES = {"igraph", "networkx", "EGGPU", "easygraph-cpu", "easygraph-cpp"}
METRICS = ("build", "e2e", "kernel")
SEMANTIC = "sampled_target_exact"
ESTIMATOR_KIND = "exact_selected_vertices"
SOURCE_POLICY = "deterministic_evenly_spaced"
SOURCE_SEED = "none"
SOTA_TIE_TOL = 0.0005


def wait_for_uncontended_gpu(env: dict[str, str], poll_seconds: float = 10.0) -> None:
    """Wait outside every timing window while another process uses the GPU."""

    while True:
        idle, note = check_eggpu_child_gpu_idle(env)
        if idle:
            return
        print(f"[closeness-large] waiting for uncontended GPU: {note}", flush=True)
        time.sleep(max(1.0, float(poll_seconds)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def dataset_stats(main_result: Path) -> dict[str, dict[str, object]]:
    stats = json.loads((main_result / "dataset_stats.json").read_text())
    return {str(row["name"]): row for row in stats}


def pick_sources(n: int, k: int) -> list[int]:
    if n <= 0:
        return []
    k = max(1, int(k))
    if k >= n:
        return list(range(n))
    step = max(1, n // k)
    out = list(range(0, n, step))[:k]
    if len(out) < k:
        tail = n - 1
        while len(out) < k and tail >= 0:
            if tail not in out:
                out.append(tail)
            tail -= 1
    return out


def source_nodes_sha(sources: list[int]) -> str:
    payload = ",".join(str(int(x)) for x in sources).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def decode_subprocess_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def closeness_supplement_datasets(main_result: Path) -> list[dict[str, str]]:
    by_dataset: dict[str, set[str]] = {}
    for row in read_csv(main_result / "results_long.csv"):
        if row.get("function") != "Closeness" or row.get("baseline") != "EGGPU" or row.get("metric") != "e2e":
            continue
        status = row.get("status", "")
        notes = row.get("notes", "")
        skip_reason = row.get("skip_reason", "")
        dataset = row.get("dataset", "")
        if not dataset:
            continue
        if status == "skipped" and (
            skip_reason == "exact_scale_guard" or "exact all-source Closeness skipped" in notes
        ):
            by_dataset.setdefault(dataset, set()).add("exact_scale_guard")
        elif status in {"timeout", "failed"}:
            by_dataset.setdefault(dataset, set()).add(f"exact_{status}")
    return [
        {"dataset": dataset, "supplement_reason": "+".join(sorted(reasons))}
        for dataset, reasons in sorted(by_dataset.items())
    ]


def add_unavailable(
    rows: list[dict[str, object]],
    stat: dict[str, object],
    baseline: str,
    log_path: Path,
    sources: int,
    supplement_reason: str,
) -> None:
    note = "no aligned sampled-target exact Closeness backend in this benchmark"
    if baseline == "Gunrock":
        note = "no matching Gunrock executable for Closeness"
    elif baseline == "nx-cugraph":
        note = "nx-cugraph supported-algorithm list does not include closeness_centrality"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(note + "\n")
    source_list = pick_sources(int(stat.get("nodes_raw", 0) or 0), sources)
    source_sha = source_nodes_sha(source_list)
    for metric in METRICS:
        rows.append(
            {
                "dataset_size": stat["size"],
                "graph_type": stat["graph_type"],
                "dataset": stat["name"],
                "function": "Closeness",
                "baseline": baseline,
                "metric": metric,
                "seconds": "",
                "status": "skipped",
                "correctness": "",
                "log": str(log_path),
                "notes": note,
                "is_timeout": "False",
                "semantic": SEMANTIC,
                "skip_reason": "",
                "estimator_kind": ESTIMATOR_KIND,
                "sample_sources": sources,
                "source_policy": SOURCE_POLICY,
                "source_seed": SOURCE_SEED,
                "source_nodes_sha": source_sha,
                "exact_matrix_inclusion": "False",
                "is_supplement": "True",
                "supplement_reason": supplement_reason,
                "sample_index": "",
            }
        )


def parse_result_json(stdout: str) -> list[dict[str, object]]:
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("RESULT_JSON "):
            continue
        try:
            rows.append(json.loads(line[len("RESULT_JSON ") :]))
        except json.JSONDecodeError:
            continue
    return rows


def run_one(
    repo: Path,
    eval_dir: Path,
    stat: dict[str, object],
    baseline: str,
    sources: int,
    gpu: str,
    timeout: float,
    out_dir: Path,
    python: str,
    supplement_reason: str,
    sample_index: int,
) -> list[dict[str, object]]:
    ds_dir = out_dir / "logs" / str(stat["name"])
    details = ds_dir / "details" / f"sample_{sample_index}"
    log_path = ds_dir / f"library_{baseline}_closeness_sampled_{sample_index}.log"
    details.mkdir(parents=True, exist_ok=True)

    if baseline not in RUN_BASELINES:
        rows: list[dict[str, object]] = []
        add_unavailable(rows, stat, baseline, log_path, sources, supplement_reason)
        return rows

    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo) if not old_pp else str(repo) + os.pathsep + old_pp
    env["EGGPU_VALIDATION_DETAIL_DIR"] = str(details.resolve())
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["EGGPU_MONITOR_GPU_INDEX"] = str(gpu)
    env["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
    env["EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY"] = "FALSE"
    env["EGGPU_STRICT_VALIDATION"] = "TRUE"
    if baseline == "EGGPU":
        env["EASYGRAPH_ENABLE_GPU"] = "TRUE"
        env["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
        env["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
        env["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
        env["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
        easy_warmup = "2"
    else:
        env["EASYGRAPH_ENABLE_GPU"] = "FALSE"
        env["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
        easy_warmup = "0"

    cmd = [
        python,
        "benchmarking/library_baselines.py",
        str(stat["path"]),
        str(stat["graph_type"]),
        "--backend",
        baseline,
        "--function",
        "Closeness",
        "--warmup",
        "0",
        "--easygraph-warmup",
        easy_warmup,
        "--cooldown",
        "0",
        "--closeness-sources",
        str(sources),
    ]

    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=eval_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        output = decode_subprocess_output(proc.stdout)
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        output = decode_subprocess_output(exc.stdout)
        rc = 124
    elapsed = time.time() - t0
    source_list = pick_sources(int(stat.get("nodes_raw", 0) or 0), sources)
    source_sha = source_nodes_sha(source_list)
    log_path.write_text(
        "COMMAND: " + " ".join(cmd) + "\n"
        f"RETURN_CODE: {rc}\n"
        f"ELAPSED_SECONDS: {elapsed:.6f}\n"
        f"SEMANTIC: {SEMANTIC}\n"
        f"ESTIMATOR_KIND: {ESTIMATOR_KIND}\n"
        f"SOURCE_POLICY: {SOURCE_POLICY}\n"
        f"SOURCE_SEED: {SOURCE_SEED}\n"
        f"SOURCE_NODES_SHA: {source_sha}\n"
        f"SUPPLEMENT_REASON: {supplement_reason}\n"
        + output
    )

    emitted = parse_result_json(output)
    rows = []
    if not emitted:
        status = "timeout" if timed_out else "failed"
        note = f"library_baselines emitted no RESULT_JSON rows; return_code={rc}"
        for metric in METRICS:
            rows.append(
                {
                    "dataset_size": stat["size"],
                    "graph_type": stat["graph_type"],
                    "dataset": stat["name"],
                    "function": "Closeness",
                    "baseline": baseline,
                    "metric": metric,
                    "seconds": "",
                    "status": status,
                    "correctness": "",
                    "log": str(log_path),
                    "notes": note,
                    "is_timeout": str(timed_out),
                    "semantic": SEMANTIC,
                    "skip_reason": "",
                    "estimator_kind": ESTIMATOR_KIND,
                    "sample_sources": sources,
                    "source_policy": SOURCE_POLICY,
                    "source_seed": SOURCE_SEED,
                    "source_nodes_sha": source_sha,
                    "exact_matrix_inclusion": "False",
                    "is_supplement": "True",
                    "supplement_reason": supplement_reason,
                    "sample_index": sample_index,
                }
            )
        return rows

    for item in emitted:
        metric = str(item.get("metric", ""))
        rows.append(
            {
                "dataset_size": stat["size"],
                "graph_type": stat["graph_type"],
                "dataset": stat["name"],
                "function": str(item.get("function", "Closeness")),
                "baseline": baseline,
                "metric": metric,
                "seconds": "" if item.get("seconds") is None else item.get("seconds"),
                "status": str(item.get("status", "")),
                "correctness": str(item.get("correctness", "")),
                "log": str(log_path),
                "notes": str(item.get("notes", "")),
                "is_timeout": str(timed_out),
                "semantic": SEMANTIC,
                "skip_reason": "",
                "estimator_kind": ESTIMATOR_KIND,
                "sample_sources": sources if metric in METRICS else "",
                "source_policy": SOURCE_POLICY,
                "source_seed": SOURCE_SEED,
                "source_nodes_sha": source_sha,
                "exact_matrix_inclusion": "False",
                "is_supplement": "True",
                "supplement_reason": supplement_reason,
                "sample_index": sample_index,
            }
        )
    return rows


def detail_path(correctness: str) -> Path | None:
    match = re.search(r"detail=([^,]+)", correctness or "")
    if not match:
        return None
    return Path(match.group(1))


def validate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    import numpy as np

    by_key: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in rows:
        if row.get("metric") != "e2e" or row.get("status") != "ok":
            continue
        key = (
            str(row["dataset"]),
            str(row["function"]),
            str(row.get("sample_index", "")),
        )
        by_key.setdefault(key, {})[str(row["baseline"])] = row

    out = []
    ref_order = ["networkx", "easygraph-cpu", "easygraph-cpp", "igraph", "EGGPU"]
    for (dataset, function, sample_index), base_rows in sorted(by_key.items()):
        ref_name = next((b for b in ref_order if b in base_rows), None)
        if ref_name is None:
            continue
        ref_path = detail_path(str(base_rows[ref_name].get("correctness", "")))
        if ref_path is None or not ref_path.exists():
            continue
        ref = np.load(ref_path)
        ref_sources = ref["sources"].astype(np.int64)
        ref_values = ref["values"].astype(np.float64)
        for baseline, row in sorted(base_rows.items()):
            path = detail_path(str(row.get("correctness", "")))
            status = "missing_detail"
            max_abs = ""
            max_rel = ""
            if path is not None and path.exists():
                cur = np.load(path)
                sources = cur["sources"].astype(np.int64)
                values = cur["values"].astype(np.float64)
                if not np.array_equal(sources, ref_sources):
                    status = "source_mismatch"
                else:
                    diff = np.abs(values - ref_values)
                    denom = np.maximum(1.0, np.abs(ref_values))
                    rel = diff / denom
                    max_abs_f = float(diff.max()) if diff.size else 0.0
                    max_rel_f = float(rel.max()) if rel.size else 0.0
                    max_abs = f"{max_abs_f:.12g}"
                    max_rel = f"{max_rel_f:.12g}"
                    status = "pass" if np.allclose(values, ref_values, rtol=1e-6, atol=1e-9) else "fail"
            out.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "baseline": baseline,
                    "sample_index": sample_index,
                    "reference": ref_name,
                    "validation_status": status,
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                    "semantic": SEMANTIC,
                    "estimator_kind": str(row.get("estimator_kind", ESTIMATOR_KIND)),
                    "sample_sources": str(row.get("sample_sources", "")),
                    "source_policy": str(row.get("source_policy", SOURCE_POLICY)),
                    "source_seed": str(row.get("source_seed", SOURCE_SEED)),
                    "source_nodes_sha": str(row.get("source_nodes_sha", "")),
                }
            )
    return out


def aggregate_samples(
    rows: list[dict[str, object]], expected_repeat: int
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("baseline", "")),
            str(row.get("metric", "")),
        )
        groups.setdefault(key, []).append(row)

    aggregates: list[dict[str, object]] = []
    for _, group in sorted(groups.items()):
        representative = dict(group[0])
        values = []
        for row in group:
            if row.get("status") != "ok":
                continue
            try:
                values.append(float(row.get("seconds", "")))
            except (TypeError, ValueError):
                continue
        if values:
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            representative.update(
                {
                    "seconds": mean,
                    "mean_seconds": mean,
                    "std_seconds": std,
                    "sample_count": len(values),
                    "expected_sample_count": expected_repeat,
                    "relative_std_percent": 100.0 * std / mean if mean else 0.0,
                    "status": "ok" if len(values) == expected_repeat else "incomplete",
                    "notes": (
                        str(representative.get("notes", ""))
                        + f"; aggregate over {len(values)}/{expected_repeat} successful samples"
                    ).strip("; "),
                    "sample_index": "",
                }
            )
        else:
            statuses = {str(row.get("status", "")) for row in group}
            representative.update(
                {
                    "seconds": "",
                    "mean_seconds": "",
                    "std_seconds": "",
                    "sample_count": 0,
                    "expected_sample_count": expected_repeat,
                    "relative_std_percent": "",
                    "status": "timeout" if "timeout" in statuses else representative.get("status", "failed"),
                    "sample_index": "",
                }
            )
        aggregates.append(representative)
    return aggregates


def apply_validation_contract(
    rows: list[dict[str, object]],
    validation: list[dict[str, object]],
    expected_repeat: int,
) -> list[dict[str, object]]:
    pass_counts: dict[tuple[str, str], int] = {}
    for row in validation:
        if row.get("validation_status") != "pass":
            continue
        key = (str(row.get("dataset", "")), str(row.get("baseline", "")))
        pass_counts[key] = pass_counts.get(key, 0) + 1

    output = []
    for source in rows:
        row = dict(source)
        baseline = str(row.get("baseline", ""))
        if baseline in RUN_BASELINES:
            key = (str(row.get("dataset", "")), baseline)
            passed = pass_counts.get(key, 0)
            row["validation_pass_samples"] = passed
            row["validation_expected_samples"] = expected_repeat
            row["validation_status"] = (
                "pass" if passed == expected_repeat else "incomplete_or_failed"
            )
            if row.get("status") == "ok" and passed != expected_repeat:
                row["status"] = "validation_failed"
                row["notes"] = (
                    str(row.get("notes", ""))
                    + f"; validation_pass_samples={passed}/{expected_repeat}"
                ).strip("; ")
        output.append(row)
    return output


def summarize_sota(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for metric in ("e2e", "kernel"):
        by_dataset: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            if row.get("metric") != metric or row.get("status") != "ok":
                continue
            try:
                float(row.get("seconds", ""))
            except Exception:
                continue
            by_dataset.setdefault(str(row["dataset"]), []).append(row)
        for dataset, items in sorted(by_dataset.items()):
            best = min(items, key=lambda r: float(r["seconds"]))
            eggpu = next((r for r in items if r["baseline"] == "EGGPU"), None)
            if eggpu is None:
                continue
            eggpu_s = float(eggpu["seconds"])
            best_s = float(best["seconds"])
            ratio = eggpu_s / best_s if best_s > 0 else 1.0
            out.append(
                {
                    "dataset": dataset,
                    "function": "Closeness",
                    "metric": metric,
                    "eggpu_seconds": f"{eggpu_s:.12g}",
                    "best_baseline": best["baseline"],
                    "best_seconds": f"{best_s:.12g}",
                    "ratio_to_best": f"{ratio:.12g}",
                    "is_pair_sota": str(ratio <= 1.0 + SOTA_TIE_TOL),
                    "semantic": SEMANTIC,
                }
            )
    return out


def write_markdown(
    out_dir: Path,
    rows: list[dict[str, object]],
    validation: list[dict[str, object]],
    sota: list[dict[str, object]],
    sources: int,
    repeat: int,
) -> None:
    ok_rows = [r for r in rows if r.get("metric") in {"e2e", "kernel"} and r.get("status") == "ok"]
    pass_rows = [r for r in validation if r.get("validation_status") == "pass"]
    lines = [
        "# Large-Graph Closeness Supplement",
        "",
        f"- Semantic: `{SEMANTIC}`.",
        f"- Estimator kind: `{ESTIMATOR_KIND}`.",
        f"- Source policy: `{SOURCE_POLICY}`, `sources={sources}`, `source_seed={SOURCE_SEED}`.",
        f"- Successful timing tasks use arithmetic mean +/- sample standard deviation over {repeat} independent runs.",
        "- A baseline that reaches the timeout on its first sample is right-censored and is not repeated.",
        "- This supplement does not replace exact all-node Closeness rows; it fills scale-guarded or exact-timeout datasets with an explicitly labeled sampled-target exact task.",
        "- The sampled task computes exact Closeness only for the deterministic target/source vertices. It must not be described as all-node exact Closeness or as an ADS/HIP/HyperBall approximation.",
        f"- Timed ok rows: {len(ok_rows)}.",
        f"- Validation pass rows: {len(pass_rows)}/{len(validation)}.",
        "",
        "## SOTA",
        "",
        "| dataset | metric | EGGPU s | best | best s | ratio | sota |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in sota:
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['eggpu_seconds']} | "
            f"{row['best_baseline']} | {row['best_seconds']} | {row['ratio_to_best']} | {row['is_pair_sota']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `closeness_large_sampled_long.csv`",
            "- `closeness_large_sampled_summary.csv`",
            "- `closeness_large_sampled_e2e.csv`",
            "- `closeness_large_sampled_kernel.csv`",
            "- `closeness_large_sampled_validation.csv`",
            "- `closeness_large_sampled_sota.csv`",
            "- `closeness_large_sampled_metadata.json`",
            "- `results_long_with_closeness_large_sampled.csv`",
        ]
    )
    (out_dir / "CLOSENESS_LARGE_SUPPLEMENT.md").write_text("\n".join(lines) + "\n")


def merged_view(main_result: Path, supplement_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    main_rows = read_csv(main_result / "results_long.csv")
    out = []
    for row in main_rows:
        row = dict(row)
        row.setdefault("semantic", "exact_all_node")
        row.setdefault("skip_reason", "")
        row.setdefault("sample_sources", "")
        row.setdefault("source_policy", "")
        row.setdefault("source_seed", "")
        row.setdefault("source_nodes_sha", "")
        row.setdefault("estimator_kind", "")
        row.setdefault("exact_matrix_inclusion", "True")
        row.setdefault("is_supplement", "False")
        row.setdefault("supplement_reason", "")
        out.append(row)
    out.extend(supplement_rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("main_result_dir", type=Path)
    ap.add_argument("--easygraph-repo", type=Path, default=Path("../Easy-Graph"))
    ap.add_argument("--sources", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.repeat < 2:
        raise SystemExit("--repeat must be at least 2")

    eval_dir = Path(__file__).resolve().parents[1]
    main_result = args.main_result_dir
    if not main_result.is_absolute():
        main_result = (eval_dir / main_result).resolve()
    repo = args.easygraph_repo
    if not repo.is_absolute():
        repo = (eval_dir / repo).resolve()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = main_result / "closeness_large_sampled"
    elif not out_dir.is_absolute():
        out_dir = (eval_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_measurement_schema(out_dir / "measurement_schema.json")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["EGGPU_MONITOR_GPU_INDEX"] = str(args.gpu)
    gpu_device_profile = collect_gpu_device_profile(args.gpu, os.environ)
    host_profile = collect_host_profile()
    marker = GpuVisibilityMarker(args.gpu, "large Closeness supplement").start()
    if marker.started:
        os.environ["EGGPU_EXTERNAL_VISIBILITY_MARKER"] = "TRUE"

    stats_by_name = dataset_stats(main_result)
    if args.datasets is not None:
        datasets = [{"dataset": name, "supplement_reason": "manual"} for name in args.datasets]
    else:
        datasets = closeness_supplement_datasets(main_result)
    if not datasets:
        print("No scale-guarded or exact-timeout Closeness datasets found.", flush=True)
        return 0

    all_rows: list[dict[str, object]] = []
    total_tasks = len(datasets) * (len(RUN_BASELINES) * args.repeat + len(BASELINES) - len(RUN_BASELINES))
    completed_tasks = 0
    for item in datasets:
        dataset = item["dataset"]
        supplement_reason = item["supplement_reason"]
        if dataset not in stats_by_name:
            raise SystemExit(f"Unknown dataset in dataset_stats.json: {dataset}")
        stat = stats_by_name[dataset]
        for baseline in BASELINES:
            samples = range(1, args.repeat + 1) if baseline in RUN_BASELINES else range(1, 2)
            for sample_index in samples:
                completed_tasks += 1
                print(
                    f"[closeness-large] {completed_tasks}/{total_tasks} "
                    f"dataset={dataset} baseline={baseline} "
                    f"sample={sample_index}/{args.repeat} sources={args.sources} "
                    f"reason={supplement_reason}",
                    flush=True,
                )
                wait_for_uncontended_gpu(os.environ)
                rows = run_one(
                    repo=repo,
                    eval_dir=eval_dir,
                    stat=stat,
                    baseline=baseline,
                    sources=args.sources,
                    gpu=str(args.gpu),
                    timeout=float(args.timeout),
                    out_dir=out_dir,
                    python=args.python,
                    supplement_reason=supplement_reason,
                    sample_index=sample_index,
                )
                all_rows.extend(rows)
                if any(row.get("status") in {"timeout", "failed"} for row in rows):
                    print(
                        f"[closeness-large] right-censoring dataset={dataset} "
                        f"baseline={baseline} after sample={sample_index}",
                        flush=True,
                    )
                    break

    fields = [
        "dataset_size",
        "graph_type",
        "dataset",
        "function",
        "baseline",
        "metric",
        "seconds",
        "status",
        "correctness",
        "log",
        "notes",
        "is_timeout",
        "semantic",
        "skip_reason",
        "estimator_kind",
        "sample_sources",
        "source_policy",
        "source_seed",
        "source_nodes_sha",
        "exact_matrix_inclusion",
        "is_supplement",
        "supplement_reason",
        "sample_index",
        "mean_seconds",
        "std_seconds",
        "sample_count",
        "expected_sample_count",
        "relative_std_percent",
    ]
    write_csv(out_dir / "closeness_large_sampled_long.csv", all_rows, fields)
    write_csv(out_dir / "closeness_large_sampled_e2e.csv", [r for r in all_rows if r["metric"] == "e2e"], fields)
    write_csv(out_dir / "closeness_large_sampled_kernel.csv", [r for r in all_rows if r["metric"] == "kernel"], fields)
    write_csv(out_dir / "closeness_large_sampled_build.csv", [r for r in all_rows if r["metric"] == "build"], fields)

    validation = validate(all_rows)
    write_csv(
        out_dir / "closeness_large_sampled_validation.csv",
        validation,
        [
            "dataset",
            "function",
            "baseline",
            "sample_index",
            "reference",
            "validation_status",
            "max_abs",
            "max_rel",
            "semantic",
            "estimator_kind",
            "sample_sources",
            "source_policy",
            "source_seed",
            "source_nodes_sha",
        ],
    )
    aggregates = apply_validation_contract(
        aggregate_samples(all_rows, args.repeat), validation, args.repeat
    )
    aggregate_fields = fields + [
        "validation_pass_samples",
        "validation_expected_samples",
        "validation_status",
    ]
    write_csv(
        out_dir / "closeness_large_sampled_summary.csv",
        aggregates,
        aggregate_fields,
    )
    sota = summarize_sota(aggregates)
    write_csv(
        out_dir / "closeness_large_sampled_sota.csv",
        sota,
        ["dataset", "function", "metric", "eggpu_seconds", "best_baseline", "best_seconds", "ratio_to_best", "is_pair_sota", "semantic"],
    )
    merged = merged_view(main_result, aggregates)
    write_csv(
        out_dir / "results_long_with_closeness_large_sampled.csv",
        merged,
        aggregate_fields,
    )
    metadata = {
        "schema_version": 1,
        "protocol": "sampled_target_exact_closeness_v1",
        "main_result": str(main_result),
        "datasets": [item["dataset"] for item in datasets],
        "baselines": BASELINES,
        "run_baselines": sorted(RUN_BASELINES),
        "sources": args.sources,
        "source_policy": SOURCE_POLICY,
        "source_seed": SOURCE_SEED,
        "repeat": args.repeat,
        "timeout_seconds": args.timeout,
        "timing_estimator": "arithmetic_mean",
        "error_bar": "sample_standard_deviation",
        "gpu_device_profile": gpu_device_profile,
        "host_profile": host_profile,
        "source_snapshot": collect_source_snapshot(),
        "implementation_source_snapshot": collect_implementation_source_snapshot(),
        "build_artifacts": {"cpp_easygraph": cpp_easygraph_artifacts()},
        "visibility_marker": {
            "requested": True,
            "started": bool(marker.started),
            "memory_metrics_collected": False,
        },
    }
    (out_dir / "closeness_large_sampled_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    write_markdown(out_dir, aggregates, validation, sota, args.sources, args.repeat)
    marker.stop()
    print(f"Wrote large Closeness supplement: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
