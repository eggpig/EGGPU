#!/usr/bin/env python3
"""Measure prepared-state igraph PageRank through the native edge-list reader.

Each timing sample is a fresh process. The worker streams an existing int32 CSR
through a FIFO into ``igraph.Graph.Read_Edgelist`` so the benchmark never
materializes one Python tuple per edge. Graph construction is excluded from the
reported PageRank latency. Three preceding calls and one measured fourth call
match the EGGPU Introduction scaling protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


WORKER = (
    Path(__file__).resolve().parent
    / "diagnostics"
    / "igraph_read_edgelist_fifo.py"
)
WRITER_SOURCE = WORKER.with_name("stream_csr_edgelist_fifo.cpp")

PROTOCOL_VERSION = "figure1-igraph-fifo-call4-v2"
FIGURE1_REPEAT = 5
FIGURE1_WARMUP_CALLS = 3
FIGURE1_MEMORY_LIMIT_GB = 128.0
FIGURE1_DAMPING = 0.75
FIGURE1_MEASURED_CALL_POSITION = 4
TIMER_BOUNDARY = (
    "igraph PageRank public invocation through complete result return; "
    "benchmark validation excluded"
)
THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "PYTHONHASHSEED",
    "MALLOC_ARENA_MAX",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=FIGURE1_REPEAT)
    parser.add_argument("--warmup-calls", type=int, default=FIGURE1_WARMUP_CALLS)
    parser.add_argument(
        "--memory-limit-gb", type=float, default=FIGURE1_MEMORY_LIMIT_GB
    )
    parser.add_argument("--process-timeout", type=float, default=2400.0)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.repeat != FIGURE1_REPEAT:
        parser.error(f"Figure 1 requires --repeat {FIGURE1_REPEAT}")
    if args.warmup_calls != FIGURE1_WARMUP_CALLS:
        parser.error(
            f"Figure 1 requires --warmup-calls {FIGURE1_WARMUP_CALLS}"
        )
    if not math.isclose(
        args.memory_limit_gb,
        FIGURE1_MEMORY_LIMIT_GB,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        parser.error(
            f"Figure 1 requires --memory-limit-gb "
            f"{FIGURE1_MEMORY_LIMIT_GB:g}"
        )
    if not math.isfinite(args.process_timeout) or args.process_timeout <= 0.0:
        parser.error("--process-timeout must be finite and positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def atomic_write_json(path: Path, record) -> None:
    atomic_write_text(
        path,
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def child_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("MALLOC_ARENA_MAX", "2")
    environment.setdefault(
        "MPLCONFIGDIR",
        str((args.out_dir / "matplotlib_cache").resolve()),
    )
    return environment


def execution_context(environment: dict[str, str]) -> dict:
    try:
        allowed_cpu_ids = sorted(int(value) for value in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed_cpu_ids = []
    return {
        "allowed_cpu_ids": allowed_cpu_ids,
        "allowed_cpu_count": len(allowed_cpu_ids),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {
            key: environment.get(key) for key in THREAD_ENVIRONMENT_KEYS
        },
    }


def make_run_config(
    args: argparse.Namespace,
    manifest: Path,
    environment: dict[str, str],
) -> dict:
    manifest = manifest.resolve()
    context = execution_context(environment)
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "measurement_args": {
            "repeat": int(args.repeat),
            "warmup_calls": int(args.warmup_calls),
            "memory_limit_gb": float(args.memory_limit_gb),
            "process_timeout_seconds": float(args.process_timeout),
            "damping": FIGURE1_DAMPING,
            "measured_call_position": FIGURE1_MEASURED_CALL_POSITION,
        },
        "worker_path": str(WORKER.resolve()),
        "worker_sha256": sha256_file(WORKER),
        "writer_source_path": str(WRITER_SOURCE.resolve()),
        "writer_source_sha256": sha256_file(WRITER_SOURCE),
        "python_executable": str(Path(sys.executable).resolve()),
        "igraph_version": importlib.metadata.version("igraph"),
        "cpu_affinity": {
            "allowed_cpu_ids": context["allowed_cpu_ids"],
            "allowed_cpu_count": context["allowed_cpu_count"],
            "logical_cpu_count": context["logical_cpu_count"],
        },
        "thread_environment": context["thread_environment"],
        "timer_boundary": TIMER_BOUNDARY,
    }
    canonical = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    config["protocol_signature"] = hashlib.sha256(canonical).hexdigest()
    return config


def attach_run_config(record: dict, run_config: dict) -> dict:
    record = dict(record)
    record["protocol_signature"] = run_config["protocol_signature"]
    record["run_config"] = run_config
    record["memory_limit_gb"] = run_config["measurement_args"][
        "memory_limit_gb"
    ]
    record["cpu_affinity"] = run_config["cpu_affinity"]
    record["thread_environment"] = run_config["thread_environment"]
    return record


def load_resumable_json(
    path: Path,
    run_config: dict,
    *,
    artifact_kind: str,
) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(
            f"refusing to reuse malformed {artifact_kind} {path}: {error}"
        ) from error
    observed = record.get("protocol_signature")
    expected = run_config["protocol_signature"]
    if observed != expected:
        raise RuntimeError(
            f"refusing to reuse {artifact_kind} {path}: protocol signature "
            f"{observed!r} != {expected!r}; use --no-resume to overwrite this "
            "explicit target"
        )
    return record


def infer_failure_stage(stderr: str, *, timed_out: bool = False) -> str:
    if timed_out:
        return "worker_process_timeout"
    lowered = stderr.lower()
    if (
        "validate_pagerank" in lowered
        or "pagerank result invariants failed" in lowered
    ):
        return "result_validation"
    if "graph.pagerank" in lowered:
        return "pagerank_call"
    if any(
        marker in lowered
        for marker in (
            "graph.read_edgelist",
            "csr stream writer",
            "cardinality mismatch",
            "stream_csr_edgelist_fifo:",
            "writer.communicate",
        )
    ):
        return "graph_prepare"
    return "worker_process"


def finite_positive(value, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {numeric!r}")
    return numeric


def validate_result_payload(payload, expected_size: int, label: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} validation payload is not an object")
    required = {"result_size", "minimum", "maximum", "sum"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{label} validation is missing {missing}")
    if int(payload["result_size"]) != expected_size:
        raise ValueError(
            f"{label} result_size={payload['result_size']} != {expected_size}"
        )
    minimum = float(payload["minimum"])
    maximum = float(payload["maximum"])
    score_sum = float(payload["sum"])
    if not all(math.isfinite(value) for value in (minimum, maximum, score_sum)):
        raise ValueError(f"{label} validation contains a non-finite value")
    if minimum < 0.0 or maximum < minimum:
        raise ValueError(
            f"{label} validation has invalid range [{minimum}, {maximum}]"
        )
    if abs(score_sum - 1.0) > 1.0e-8:
        raise ValueError(f"{label} PageRank sum {score_sum} is not one")


def validate_sample(
    sample: dict,
    *,
    dataset: str,
    metadata: dict,
    run_config: dict,
    expected_index: int,
) -> str:
    if not isinstance(sample, dict):
        raise ValueError(f"sample {expected_index} is not an object")
    if sample.get("status") != "ok":
        raise ValueError(
            f"sample {expected_index} status is {sample.get('status')!r}"
        )
    if sample.get("protocol_signature") != run_config["protocol_signature"]:
        raise ValueError(f"sample {expected_index} protocol signature mismatch")
    if sample.get("run_config") != run_config:
        raise ValueError(f"sample {expected_index} run_config mismatch")
    if int(sample.get("timing_process_index", -1)) != expected_index:
        raise ValueError(
            f"sample {expected_index} has timing_process_index="
            f"{sample.get('timing_process_index')!r}"
        )
    if sample.get("dataset") != dataset:
        raise ValueError(
            f"sample {expected_index} dataset={sample.get('dataset')!r}, "
            f"expected {dataset!r}"
        )
    expected_nodes = int(metadata["num_nodes"])
    expected_entries = int(metadata["num_entries"])
    if int(sample.get("num_nodes", -1)) != expected_nodes:
        raise ValueError(f"sample {expected_index} node count mismatch")
    if int(sample.get("num_entries", -1)) != expected_entries:
        raise ValueError(f"sample {expected_index} entry count mismatch")
    version = sample.get("igraph_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"sample {expected_index} has no igraph_version")
    if version != run_config["igraph_version"]:
        raise ValueError(
            f"sample {expected_index} igraph_version={version!r}, expected "
            f"{run_config['igraph_version']!r}"
        )
    parsed_vertices = int(sample.get("parsed_vertices_before_isolate_fill", -1))
    if parsed_vertices < 0 or parsed_vertices > expected_nodes:
        raise ValueError(
            f"sample {expected_index} has invalid parsed vertex count "
            f"{parsed_vertices}"
        )
    finite_positive(
        sample.get("read_edgelist_seconds"),
        f"sample {expected_index} graph preparation time",
    )
    finite_positive(
        sample.get("host_rss_peak_mib"),
        f"sample {expected_index} host RSS peak",
    )

    pagerank = sample.get("pagerank")
    if not isinstance(pagerank, dict):
        raise ValueError(f"sample {expected_index} has no pagerank record")
    damping = float(pagerank.get("damping", float("nan")))
    if not math.isclose(
        damping, FIGURE1_DAMPING, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(
            f"sample {expected_index} damping={damping}, "
            f"expected {FIGURE1_DAMPING}"
        )
    if int(pagerank.get("warmup_calls", -1)) != FIGURE1_WARMUP_CALLS:
        raise ValueError(f"sample {expected_index} warmup count mismatch")
    if (
        int(pagerank.get("measured_call_position", -1))
        != FIGURE1_MEASURED_CALL_POSITION
    ):
        raise ValueError(f"sample {expected_index} measured position mismatch")
    if pagerank.get("timer_boundary") != TIMER_BOUNDARY:
        raise ValueError(f"sample {expected_index} timer boundary mismatch")
    finite_positive(
        pagerank.get("seconds"),
        f"sample {expected_index} measured PageRank time",
    )
    validate_result_payload(
        pagerank.get("validation"),
        expected_nodes,
        f"sample {expected_index} measured call",
    )

    warmups = pagerank.get("warmup_details")
    if not isinstance(warmups, list) or len(warmups) != FIGURE1_WARMUP_CALLS:
        raise ValueError(
            f"sample {expected_index} must contain "
            f"{FIGURE1_WARMUP_CALLS} warmup records"
        )
    for warmup_index, warmup in enumerate(warmups, start=1):
        if not isinstance(warmup, dict):
            raise ValueError(
                f"sample {expected_index} warmup {warmup_index} is not an object"
            )
        if int(warmup.get("call_position", -1)) != warmup_index:
            raise ValueError(
                f"sample {expected_index} warmup position mismatch"
            )
        finite_positive(
            warmup.get("seconds"),
            f"sample {expected_index} warmup {warmup_index} time",
        )
        validate_result_payload(
            warmup.get("validation"),
            expected_nodes,
            f"sample {expected_index} warmup {warmup_index}",
        )
    return version


def validate_success_batch(
    samples: list[dict],
    *,
    dataset: str,
    metadata: dict,
    run_config: dict,
) -> str:
    if len(samples) != FIGURE1_REPEAT:
        raise ValueError(
            f"{dataset} contains {len(samples)} samples; "
            f"Figure 1 requires {FIGURE1_REPEAT}"
        )
    versions = {
        validate_sample(
            sample,
            dataset=dataset,
            metadata=metadata,
            run_config=run_config,
            expected_index=index,
        )
        for index, sample in enumerate(samples, start=1)
    }
    if len(versions) != 1:
        raise ValueError(
            f"{dataset} timing samples use different igraph versions: "
            f"{sorted(versions)}"
        )
    return next(iter(versions))


def flatten(record: dict) -> dict:
    return {
        key: (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
        )
        for key, value in record.items()
    }


def run_sample(
    args: argparse.Namespace,
    manifest: Path,
    output_path: Path,
    *,
    sample_index: int,
    environment: dict[str, str],
    run_config: dict,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.worker.",
        suffix=".json",
        delete=False,
    ) as handle:
        worker_output = Path(handle.name)
    worker_output.unlink()
    command = [
        sys.executable,
        str(WORKER),
        str(manifest.resolve()),
        "--warmup-calls",
        str(args.warmup_calls),
        "--memory-limit-gb",
        str(args.memory_limit_gb),
        "--output",
        str(worker_output.resolve()),
    ]
    record = None
    try:
        try:
            completed = subprocess.run(
                command,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=args.process_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stderr = error.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            record = {
                "status": "failed",
                "failure_kind": "timeout",
                "failure_stage": infer_failure_stage(
                    stderr, timed_out=True
                ),
                "error": str(error),
                "stderr_tail": stderr[-1200:],
            }
        else:
            if completed.returncode != 0 or not worker_output.is_file():
                lowered = completed.stderr.lower()
                resource_markers = (
                    "memoryerror",
                    "cannot allocate",
                    "out of memory",
                    "std::bad_alloc",
                )
                failure_kind = (
                    "resource_limit"
                    if any(marker in lowered for marker in resource_markers)
                    else "execution_error"
                )
                record = {
                    "status": "failed",
                    "failure_kind": failure_kind,
                    "failure_stage": infer_failure_stage(completed.stderr),
                    "returncode": completed.returncode,
                    "error": completed.stderr[-4000:],
                    "stderr_tail": completed.stderr[-1200:],
                }
            else:
                try:
                    record = json.loads(
                        worker_output.read_text(encoding="utf-8")
                    )
                except Exception as error:
                    record = {
                        "status": "failed",
                        "failure_kind": "invalid_worker_output",
                        "failure_stage": "worker_output_parse",
                        "returncode": completed.returncode,
                        "error": f"{type(error).__name__}: {error}",
                        "stderr_tail": completed.stderr[-1200:],
                    }
                else:
                    record["stderr_tail"] = completed.stderr[-1200:]
                    if record.get("status") != "ok":
                        record.setdefault("failure_kind", "execution_error")
                        record.setdefault(
                            "failure_stage",
                            infer_failure_stage(completed.stderr),
                        )
    finally:
        worker_output.unlink(missing_ok=True)

    if record is None:
        record = {
            "status": "failed",
            "failure_kind": "coordinator_error",
            "failure_stage": "coordinator",
            "error": "worker execution produced no record",
        }
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    record.setdefault("dataset", str(metadata["name"]))
    record.setdefault("baseline", "igraph")
    record.setdefault("function", "PageRank")
    record.setdefault("num_nodes", int(metadata["num_nodes"]))
    record.setdefault("num_entries", int(metadata["num_entries"]))
    record["timing_process_index"] = int(sample_index)
    record["expected_timing_processes"] = FIGURE1_REPEAT
    record = attach_run_config(record, run_config)
    atomic_write_json(output_path, record)
    return record


def aggregate(
    dataset: str,
    metadata: dict,
    samples: list[dict],
    run_config: dict,
) -> dict:
    base = attach_run_config(
        {
            "dataset": dataset,
            "baseline": "igraph",
            "function": "PageRank",
            "num_nodes": int(metadata["num_nodes"]),
            "num_entries": int(metadata["num_entries"]),
            "expected_timing_processes": FIGURE1_REPEAT,
            "attempted_timing_processes": len(samples),
            "unattempted_timing_processes": max(
                0, FIGURE1_REPEAT - len(samples)
            ),
        },
        run_config,
    )
    signature_failures = [
        index
        for index, sample in enumerate(samples, start=1)
        if sample.get("protocol_signature")
        != run_config["protocol_signature"]
    ]
    if signature_failures:
        return {
            **base,
            "status": "failed",
            "validation": "fail",
            "failure_kind": "protocol_signature_mismatch",
            "failure_stage": "aggregate_validation",
            "successful_timing_processes": sum(
                sample.get("status") == "ok" for sample in samples
            ),
            "failed_timing_processes": sum(
                sample.get("status") != "ok" for sample in samples
            ),
            "error": f"sample signatures differ at {signature_failures}",
            "process_records": samples,
        }

    failures = [sample for sample in samples if sample.get("status") != "ok"]
    if failures or len(samples) != FIGURE1_REPEAT:
        failure_kinds = {
            failure.get("failure_kind", "execution_error")
            for failure in failures
        }
        failure_stages = {
            failure.get("failure_stage", "worker_process")
            for failure in failures
        }
        if len(samples) != FIGURE1_REPEAT:
            failure_kinds.add("incomplete_sample_batch")
            failure_stages.add("sample_collection")
        return {
            **base,
            "status": "failed",
            "validation": "fail",
            "successful_timing_processes": len(samples) - len(failures),
            "failed_timing_processes": len(failures),
            "failure_kind": "+".join(sorted(failure_kinds)),
            "failure_stage": "+".join(sorted(failure_stages)),
            "process_records": samples,
        }

    try:
        igraph_version = validate_success_batch(
            samples,
            dataset=dataset,
            metadata=metadata,
            run_config=run_config,
        )
    except Exception as error:
        return {
            **base,
            "status": "failed",
            "validation": "fail",
            "failure_kind": "protocol_validation_error",
            "failure_stage": "aggregate_validation",
            "successful_timing_processes": len(samples),
            "failed_timing_processes": 0,
            "error": f"{type(error).__name__}: {error}",
            "process_records": samples,
        }

    values = [float(sample["pagerank"]["seconds"]) for sample in samples]
    builds = [float(sample["read_edgelist_seconds"]) for sample in samples]
    return {
        **base,
        "status": "ok",
        "validation": "pass",
        "igraph_version": igraph_version,
        "e2e_best_seconds": min(values),
        "e2e_mean_seconds": statistics.mean(values),
        "e2e_stdev_seconds": statistics.stdev(values),
        "timing_process_samples": len(values),
        "successful_timing_processes": len(values),
        "failed_timing_processes": 0,
        "graph_prepare_mean_seconds": statistics.mean(builds),
        "graph_prepare_stdev_seconds": statistics.stdev(builds),
        "damping": FIGURE1_DAMPING,
        "warmup_calls": FIGURE1_WARMUP_CALLS,
        "measured_call_position": FIGURE1_MEASURED_CALL_POSITION,
        "timer_boundary": TIMER_BOUNDARY,
        "graph_construction_path": "csr_fifo_to_igraph_read_edgelist",
        "host_rss_peak_mib_mean": statistics.mean(
            float(sample["host_rss_peak_mib"]) for sample in samples
        ),
        "process_records": samples,
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = args.out_dir / "raw"
    raw.mkdir(exist_ok=True)
    environment = child_environment(args)
    json_path = args.out_dir / "igraph_intro_scaling_fifo.json"
    csv_path = args.out_dir / "igraph_intro_scaling_fifo.csv"
    jobs = []
    final_paths = set()
    for manifest in args.manifests:
        manifest = manifest.resolve()
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        dataset = str(metadata["name"])
        run_config = make_run_config(args, manifest, environment)
        final_path = args.out_dir / f"{dataset}_igraph_PageRank.json"
        if final_path in final_paths:
            raise ValueError(
                f"duplicate Figure 1 dataset target for {dataset}: {final_path}"
            )
        final_paths.add(final_path)
        jobs.append((manifest, metadata, dataset, run_config, final_path))

    if not args.resume:
        pending_aggregates = []
        for _, _, dataset, run_config, final_path in jobs:
            pending_base = attach_run_config(
                {
                    "status": "pending",
                    "dataset": dataset,
                    "baseline": "igraph",
                    "function": "PageRank",
                    "expected_timing_processes": FIGURE1_REPEAT,
                    "attempted_timing_processes": 0,
                    "failure_stage": "sample_collection",
                },
                run_config,
            )
            atomic_write_json(final_path, pending_base)
            pending_aggregates.append(pending_base)
            for index in range(1, FIGURE1_REPEAT + 1):
                pending_sample = attach_run_config(
                    {
                        "status": "pending",
                        "dataset": dataset,
                        "baseline": "igraph",
                        "function": "PageRank",
                        "timing_process_index": index,
                        "expected_timing_processes": FIGURE1_REPEAT,
                        "failure_stage": "sample_collection",
                    },
                    run_config,
                )
                atomic_write_json(
                    raw / f"{dataset}_PageRank_timing_{index}.json",
                    pending_sample,
                )
        atomic_write_json(json_path, pending_aggregates)
        pending_rows = [flatten(record) for record in pending_aggregates]
        pending_fields = sorted(
            {key for row in pending_rows for key in row}
        )
        pending_csv = io.StringIO(newline="")
        pending_writer = csv.DictWriter(
            pending_csv, fieldnames=pending_fields
        )
        pending_writer.writeheader()
        pending_writer.writerows(pending_rows)
        atomic_write_text(csv_path, pending_csv.getvalue())

    records = []
    for manifest, metadata, dataset, run_config, final_path in jobs:
        if args.resume and final_path.is_file():
            prior = load_resumable_json(
                final_path,
                run_config,
                artifact_kind="aggregate",
            )
            if prior.get("status") == "ok":
                record = aggregate(
                    dataset,
                    metadata,
                    prior.get("process_records", []),
                    run_config,
                )
                if record.get("status") != "ok":
                    raise RuntimeError(
                        f"refusing to reuse invalid aggregate {final_path}: "
                        f"{record.get('error', record.get('failure_kind'))}"
                    )
                records.append(record)
                print(f"[igraph-intro] reused {dataset}", flush=True)
                continue
            print(
                f"[igraph-intro] retrying incomplete aggregate {dataset}",
                flush=True,
            )

        samples = []
        for index in range(1, args.repeat + 1):
            sample_path = raw / f"{dataset}_PageRank_timing_{index}.json"
            if args.resume and sample_path.is_file():
                prior_sample = load_resumable_json(
                    sample_path,
                    run_config,
                    artifact_kind=f"sample {index}",
                )
                if prior_sample.get("status") == "ok":
                    validate_sample(
                        prior_sample,
                        dataset=dataset,
                        metadata=metadata,
                        run_config=run_config,
                        expected_index=index,
                    )
                    sample = prior_sample
                    action = "reused"
                else:
                    sample = run_sample(
                        args,
                        manifest,
                        sample_path,
                        sample_index=index,
                        environment=environment,
                        run_config=run_config,
                    )
                    action = "reran"
            else:
                sample = run_sample(
                    args,
                    manifest,
                    sample_path,
                    sample_index=index,
                    environment=environment,
                    run_config=run_config,
                )
                action = "ran"
            samples.append(sample)
            print(
                f"[igraph-intro] {dataset} sample {index}/{args.repeat} "
                f"{action}: "
                f"{sample.get('status', 'failed')}",
                flush=True,
            )
            if sample.get("status") != "ok":
                break
        record = aggregate(dataset, metadata, samples, run_config)
        atomic_write_json(final_path, record)
        records.append(record)

    atomic_write_json(json_path, records)
    rows = [flatten(record) for record in records]
    fields = sorted({key for row in rows for key in row})
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(csv_path, csv_buffer.getvalue())
    return 0 if all(record.get("status") == "ok" for record in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
