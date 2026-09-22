#!/usr/bin/env python3
"""Generate the controlled R-MAT PageRank scaling figure.

Every successful system contributes five fresh-process observations and
measures the fourth call after three preceding calls.  Current EGGPU points use
the minimum of five; external baselines use the arithmetic mean of five.  Every
whisker is the sample standard deviation (ddof=1) of those same observations.
The five raw values, mean, standard deviation, minimum, and maximum remain in
the evidence artifacts.  The nx-cugraph input uses a prepared native
CudaDiGraph; the igraph input uses a graph prepared through igraph's native
edge-list reader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

EXPECTED_SCALES = (20, 22, 24, 26)
EGGPU_TIMER_BOUNDARY = (
    "EGGPU public invocation through complete result return; "
    "benchmark validation excluded"
)
IGRAPH_TIMER_BOUNDARY = (
    "igraph PageRank public invocation through complete result return; "
    "benchmark validation excluded"
)
NX_CUGRAPH_TIMER_BOUNDARY = (
    "strict NetworkX public invocation through complete result return and "
    "CUDA synchronization; benchmark validation excluded"
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing evidence artifact: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def copy_evidence(source: Path, destination: Path) -> dict[str, object]:
    source = source.resolve()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_record = artifact_record(source)
    destination_record = artifact_record(destination)
    if source_record["sha256"] != destination_record["sha256"]:
        raise ValueError(
            f"packaged evidence hash mismatch: {source} -> {destination}"
        )
    return {
        "source": source_record,
        "packaged_copy": destination_record,
    }


def raw_sample_records(raw_dir: Path, system: str) -> list[dict[str, object]]:
    records = []
    for scale in EXPECTED_SCALES:
        dataset = f"R-MAT-S{scale}-EF16"
        for process_index in range(1, 6):
            path = (
                raw_dir
                / f"{dataset}_PageRank_timing_{process_index}.json"
            )
            record = artifact_record(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "ok":
                raise ValueError(
                    f"{system} S{scale} process {process_index} is not ok"
                )
            observed_dataset = payload.get("dataset")
            if observed_dataset != dataset:
                raise ValueError(
                    f"{system} S{scale} process {process_index} has dataset "
                    f"{observed_dataset!r}"
                )
            if (
                payload.get("function") not in (None, "PageRank")
                or payload.get("measurement") not in (None, "timing")
            ):
                raise ValueError(
                    f"{system} S{scale} process {process_index} has the wrong "
                    "function or measurement kind"
                )
            embedded_index = payload.get("timing_process_index")
            if embedded_index is not None and int(embedded_index) != process_index:
                raise ValueError(
                    f"{system} S{scale} process index mismatch: "
                    f"{embedded_index!r} != {process_index}"
                )
            if system == "EGGPU":
                samples = (payload.get("steady_e2e") or {}).get("samples")
                validation_detail = payload.get("result") or {}
                protocol_valid = (
                    int(payload.get("preceding_public_calls", -1)) == 3
                    and int(payload.get("measured_call_position", -1)) == 4
                    and payload.get("validation_outside_timer") is True
                    and (payload.get("result_validation") or {}).get("status")
                    == "pass"
                    and int(validation_detail.get("finite_count", -1))
                    == 2**scale
                    and int(validation_detail.get("nan_count", -1)) == 0
                    and int(validation_detail.get("inf_count", -1)) == 0
                    and float(validation_detail.get("minimum", -1.0)) >= 0.0
                    and math.isclose(
                        float(validation_detail.get("sum", float("nan"))),
                        1.0,
                        rel_tol=1.0e-5,
                        abs_tol=1.0e-6,
                    )
                )
            elif system == "nx-cugraph":
                samples = (payload.get("e2e") or {}).get("samples")
                details = payload.get("validation_details") or []
                validation_detail = (
                    details[0] if len(details) == 1 else {}
                )
                protocol_valid = (
                    int(payload.get("warmup_calls", -1)) == 3
                    and int(payload.get("measured_call_position", -1)) == 4
                    and payload.get("validation_outside_timer") is True
                    and payload.get("validation") == "pass"
                    and int(validation_detail.get("result_size", -1))
                    == 2**scale
                    and float(validation_detail.get("minimum", -1.0)) >= 0.0
                    and math.isclose(
                        float(
                            validation_detail.get(
                                "score_sum", float("nan")
                            )
                        ),
                        1.0,
                        rel_tol=5.0e-5,
                        abs_tol=5.0e-5,
                    )
                )
            elif system == "igraph":
                pagerank = payload.get("pagerank") or {}
                samples = [pagerank.get("seconds")]
                validation_detail = pagerank.get("validation") or {}
                protocol_valid = (
                    int(pagerank.get("warmup_calls", -1)) == 3
                    and int(pagerank.get("measured_call_position", -1)) == 4
                    and pagerank.get("timer_boundary") == IGRAPH_TIMER_BOUNDARY
                    and int(validation_detail.get("result_size", -1))
                    == 2**scale
                    and float(validation_detail.get("minimum", -1.0)) >= 0.0
                    and math.isclose(
                        float(validation_detail.get("sum", float("nan"))),
                        1.0,
                        rel_tol=1.0e-8,
                        abs_tol=1.0e-8,
                    )
                )
            else:
                raise ValueError(f"unsupported Figure 1 system: {system}")
            if (
                not protocol_valid
                or not isinstance(samples, list)
                or len(samples) != 1
                or not math.isfinite(float(samples[0]))
                or float(samples[0]) <= 0.0
            ):
                raise ValueError(
                    f"{system} S{scale} process {process_index} violates the "
                    "fourth-call raw-sample protocol"
                )
            record.update(
                {
                    "system": system,
                    "scale": scale,
                    "dataset": dataset,
                    "timing_process_index": process_index,
                    "status": "ok",
                    "public_call_seconds": float(samples[0]),
                    "complete_result_vector_validation": {
                        "status": "pass",
                        "result_size": int(
                            validation_detail.get(
                                "result_size",
                                validation_detail.get("finite_count"),
                            )
                        ),
                        "minimum": float(validation_detail["minimum"]),
                        "score_sum": float(
                            validation_detail.get(
                                "score_sum", validation_detail.get("sum")
                            )
                        ),
                    },
                }
            )
            records.append(record)
    if len(records) != 20:
        raise AssertionError(f"{system} must contribute 20 raw samples")
    return records


def require_true(value: str, label: str) -> None:
    if value.strip().lower() != "true":
        raise ValueError(f"{label} must be true, got {value!r}")


def require_close(observed, expected, label: str) -> None:
    observed_float = float(observed)
    expected_float = float(expected)
    if not math.isclose(
        observed_float,
        expected_float,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError(
            f"{label} mismatch: {observed_float} != {expected_float}"
        )


def parse_json(value: str, label: str):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error


def validate_five_sample_summary(
    system: str,
    scale: int,
    samples,
    *,
    reported_mean,
    reported_stdev,
    reported_best=None,
) -> list[float]:
    if not isinstance(samples, list) or len(samples) != 5:
        raise ValueError(f"{system} S{scale} must contain five raw samples")
    numeric = [float(value) for value in samples]
    if not all(math.isfinite(value) and value > 0.0 for value in numeric):
        raise ValueError(f"{system} S{scale} contains an invalid raw sample")
    require_close(
        reported_mean,
        statistics.mean(numeric),
        f"{system} S{scale} mean",
    )
    require_close(
        reported_stdev,
        statistics.stdev(numeric),
        f"{system} S{scale} sample standard deviation",
    )
    if reported_best is not None:
        require_close(
            reported_best,
            min(numeric),
            f"{system} S{scale} best",
        )
    return numeric


def point_from_samples(
    *,
    entries: int,
    samples: list[float],
    estimator: str,
    status: str = "ok",
) -> dict[str, object]:
    """Construct one policy-estimated plot point from a complete raw batch."""

    minimum = min(samples)
    maximum = max(samples)
    mean = statistics.mean(samples)
    if estimator == "minimum_of_five":
        reported = minimum
    elif estimator == "arithmetic_mean_of_five":
        reported = mean
    else:
        raise ValueError(f"unsupported Figure 1 estimator: {estimator}")
    return {
        "entries": int(entries),
        "latency": reported,
        "mean": mean,
        "std": statistics.stdev(samples),
        "minimum": minimum,
        "maximum": maximum,
        "samples": list(samples),
        "estimator": estimator,
        "status": status,
    }


def validate_dataset_identity(
    system: str,
    row: dict[str, str],
    scale: int,
) -> None:
    expected_dataset = f"R-MAT-S{scale}-EF16"
    if row.get("dataset") != expected_dataset:
        raise ValueError(
            f"{system} S{scale} dataset is {row.get('dataset')!r}, "
            f"expected {expected_dataset!r}"
        )
    if int(row.get("num_nodes") or -1) != 2**scale:
        raise ValueError(f"{system} S{scale} has the wrong vertex count")
    require_true(row.get("directed", ""), f"{system} S{scale} directed")


def add_unique(
    system: str,
    points: dict[int, dict[str, object]],
    scale: int,
    point: dict[str, object],
) -> None:
    if scale not in EXPECTED_SCALES:
        raise ValueError(f"{system} contains unexpected R-MAT scale S{scale}")
    if scale in points:
        raise ValueError(f"{system} contains duplicate rows for R-MAT scale S{scale}")
    latency = float(point["latency"])
    mean = float(point["mean"])
    deviation = float(point["std"])
    minimum = float(point["minimum"])
    maximum = float(point["maximum"])
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError(f"{system} S{scale} has invalid latency {latency}")
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError(f"{system} S{scale} has invalid mean {mean}")
    if not math.isfinite(deviation) or deviation < 0:
        raise ValueError(f"{system} S{scale} has invalid standard deviation {deviation}")
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum <= 0
        or maximum < minimum
    ):
        raise ValueError(
            f"{system} S{scale} has invalid minimum/maximum "
            f"{minimum}/{maximum}"
        )
    estimator = str(point.get("estimator", ""))
    if estimator == "minimum_of_five":
        expected_latency = minimum
    elif estimator == "arithmetic_mean_of_five":
        expected_latency = mean
    else:
        raise ValueError(f"{system} S{scale} has invalid estimator {estimator!r}")
    require_close(
        latency,
        expected_latency,
        f"{system} S{scale} plotted policy estimator",
    )
    points[scale] = point


def require_complete(
    system: str,
    points: dict[int, dict[str, object]],
) -> None:
    present = tuple(sorted(points))
    if present != EXPECTED_SCALES:
        raise ValueError(
            f"{system} must contain exactly S20/S22/S24/S26; found {present}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eggpu-scaling", required=True, type=Path)
    parser.add_argument("--igraph-scaling", required=True, type=Path)
    parser.add_argument("--nxcugraph-scaling", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence-csv", required=True, type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--publication-output", type=Path)
    parser.add_argument("--eggpu-raw-dir", type=Path)
    parser.add_argument("--igraph-raw-dir", type=Path)
    parser.add_argument("--nxcugraph-raw-dir", type=Path)
    parser.add_argument("--igraph-residual-evidence", type=Path)
    parser.add_argument("--igraph-qualification-record", type=Path)
    parser.add_argument("--package-evidence-dir", type=Path)
    args = parser.parse_args()
    if args.metadata_json is not None:
        missing_raw_arguments = [
            name
            for name, value in (
                ("--eggpu-raw-dir", args.eggpu_raw_dir),
                ("--igraph-raw-dir", args.igraph_raw_dir),
                ("--nxcugraph-raw-dir", args.nxcugraph_raw_dir),
            )
            if value is None
        ]
        if missing_raw_arguments:
            parser.error(
                "--metadata-json requires " + ", ".join(missing_raw_arguments)
            )
    if args.package_evidence_dir is not None and args.metadata_json is None:
        parser.error("--package-evidence-dir requires --metadata-json")

    eggpu = {}
    for row in rows(args.eggpu_scaling):
        if (
            row.get("measurement") == "timing"
            and row.get("function") == "PageRank"
            and row.get("input_family") == "controlled_rmat"
        ):
            scale = int(row["rmat_scale"])
            if row.get("status") != "ok":
                raise ValueError(
                    f"EGGPU S{scale} is not successful: {row.get('status')}"
                )
            validate_dataset_identity("EGGPU", row, scale)
            if int(row.get("rmat_edge_factor") or 0) != 16:
                raise ValueError(f"EGGPU S{scale} does not use edge factor 16")
            if int(row.get("timing_process_samples") or 0) != 5:
                raise ValueError(f"EGGPU S{scale} does not contain five processes")
            if int(row.get("first_use_calls") or 0) != 1:
                raise ValueError(f"EGGPU S{scale} does not contain one first-use call")
            if int(row.get("additional_warmup_calls") or 0) != 2:
                raise ValueError(f"EGGPU S{scale} does not contain two warmup calls")
            if int(row.get("preceding_public_calls") or 0) != 3:
                raise ValueError(f"EGGPU S{scale} does not use three preceding calls")
            if int(row.get("measured_call_position") or 0) != 4:
                raise ValueError(f"EGGPU S{scale} is not a fourth-call measurement")
            if int(row.get("measured_calls_per_process") or 0) != 1:
                raise ValueError(f"EGGPU S{scale} does not contain one measured call")
            if row.get("result_validation") != "pass":
                raise ValueError(f"EGGPU S{scale} did not pass validation")
            require_true(
                row.get("validation_outside_timer", ""),
                f"EGGPU S{scale} validation_outside_timer",
            )
            if row.get("timer_boundary") != EGGPU_TIMER_BOUNDARY:
                raise ValueError(f"EGGPU S{scale} has the wrong timer boundary")
            if row.get("batch_acceptance_status") != "accepted_complete_batch":
                raise ValueError(f"EGGPU S{scale} timing batch was not accepted")
            if row.get("paper_estimator") != "minimum_of_five":
                raise ValueError(f"EGGPU S{scale} has the wrong paper estimator")
            samples = validate_five_sample_summary(
                "EGGPU",
                scale,
                parse_json(
                    row.get("steady_e2e_samples", ""),
                    f"EGGPU S{scale} steady_e2e_samples",
                ),
                reported_mean=row["steady_e2e_mean"],
                reported_stdev=row["steady_e2e_stdev"],
                reported_best=row["steady_e2e_minimum"],
            )
            require_close(
                row["submission_e2e_seconds"],
                min(samples),
                f"EGGPU S{scale} submission estimator",
            )
            if int(row.get("result_finite_count") or -1) != int(
                row["num_nodes"]
            ):
                raise ValueError(f"EGGPU S{scale} result is not fully finite")
            add_unique(
                "EGGPU",
                eggpu,
                scale,
                point_from_samples(
                    entries=int(row["num_entries"]),
                    samples=samples,
                    estimator="minimum_of_five",
                ),
            )
    require_complete("EGGPU", eggpu)

    igraph = {}
    igraph_manifest_evidence = {}
    for row in rows(args.igraph_scaling):
        if row.get("function") != "PageRank":
            continue
        dataset = row.get("dataset", "")
        if not dataset.startswith("R-MAT-S"):
            continue
        scale = int(dataset.split("-S", 1)[1].split("-", 1)[0])
        if row.get("status") != "ok":
            raise ValueError(f"igraph S{scale} is not successful: {row.get('status')}")
        if row.get("baseline") != "igraph":
            raise ValueError(f"igraph S{scale} has the wrong baseline identity")
        expected_dataset = f"R-MAT-S{scale}-EF16"
        if dataset != expected_dataset:
            raise ValueError(
                f"igraph S{scale} dataset is {dataset!r}, "
                f"expected {expected_dataset!r}"
            )
        if int(row.get("num_nodes") or -1) != 2**scale:
            raise ValueError(f"igraph S{scale} has the wrong vertex count")
        if int(row.get("timing_process_samples") or 0) != 5:
            raise ValueError(f"igraph S{scale} does not contain five processes")
        if int(row.get("warmup_calls") or 0) != 3:
            raise ValueError(f"igraph S{scale} does not use three preceding calls")
        if int(row.get("measured_call_position") or 0) != 4:
            raise ValueError(f"igraph S{scale} is not a fourth-call measurement")
        if row.get("validation") != "pass":
            raise ValueError(f"igraph S{scale} did not pass validation")
        if row.get("timer_boundary") != IGRAPH_TIMER_BOUNDARY:
            raise ValueError(f"igraph S{scale} has the wrong timer boundary")
        if row.get("graph_construction_path") != (
            "csr_fifo_to_igraph_read_edgelist"
        ):
            raise ValueError(f"igraph S{scale} did not use the native reader path")
        if row.get("audit_mode") != "strict_audit_of_immutable_raw_batch":
            raise ValueError(f"igraph S{scale} is not a strictly audited batch")
        csr_integrity = parse_json(
            row.get("csr_integrity", ""),
            f"igraph S{scale} csr_integrity",
        )
        if (
            not isinstance(csr_integrity, dict)
            or csr_integrity.get("status") != "pass"
        ):
            raise ValueError(f"igraph S{scale} failed CSR integrity audit")
        for artifact_name in ("offsets", "indices"):
            artifact = (csr_integrity.get("artifacts") or {}).get(artifact_name)
            if (
                not isinstance(artifact, dict)
                or artifact.get("status") != "pass"
                or int(artifact.get("bytes", 0)) <= 0
                or len(str(artifact.get("sha256", ""))) != 64
            ):
                raise ValueError(
                    f"igraph S{scale} has invalid {artifact_name} integrity "
                    "evidence"
                )
        source_raw_files = parse_json(
            row.get("source_raw_files", ""),
            f"igraph S{scale} source_raw_files",
        )
        if (
            not isinstance(source_raw_files, list)
            or len(source_raw_files) != 5
            or [int(item.get("timing_process_index", -1))
                for item in source_raw_files] != [1, 2, 3, 4, 5]
            or any(
                item.get("sample_status") != "ok"
                or int(item.get("bytes", 0)) <= 0
                or len(str(item.get("sha256", ""))) != 64
                for item in source_raw_files
            )
        ):
            raise ValueError(f"igraph S{scale} raw-file audit is incomplete")
        worker_exit_evidence = parse_json(
            row.get("worker_exit_code_evidence", ""),
            f"igraph S{scale} worker_exit_code_evidence",
        )
        if (
            not isinstance(worker_exit_evidence, dict)
            or worker_exit_evidence.get("status") != "pass"
        ):
            raise ValueError(f"igraph S{scale} lacks worker-exit evidence")
        run_config = parse_json(
            row.get("run_config", ""),
            f"igraph S{scale} run_config",
        )
        manifest_path = Path(run_config["manifest_path"])
        manifest_record = artifact_record(manifest_path)
        if manifest_record["sha256"] != run_config.get("manifest_sha256"):
            raise ValueError(f"igraph S{scale} manifest hash changed after audit")
        igraph_manifest_evidence[scale] = {
            "manifest": manifest_record,
            "csr_integrity": csr_integrity,
            "source_raw_files": source_raw_files,
            "worker_exit_code_evidence": worker_exit_evidence,
        }
        require_close(row.get("damping"), 0.75, f"igraph S{scale} damping")
        require_close(
            row.get("memory_limit_gb"),
            128.0,
            f"igraph S{scale} memory limit",
        )
        process_records = parse_json(
            row.get("process_records", ""),
            f"igraph S{scale} process_records",
        )
        if not isinstance(process_records, list) or len(process_records) != 5:
            raise ValueError(f"igraph S{scale} must contain five process records")
        process_seconds = []
        for index, record in enumerate(process_records, start=1):
            pagerank = record.get("pagerank") if isinstance(record, dict) else None
            if (
                not isinstance(pagerank, dict)
                or record.get("status") != "ok"
                or record.get("dataset") != dataset
                or int(record.get("timing_process_index", -1)) != index
                or int(record.get("num_nodes", -1)) != 2**scale
                or int(record.get("num_entries", -1))
                != int(row["num_entries"])
                or int(pagerank.get("warmup_calls", -1)) != 3
                or int(pagerank.get("measured_call_position", -1)) != 4
                or pagerank.get("timer_boundary") != IGRAPH_TIMER_BOUNDARY
            ):
                raise ValueError(
                    f"igraph S{scale} process record {index} violates protocol"
                )
            process_seconds.append(float(pagerank["seconds"]))
        samples = validate_five_sample_summary(
            "igraph",
            scale,
            process_seconds,
            reported_mean=row["e2e_mean_seconds"],
            reported_stdev=row["e2e_stdev_seconds"],
            reported_best=row["e2e_best_seconds"],
        )
        add_unique(
            "igraph",
            igraph,
            scale,
            point_from_samples(
                entries=int(row["num_entries"]),
                samples=samples,
                estimator="arithmetic_mean_of_five",
            ),
        )
    require_complete("igraph", igraph)

    nxcugraph = {}
    for row in rows(args.nxcugraph_scaling):
        if row.get("function") != "PageRank":
            continue
        dataset = row.get("dataset", "")
        if not dataset.startswith("R-MAT-S"):
            continue
        scale = int(dataset.split("-S", 1)[1].split("-", 1)[0])
        if row.get("status") != "ok":
            raise ValueError(
                f"nx-cugraph S{scale} is not successful: {row.get('status')}"
            )
        validate_dataset_identity("nx-cugraph", row, scale)
        if row.get("baseline") != "nx-cugraph":
            raise ValueError(
                f"nx-cugraph S{scale} has the wrong baseline identity"
            )
        if row.get("measurement") != "timing":
            raise ValueError(f"nx-cugraph S{scale} is not a timing record")
        if row.get("input_path") != "normalized_host_csr_to_device_coo":
            raise ValueError(
                f"nx-cugraph S{scale} has an unexpected prepared input path"
            )
        if int(row.get("timing_process_samples") or 0) != 5:
            raise ValueError(f"nx-cugraph S{scale} does not contain five processes")
        if int(row.get("warmup_calls") or 0) != 3:
            raise ValueError(f"nx-cugraph S{scale} does not use three preceding calls")
        if int(row.get("measured_call_position") or 0) != 4:
            raise ValueError(f"nx-cugraph S{scale} is not a fourth-call measurement")
        if row.get("validation") != "pass":
            raise ValueError(f"nx-cugraph S{scale} did not pass validation")
        require_true(
            row.get("validation_outside_timer", ""),
            f"nx-cugraph S{scale} validation_outside_timer",
        )
        if row.get("timer_boundary") != NX_CUGRAPH_TIMER_BOUNDARY:
            raise ValueError(f"nx-cugraph S{scale} has the wrong timer boundary")
        e2e = parse_json(row.get("e2e", ""), f"nx-cugraph S{scale} e2e")
        if not isinstance(e2e, dict):
            raise ValueError(f"nx-cugraph S{scale} e2e is not an object")
        samples = validate_five_sample_summary(
            "nx-cugraph",
            scale,
            e2e.get("samples"),
            reported_mean=row["e2e_mean_seconds"],
            reported_stdev=row["e2e_stdev_seconds"],
            reported_best=row["e2e_best_seconds"],
        )
        process_records = parse_json(
            row.get("timing_process_records", ""),
            f"nx-cugraph S{scale} timing_process_records",
        )
        if not isinstance(process_records, list) or len(process_records) != 5:
            raise ValueError(
                f"nx-cugraph S{scale} must contain five process records"
            )
        for index, record in enumerate(process_records, start=1):
            record_samples = (
                (record.get("e2e") or {}).get("samples")
                if isinstance(record, dict)
                else None
            )
            warmups = record.get("warmup_details") if isinstance(record, dict) else None
            if (
                not isinstance(record, dict)
                or record.get("status") != "ok"
                or record.get("measurement") != "timing"
                or record.get("baseline") != "nx-cugraph"
                or record.get("dataset") != dataset
                or int(record.get("timing_process_index", -1)) != index
                or int(record.get("num_nodes", -1)) != 2**scale
                or int(record.get("num_entries", -1))
                != int(row["num_entries"])
                or int(record.get("warmup_calls", -1)) != 3
                or int(record.get("measured_call_position", -1)) != 4
                or record.get("validation") != "pass"
                or record.get("validation_outside_timer") is not True
                or record.get("timer_boundary") != NX_CUGRAPH_TIMER_BOUNDARY
                or not isinstance(record_samples, list)
                or len(record_samples) != 1
                or not isinstance(warmups, list)
                or [item.get("call_position") for item in warmups]
                != [1, 2, 3]
            ):
                raise ValueError(
                    f"nx-cugraph S{scale} process record {index} "
                    "violates protocol"
                )
            require_close(
                record_samples[0],
                samples[index - 1],
                f"nx-cugraph S{scale} process record {index}",
            )
        add_unique(
            "nx-cugraph",
            nxcugraph,
            scale,
            point_from_samples(
                entries=int(row["num_entries"]),
                samples=samples,
                estimator="arithmetic_mean_of_five",
            ),
        )
    require_complete("nx-cugraph", nxcugraph)

    for scale in EXPECTED_SCALES:
        expected_entries = int(eggpu[scale]["entries"])
        for system, points in (("igraph", igraph), ("nx-cugraph", nxcugraph)):
            if int(points[scale]["entries"]) != expected_entries:
                raise ValueError(
                    f"{system} S{scale} entries do not match EGGPU: "
                    f"{points[scale]['entries']} != {expected_entries}"
                )

    scales = list(EXPECTED_SCALES)
    evidence = []
    for scale in scales:
        systems = [
            ("EGGPU", eggpu[scale]),
            ("igraph", igraph.get(scale)),
        ]
        systems.insert(1, ("nx-cugraph", nxcugraph.get(scale)))
        for system, data in systems:
            if data is None:
                continue
            evidence.append(
                {
                    "system": system,
                    "scale": scale,
                    "vertices": 2**scale,
                    "adjacency_entries": data["entries"],
                    "directed": "yes",
                    "pagerank_public_call_seconds": (
                        "" if data["latency"] is None else data["latency"]
                    ),
                    "arithmetic_mean_public_call_seconds": data["mean"],
                    "sample_standard_deviation_seconds": (
                        "" if data["std"] is None else data["std"]
                    ),
                    "minimum_public_call_seconds": data["minimum"],
                    "maximum_public_call_seconds": data["maximum"],
                    "raw_public_call_samples_seconds": json.dumps(
                        data["samples"], separators=(",", ":")
                    ),
                    "reported_time_relative_to_eggpu": (
                        float(data["latency"])
                        / float(eggpu[scale]["latency"])
                    ),
                    "whisker_lower_seconds": max(
                        float(data["latency"]) - float(data["std"]),
                        0.05 * float(data["latency"]),
                    ),
                    "whisker_upper_seconds": (
                        float(data["latency"]) + float(data["std"])
                    ),
                    "whisker_semantics": (
                        "symmetric sample standard deviation (ddof=1) over "
                        "the same five fresh processes; a lower endpoint "
                        "crossing zero is clipped for the logarithmic plot"
                    ),
                    "status": data["status"],
                    "estimator": (
                        (
                            "minimum of five fresh processes"
                            if system == "EGGPU"
                            else "arithmetic mean of five fresh processes"
                        )
                        if data["status"] == "ok"
                        else "not applicable"
                    ),
                    "call_protocol": (
                        "fourth call after three preceding calls"
                        if data["status"] == "ok"
                        else "not applicable"
                    ),
                    "prepared_graph_form": {
                        "EGGPU": "versioned EGGPU graph state",
                        "nx-cugraph": "native CudaDiGraph",
                        "igraph": "native Read_Edgelist graph",
                    }[system],
                    "graph_preparation_in_timed_region": "no",
                    "benchmark_validation_in_timed_region": "no",
                    "result_validation": (
                        "complete returned vector: one finite nonnegative "
                        "score per vertex and total score approximately one"
                    ),
                    "timer_boundary": (
                        "complete public PageRank result return; "
                        "GPU systems synchronized"
                    ),
                    "damping": 0.75,
                    "solver_convergence": "system native",
                }
            )
    args.evidence_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evidence[0]))
        writer.writeheader()
        writer.writerows(evidence)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(3.48, 2.15))
    blue = "#2878B5"
    orange = "#D95F02"
    purple = "#7A5195"
    eggpu_line = axis.errorbar(
        scales,
        [eggpu[scale]["latency"] for scale in scales],
        yerr=[
            [
                min(
                    eggpu[scale]["std"],
                    0.95 * eggpu[scale]["latency"],
                )
                for scale in scales
            ],
            [
                eggpu[scale]["std"]
                for scale in scales
            ],
        ],
        color=blue,
        marker="o",
        linewidth=1.8,
        markersize=4.8,
        elinewidth=0.7,
        capthick=0.7,
        capsize=2.0,
        label="EGGPU (GPU)",
    )
    common = [scale for scale in scales if igraph.get(scale, {}).get("status") == "ok"]
    igraph_line = axis.errorbar(
        common,
        [igraph[scale]["latency"] for scale in common],
        yerr=[
            [
                min(
                    igraph[scale]["std"],
                    0.95 * igraph[scale]["latency"],
                )
                for scale in common
            ],
            [
                igraph[scale]["std"]
                for scale in common
            ],
        ],
        color=orange,
        marker="s",
        linewidth=1.8,
        markersize=4.5,
        elinewidth=0.7,
        capthick=0.7,
        capsize=2.0,
        label="igraph (CPU)",
    )
    nx_common = [
        scale
        for scale in scales
        if nxcugraph.get(scale, {}).get("status") == "ok"
    ]
    if nx_common:
        nxcugraph_line = axis.errorbar(
            nx_common,
            [nxcugraph[scale]["latency"] for scale in nx_common],
            yerr=[
                [
                    min(
                        nxcugraph[scale]["std"],
                        0.95 * nxcugraph[scale]["latency"],
                    )
                    for scale in nx_common
                ],
                [
                    nxcugraph[scale]["std"]
                    for scale in nx_common
                ],
            ],
            color=purple,
            marker="D",
            linewidth=1.8,
            markersize=4.2,
            elinewidth=0.7,
            capthick=0.7,
            capsize=2.0,
            label="nx-cugraph (GPU)",
        )
    else:
        nxcugraph_line = None
    axis.set_yscale("log")
    successful = [
        data
        for system_rows in (eggpu, nxcugraph, igraph)
        for data in system_rows.values()
        if data.get("status") == "ok"
    ]
    upper = max(data["latency"] + data["std"] for data in successful)
    lower = min(
        max(data["latency"] - data["std"], 0.05 * data["latency"])
        for data in successful
    ) * 0.55
    axis.set_ylim(lower, upper * 1.65)
    axis.set_xticks(scales, [f"S{scale}" for scale in scales])
    axis.set_xlabel(
        "R-MAT scale ($2^S$ vertices; edge factor 16)",
        labelpad=17,
    )
    axis.set_ylabel("PageRank public-call time (s)")
    axis.grid(
        axis="y",
        which="major",
        color="#D7DCE2",
        linewidth=0.55,
    )
    axis.grid(
        axis="y",
        which="minor",
        color="#EDF0F3",
        linewidth=0.35,
    )
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    legend_handles = [igraph_line]
    if nxcugraph_line is not None:
        legend_handles.append(nxcugraph_line)
    legend_handles.append(eggpu_line)
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.45,
        columnspacing=0.9,
        handletextpad=0.4,
        fontsize=6.8,
    )

    entry_labels = {
        scale: (
            f"{eggpu[scale]['entries'] / 1e6:.0f}M"
            if eggpu[scale]["entries"] < 1e9
            else f"{eggpu[scale]['entries'] / 1e9:.2f}B"
        )
        for scale in scales
    }
    for scale in scales:
        axis.text(
            scale,
            -0.155,
            entry_labels[scale],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.7,
            color="#666666",
            clip_on=False,
        )

    fig.subplots_adjust(left=0.205, right=0.99, top=0.84, bottom=0.30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    publication_artifacts = []
    if args.publication_output is not None:
        args.publication_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.output, args.publication_output)
        publication_png = args.publication_output.with_suffix(".png")
        shutil.copyfile(args.output.with_suffix(".png"), publication_png)
        publication_artifacts = [
            artifact_record(args.publication_output),
            artifact_record(publication_png),
        ]

    if args.metadata_json is not None:
        residual_evidence = None
        if args.igraph_residual_evidence is not None:
            residual_record = artifact_record(args.igraph_residual_evidence)
            residual_payload = json.loads(
                args.igraph_residual_evidence.read_text(encoding="utf-8")
            )
            if (
                residual_payload.get("status") != "pass"
                or int(residual_payload.get("num_nodes", -1)) != 2**26
                or int(residual_payload.get("num_entries", -1))
                != int(igraph[26]["entries"])
                or not math.isclose(
                    float(residual_payload.get("alpha", float("nan"))),
                    0.75,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or not math.isclose(
                    float(
                        residual_payload.get(
                            "mean_residual_tolerance", float("nan")
                        )
                    ),
                    1.0e-6,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or float(residual_payload.get("mean_residual", float("inf")))
                > 1.0e-6
            ):
                raise ValueError(
                    "igraph PageRank residual qualification did not pass the "
                    "S26 fixed-point gate"
                )
            residual_record["evidence"] = residual_payload
            residual_record["qualified_result"] = artifact_record(
                Path(residual_payload["result_path"])
            )
            residual_evidence = residual_record

        raw_samples = []
        for raw_dir, system in (
            (args.eggpu_raw_dir, "EGGPU"),
            (args.nxcugraph_raw_dir, "nx-cugraph"),
            (args.igraph_raw_dir, "igraph"),
        ):
            raw_samples.extend(raw_sample_records(raw_dir.resolve(), system))
        if len(raw_samples) != 60:
            raise AssertionError("the three systems must contribute 60 raw samples")
        point_sets = {
            "EGGPU": eggpu,
            "nx-cugraph": nxcugraph,
            "igraph": igraph,
        }
        for system, points in point_sets.items():
            for scale in EXPECTED_SCALES:
                observed = [
                    float(record["public_call_seconds"])
                    for record in raw_samples
                    if record["system"] == system
                    and int(record["scale"]) == scale
                ]
                expected = [float(value) for value in points[scale]["samples"]]
                if len(observed) != 5 or any(
                    not math.isclose(
                        actual,
                        target,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-15,
                    )
                    for actual, target in zip(observed, expected)
                ):
                    raise ValueError(
                        f"{system} S{scale} raw samples differ from the "
                        "validated aggregate"
                    )
        current_igraph_hashes = {
            (int(record["scale"]), int(record["timing_process_index"])): record[
                "sha256"
            ]
            for record in raw_samples
            if record["system"] == "igraph"
        }
        audited_igraph_hashes = {
            (scale, int(record["timing_process_index"])): record["sha256"]
            for scale in EXPECTED_SCALES
            for record in igraph_manifest_evidence[scale]["source_raw_files"]
        }
        if current_igraph_hashes != audited_igraph_hashes:
            raise ValueError(
                "current igraph raw sample hashes differ from the strict audit"
            )

        qualification_record = None
        if args.igraph_qualification_record is not None:
            qualification_record = artifact_record(
                args.igraph_qualification_record
            )
            qualification_payload = json.loads(
                args.igraph_qualification_record.read_text(encoding="utf-8")
            )
            qualification_pagerank = qualification_payload.get("pagerank") or {}
            qualification_result = (
                qualification_pagerank.get("qualification_result") or {}
            )
            if (
                qualification_payload.get("status") != "ok"
                or qualification_payload.get("dataset") != "R-MAT-S26-EF16"
                or int(qualification_payload.get("num_nodes", -1)) != 2**26
                or int(qualification_payload.get("num_entries", -1))
                != int(igraph[26]["entries"])
                or not math.isclose(
                    float(qualification_pagerank.get("damping", float("nan"))),
                    0.75,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or int(qualification_pagerank.get("warmup_calls", -1)) != 0
                or int(
                    qualification_pagerank.get(
                        "measured_call_position", -1
                    )
                )
                != 1
                or int(qualification_result.get("num_values", -1)) != 2**26
                or qualification_result.get("dtype") != "float32"
                or qualification_result.get("measurement_window")
                != "after_measured_timer"
            ):
                raise ValueError(
                    "igraph qualification worker did not produce the required "
                    "S26 complete PageRank vector"
                )
            if (
                residual_evidence is not None
                and Path(qualification_result["path"]).resolve()
                != Path(residual_payload["result_path"]).resolve()
            ):
                raise ValueError(
                    "igraph worker and fixed-point evidence reference different "
                    "PageRank vectors"
                )

        packaged_evidence = []
        if args.package_evidence_dir is not None:
            package_dir = args.package_evidence_dir.resolve()
            aggregate_sources = (
                ("eggpu_scaling_all.csv", args.eggpu_scaling),
                (
                    "eggpu_scaling_all.json",
                    args.eggpu_scaling.with_suffix(".json"),
                ),
                ("nxcugraph_large_matrix.csv", args.nxcugraph_scaling),
                (
                    "nxcugraph_large_matrix.json",
                    args.nxcugraph_scaling.with_suffix(".json"),
                ),
                ("igraph_scaling_audited.csv", args.igraph_scaling),
                (
                    "igraph_scaling_audited.json",
                    args.igraph_scaling.with_suffix(".json"),
                ),
            )
            for destination_name, source in aggregate_sources:
                packaged_evidence.append(
                    copy_evidence(
                        source,
                        package_dir / "aggregates" / destination_name,
                    )
                )
            for record in raw_samples:
                source = Path(record["path"])
                system_directory = str(record["system"]).lower().replace(
                    "-", "_"
                )
                packaged_evidence.append(
                    copy_evidence(
                        source,
                        package_dir
                        / "raw"
                        / system_directory
                        / source.name,
                    )
                )
            for scale in EXPECTED_SCALES:
                source = Path(
                    igraph_manifest_evidence[scale]["manifest"]["path"]
                )
                packaged_evidence.append(
                    copy_evidence(
                        source,
                        package_dir / "manifests" / source.name,
                    )
                )
            protocol_sources = {
                "generate_intro_rmat_scaling_20260729.py": Path(__file__),
                "run_igraph_intro_scaling_fifo.py": Path(__file__).with_name(
                    "run_igraph_intro_scaling_fifo.py"
                ),
                "igraph_read_edgelist_fifo.py": (
                    Path(__file__).with_name("diagnostics")
                    / "igraph_read_edgelist_fifo.py"
                ),
                "audit_igraph_intro_raw_batch.py": Path(__file__).with_name(
                    "audit_igraph_intro_raw_batch.py"
                ),
                "pagerank_residual_validation.py": Path(__file__).with_name(
                    "pagerank_residual_validation.py"
                ),
            }
            for destination_name, source in protocol_sources.items():
                packaged_evidence.append(
                    copy_evidence(
                        source,
                        package_dir / "protocol" / destination_name,
                    )
                )
            if args.igraph_residual_evidence is not None:
                packaged_evidence.append(
                    copy_evidence(
                        args.igraph_residual_evidence,
                        package_dir
                        / "qualification"
                        / "igraph_s26_fixed_point.json",
                    )
                )
            if args.igraph_qualification_record is not None:
                packaged_evidence.append(
                    copy_evidence(
                        args.igraph_qualification_record,
                        package_dir
                        / "qualification"
                        / "igraph_s26_worker.json",
                    )
                )

        metadata = {
            "artifact_version": (
                "figure1-three-system-pagerank-call4-mixed-estimator-v3"
            ),
            "experiment": (
                "prepared-graph PageRank scaling on controlled directed "
                "R-MAT graphs"
            ),
            "systems": ["igraph (CPU)", "nx-cugraph (GPU)", "EGGPU (GPU)"],
            "scales": list(EXPECTED_SCALES),
            "edge_factor": 16,
            "damping": 0.75,
            "reported_estimator": (
                "current EGGPU uses the minimum of five fresh-process "
                "observations; nx-cugraph and igraph use their arithmetic mean"
            ),
            "reported_estimator_id": (
                "eggpu_minimum_external_baseline_mean_five_processes"
            ),
            "reported_estimator_scope": (
                "EGGPU, nx-cugraph, and igraph at S20/S22/S24/S26"
            ),
            "error_bar": (
                "sample standard deviation (ddof=1) of the same five "
                "fresh-process observations"
            ),
            "sample_standard_deviation_role": (
                "rendered as the figure whisker and retained in CSV evidence"
            ),
            "per_process_protocol": (
                "measure the fourth public PageRank call after three preceding "
                "calls on the prepared graph"
            ),
            "measurement_boundary": (
                "public PageRank invocation through complete result return; "
                "GPU systems synchronized"
            ),
            "graph_preparation_in_timed_region": False,
            "benchmark_validation_in_timed_region": False,
            "prepared_graph_forms": {
                "igraph": "native graph built by Graph.Read_Edgelist",
                "nx-cugraph": "native CudaDiGraph",
                "EGGPU": "versioned EGGPU graph state",
            },
            "solver_convergence": "system native",
            "result_validation_protocol": {
                "all_timed_processes": (
                    "outside the timer, scan the complete public PageRank "
                    "result and require one finite nonnegative score per vertex "
                    "with total score approximately one"
                ),
                "igraph_s26_additional_qualification": (
                    "a separately emitted complete float32 vector is checked "
                    "against the PageRank fixed-point equation at damping 0.75 "
                    "with mean absolute residual tolerance 1e-6"
                ),
            },
            "points": evidence,
            "input_aggregates": {
                "EGGPU": {
                    "csv": artifact_record(args.eggpu_scaling),
                    "json": artifact_record(
                        args.eggpu_scaling.with_suffix(".json")
                    ),
                },
                "nx-cugraph": {
                    "csv": artifact_record(args.nxcugraph_scaling),
                    "json": artifact_record(
                        args.nxcugraph_scaling.with_suffix(".json")
                    ),
                },
                "igraph": {
                    "csv": artifact_record(args.igraph_scaling),
                    "json": artifact_record(
                        args.igraph_scaling.with_suffix(".json")
                    ),
                },
            },
            "raw_samples": raw_samples,
            "raw_sample_count": len(raw_samples),
            "graph_input_evidence": [
                {
                    "scale": scale,
                    **igraph_manifest_evidence[scale],
                }
                for scale in EXPECTED_SCALES
            ],
            "igraph_residual_qualification": residual_evidence,
            "igraph_qualification_worker": qualification_record,
            "generated_artifacts": [
                artifact_record(args.output),
                artifact_record(args.output.with_suffix(".png")),
                artifact_record(args.evidence_csv),
            ],
            "publication_copies": publication_artifacts,
            "packaged_evidence": packaged_evidence,
            "protocol_sources": {
                "generator": artifact_record(Path(__file__)),
                "igraph_runner": artifact_record(
                    Path(__file__).with_name(
                        "run_igraph_intro_scaling_fifo.py"
                    )
                ),
                "igraph_worker": artifact_record(
                    Path(__file__).with_name("diagnostics")
                    / "igraph_read_edgelist_fifo.py"
                ),
                "igraph_raw_auditor": artifact_record(
                    Path(__file__).with_name(
                        "audit_igraph_intro_raw_batch.py"
                    )
                ),
                "pagerank_residual_validator": artifact_record(
                    Path(__file__).with_name(
                        "pagerank_residual_validation.py"
                    )
                ),
            },
        }
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        with args.metadata_json.open("w", encoding="utf-8") as handle:
            json.dump(
                metadata,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")


if __name__ == "__main__":
    main()
