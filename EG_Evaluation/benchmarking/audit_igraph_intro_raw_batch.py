#!/usr/bin/env python3
"""Strictly audit completed Figure 1 igraph raw batches.

The first native-reader batches were launched before the coordinator embedded
its protocol signature in every sample, while later batches contain the
signature at collection time.  This utility preserves both kinds of raw file,
verifies any embedded signature before promotion, validates all five nested
records with the strict coordinator checks, records their hashes, and writes a
separate audited aggregate for plotting.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import run_igraph_intro_scaling_fifo as protocol


EXPECTED_NORMALIZATION = (
    "directed edge stream; self-loops removed; duplicate directed pairs "
    "removed; CSR destinations sorted"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--process-timeout", type=float, default=2400.0)
    return parser.parse_args()


def validate_csr_integrity(manifest_path: Path, metadata: dict) -> dict:
    if metadata.get("normalization") != EXPECTED_NORMALIZATION:
        raise ValueError(
            f"{metadata.get('name')}: unexpected normalization declaration"
        )
    if metadata.get("node_labels") != "zero_based_contiguous":
        raise ValueError(
            f"{metadata.get('name')}: node labels are not zero-based contiguous"
        )
    if metadata.get("directed") is not True:
        raise ValueError(f"{metadata.get('name')}: Figure 1 input is not directed")
    if int(metadata["num_edges"]) != int(metadata["num_entries"]):
        raise ValueError(
            f"{metadata.get('name')}: directed edge and CSR-entry counts differ"
        )

    artifact_records = {}
    artifacts = metadata.get("csr_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{metadata.get('name')}: csr_artifacts are missing")
    for name, top_level_key in (
        ("offsets", "offsets_path"),
        ("indices", "indices_path"),
    ):
        declaration = artifacts.get(name)
        if not isinstance(declaration, dict):
            raise ValueError(
                f"{metadata.get('name')}: {name} artifact declaration is missing"
            )
        relative_path = declaration.get("path")
        if not relative_path or relative_path != metadata.get(top_level_key):
            raise ValueError(
                f"{metadata.get('name')}: {name} artifact path is inconsistent"
            )
        artifact_path = (manifest_path.parent / relative_path).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing CSR artifact: {artifact_path}")
        observed_bytes = artifact_path.stat().st_size
        expected_bytes = int(declaration.get("bytes", -1))
        if observed_bytes != expected_bytes:
            raise ValueError(
                f"{metadata.get('name')}: {name} bytes {observed_bytes} "
                f"!= {expected_bytes}"
            )
        observed_sha256 = protocol.sha256_file(artifact_path)
        expected_sha256 = str(declaration.get("sha256", "")).lower()
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"{metadata.get('name')}: {name} SHA-256 mismatch"
            )
        artifact_records[name] = {
            "path": str(artifact_path),
            "bytes": observed_bytes,
            "sha256": observed_sha256,
            "status": "pass",
        }
    return {
        "status": "pass",
        "normalization": metadata["normalization"],
        "node_labels": metadata["node_labels"],
        "directed": True,
        "num_edges_equals_num_entries": True,
        "artifacts": artifact_records,
    }


def main() -> int:
    args = parse_args()
    protocol_args = SimpleNamespace(
        out_dir=args.output_dir,
        repeat=protocol.FIGURE1_REPEAT,
        warmup_calls=protocol.FIGURE1_WARMUP_CALLS,
        memory_limit_gb=protocol.FIGURE1_MEMORY_LIMIT_GB,
        process_timeout=args.process_timeout,
    )
    environment = protocol.child_environment(protocol_args)
    records = []
    for manifest in args.manifests:
        manifest = manifest.resolve()
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        dataset = str(metadata["name"])
        csr_integrity = validate_csr_integrity(manifest, metadata)
        run_config = protocol.make_run_config(
            protocol_args,
            manifest,
            environment,
        )
        samples = []
        raw_hashes = []
        embedded_signatures = 0
        for index in range(1, protocol.FIGURE1_REPEAT + 1):
            raw_path = (
                args.raw_dir
                / f"{dataset}_PageRank_timing_{index}.json"
            )
            if not raw_path.is_file():
                raise FileNotFoundError(f"missing raw timing sample: {raw_path}")
            sample = json.loads(raw_path.read_text(encoding="utf-8"))
            embedded_signature = sample.get("protocol_signature")
            if embedded_signature is not None:
                embedded_signatures += 1
                if embedded_signature != run_config["protocol_signature"]:
                    raise ValueError(
                        f"{dataset} sample {index}: embedded protocol signature "
                        "does not match the audited configuration"
                    )
                if sample.get("run_config") != run_config:
                    raise ValueError(
                        f"{dataset} sample {index}: embedded run_config does "
                        "not match the audited configuration"
                    )
            raw_hashes.append(
                {
                    "timing_process_index": index,
                    "path": str(raw_path.resolve()),
                    "bytes": raw_path.stat().st_size,
                    "sha256": protocol.sha256_file(raw_path),
                    "sample_status": sample.get("status"),
                    "embedded_protocol_signature": embedded_signature,
                    "protocol_signature_observed_at_collection": (
                        embedded_signature is not None
                    ),
                    "worker_exit_gate": (
                        "coordinator accepted the sample only after a zero "
                        "worker return code and a complete worker-output file"
                    ),
                    "worker_exit_code_directly_recorded": sample.get(
                        "worker_exit_code", sample.get("returncode")
                    ),
                }
            )
            sample["timing_process_index"] = index
            sample["expected_timing_processes"] = protocol.FIGURE1_REPEAT
            samples.append(protocol.attach_run_config(sample, run_config))

        record = protocol.aggregate(
            dataset,
            metadata,
            samples,
            run_config,
        )
        if record.get("status") != "ok":
            raise RuntimeError(
                f"strict audit failed for {dataset}: "
                f"{record.get('error', record.get('failure_kind'))}"
            )
        record["audit_mode"] = "strict_audit_of_immutable_raw_batch"
        record["csr_integrity"] = csr_integrity
        record["source_raw_files"] = raw_hashes
        if embedded_signatures == protocol.FIGURE1_REPEAT:
            record["source_signature_note"] = (
                "Every immutable raw sample carried the audited coordinator "
                "signature and run configuration at collection time."
            )
        elif embedded_signatures == 0:
            record["source_signature_note"] = (
                "These immutable raw files predate embedded coordinator "
                "signatures. The current worker differs only by an unused "
                "optional result-output argument; the measured path and timer "
                "boundary are unchanged."
            )
        else:
            raise ValueError(
                f"{dataset}: only {embedded_signatures}/"
                f"{protocol.FIGURE1_REPEAT} raw samples contain an embedded "
                "protocol signature"
            )
        record["worker_exit_code_evidence"] = {
            "status": "pass",
            "directly_recorded_per_sample": all(
                item["worker_exit_code_directly_recorded"] == 0
                for item in raw_hashes
            ),
            "acceptance_gate": (
                "run_sample accepts a worker record only when subprocess "
                "returncode is zero and the worker output exists; all five "
                "samples passed strict aggregate validation"
            ),
            "coordinator_source_path": str(
                Path(protocol.__file__).resolve()
            ),
            "coordinator_source_sha256": protocol.sha256_file(
                Path(protocol.__file__).resolve()
            ),
        }
        records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol.atomic_write_json(
        args.output_dir / "igraph_intro_scaling_fifo.audited.json",
        records,
    )
    flattened = [protocol.flatten(record) for record in records]
    fields = sorted({key for row in flattened for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(flattened)
    protocol.atomic_write_text(
        args.output_dir / "igraph_intro_scaling_fifo.audited.csv",
        buffer.getvalue(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
