#!/usr/bin/env python3
"""Audit EasyGraph CPU/C++ and EGGPU backend separation in a benchmark result.

The benchmark intentionally runs EGGPU, easygraph-cpp, and easygraph-cpu in
separate child processes.  This script verifies that the logs and result rows
match that contract, so later paper tables can cite a concrete audit artifact
instead of relying on runner intent.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


EXPECTED = {
    "EGGPU": {
        "mode": "gpu",
        "enable_gpu": "TRUE",
        "gpu_backend": "mine",
        "warmup": "2",
    },
    "easygraph-cpp": {
        "mode": "cpp",
        "enable_gpu": "FALSE",
        "gpu_backend": "",
        "warmup": "0",
    },
    "easygraph-cpu": {
        "mode": "cpu",
        "enable_gpu": "FALSE",
        "gpu_backend": "",
        "warmup": "0",
    },
}

STRUCTURAL_HOLES = {"effectivesize", "efficiency", "constraint", "hierarchy"}
ACCEPTED_SCALE_SKIP_NOTE = "exact all-source Closeness skipped by symmetric scale guard"

MODE_RE = re.compile(
    r"\[easygraph-mode\]\s+backend=(?P<backend>\S+)\s+"
    r"mode=(?P<mode>\S+)\s+"
    r"EASYGRAPH_ENABLE_GPU=(?P<enable>\S*)\s+"
    r"EASYGRAPH_GPU_BACKEND=(?P<gpu_backend>\S*)"
)
PARAM_RE = re.compile(r"\[params\].*easygraph_warmup=(?P<warmup>\d+)")
RESULT_RE = re.compile(r"RESULT_JSON\s+(?P<json>\{.*\})")


def _extract_function_from_log(path: Path, baseline: str) -> str:
    prefix = f"library_{baseline}_"
    name = path.name
    if name.startswith(prefix) and name.endswith(".log"):
        return name[len(prefix) : -4]
    return ""


def _read_log(path: Path) -> str:
    return path.read_text(errors="replace")


def audit_logs(result_dir: Path) -> tuple[list[dict], Counter]:
    logs_dir = result_dir / "logs"
    rows: list[dict] = []
    summary: Counter = Counter()

    for baseline, expected in EXPECTED.items():
        for log in sorted(logs_dir.glob(f"*/library_{baseline}_*.log")):
            text = _read_log(log)
            dataset = log.parent.name
            function = _extract_function_from_log(log, baseline)
            mode_match = MODE_RE.search(text)
            param_match = PARAM_RE.search(text)
            result_backends = []
            result_statuses = Counter()
            result_notes = []
            for match in RESULT_RE.finditer(text):
                try:
                    obj = json.loads(match.group("json"))
                except Exception:
                    continue
                result_backends.append(str(obj.get("backend", "")))
                result_statuses[str(obj.get("status", ""))] += 1
                note = str(obj.get("notes", ""))
                if note:
                    result_notes.append(note)

            checks = []
            timed_out_before_header = (not mode_match) and "TIMEOUT_TOO_LONG" in text
            skipped_before_header = (not mode_match) and ACCEPTED_SCALE_SKIP_NOTE in text
            if timed_out_before_header:
                checks.append("timeout_before_backend_entry")
                mode_backend = mode = enable = gpu_backend = ""
            elif skipped_before_header:
                checks.append("skipped_before_backend_entry")
                mode_backend = mode = enable = gpu_backend = ""
            elif not mode_match:
                checks.append("missing_easygraph_mode_header")
                mode_backend = mode = enable = gpu_backend = ""
            else:
                mode_backend = mode_match.group("backend")
                mode = mode_match.group("mode")
                enable = mode_match.group("enable")
                gpu_backend = mode_match.group("gpu_backend")
                if mode_backend != baseline:
                    checks.append(f"mode_backend={mode_backend}")
                if mode != expected["mode"]:
                    checks.append(f"mode={mode}")
                if enable != expected["enable_gpu"]:
                    checks.append(f"EASYGRAPH_ENABLE_GPU={enable}")
                if gpu_backend != expected["gpu_backend"]:
                    checks.append(f"EASYGRAPH_GPU_BACKEND={gpu_backend}")

            warmup = param_match.group("warmup") if param_match else ""
            if timed_out_before_header or skipped_before_header:
                pass
            elif warmup != expected["warmup"]:
                checks.append(f"easygraph_warmup={warmup}")

            unexpected_result_backend = sorted(set(result_backends) - {baseline})
            if unexpected_result_backend:
                checks.append("result_backend_mismatch=" + ",".join(unexpected_result_backend))

            skip_note_ok = False
            if baseline == "easygraph-cpp" and function.lower() in STRUCTURAL_HOLES:
                skip_note_ok = any(
                    "skipped to keep CPU C++ baseline isolated" in note for note in result_notes
                )
                if not skip_note_ok:
                    checks.append("missing_structural_hole_isolation_skip_note")

            if timed_out_before_header and checks == ["timeout_before_backend_entry"]:
                status = "timeout"
                summary["timeout"] += 1
            elif skipped_before_header and checks == ["skipped_before_backend_entry"]:
                status = "skipped"
                summary["skipped"] += 1
            elif checks:
                status = "fail"
                summary["fail"] += 1
            else:
                status = "ok"
                summary["ok"] += 1

            rows.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "baseline": baseline,
                    "status": status,
                    "checks": "; ".join(checks),
                    "mode_backend": mode_backend,
                    "mode": mode,
                    "EASYGRAPH_ENABLE_GPU": enable,
                    "EASYGRAPH_GPU_BACKEND": gpu_backend,
                    "easygraph_warmup": warmup,
                    "result_statuses": dict(result_statuses),
                    "log": str(log),
                }
            )

    return rows, summary


def audit_result_rows(result_dir: Path) -> list[dict]:
    rows = []
    long_path = result_dir / "results_long.csv"
    long_df = pd.read_csv(long_path) if long_path.exists() else pd.DataFrame()
    e2e_path = result_dir / "results_e2e.csv"
    expected_eggpu_rows = None
    if not long_df.empty:
        e2e_df = long_df[long_df["metric"] == "e2e"].copy()
    elif e2e_path.exists():
        e2e_df = pd.read_csv(e2e_path)
    else:
        e2e_df = pd.DataFrame()
    if not e2e_df.empty:
        eggpu_e2e = e2e_df[e2e_df["baseline"] == "EGGPU"]
        datasets = set(eggpu_e2e["dataset"].dropna().astype(str))
        functions = set(eggpu_e2e["function"].dropna().astype(str))
        if datasets and functions:
            expected_eggpu_rows = len(datasets) * len(functions)

    for metric in ("build", "e2e", "kernel"):
        path = result_dir / f"results_{metric}.csv"
        if not long_df.empty:
            df = long_df[long_df["metric"] == metric].copy()
        elif path.exists():
            df = pd.read_csv(path)
        else:
            rows.append(
                {
                    "metric": metric,
                    "baseline": "",
                    "rows": 0,
                    "ok": 0,
                    "timeout": 0,
                    "skipped": 0,
                    "status": "fail",
                    "note": f"missing {path.name}",
                }
            )
            continue
        for baseline in EXPECTED:
            bdf = df[df["baseline"] == baseline]
            counts = bdf["status"].fillna("").value_counts().to_dict()
            status = "ok" if len(bdf) else "fail"
            note = ""
            if baseline == "EGGPU" and metric in {"e2e", "kernel"}:
                ok_count = int(counts.get("ok", 0))
                accepted_skip_count = 0
                accepted_skip = bdf["status"].fillna("").eq("skipped")
                if "skip_reason" in bdf.columns:
                    accepted_skip = accepted_skip & bdf["skip_reason"].fillna("").eq("exact_scale_guard")
                elif "notes" in bdf.columns:
                    accepted_skip = accepted_skip & bdf["notes"].fillna("").str.contains(ACCEPTED_SCALE_SKIP_NOTE, regex=False)
                else:
                    accepted_skip = accepted_skip & False
                accepted_skip_count = int(accepted_skip.sum())
                expected = expected_eggpu_rows if expected_eggpu_rows is not None else ok_count
                if ok_count + accepted_skip_count != expected:
                    status = "fail"
                    note = f"expected {expected} ok-or-accepted-skipped EGGPU {metric} rows, got ok={ok_count}, accepted_skipped={accepted_skip_count}"
            rows.append(
                {
                    "metric": metric,
                    "baseline": baseline,
                    "rows": int(len(bdf)),
                    "ok": int(counts.get("ok", 0)),
                    "timeout": int(counts.get("timeout", 0)),
                    "skipped": int(counts.get("skipped", 0)),
                    "status": status,
                    "note": note,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(out_dir: Path, result_dir: Path, log_rows: list[dict], result_rows: list[dict]) -> None:
    log_counts = Counter(row["status"] for row in log_rows)
    result_counts = Counter(row["status"] for row in result_rows)
    by_baseline = defaultdict(Counter)
    for row in log_rows:
        by_baseline[row["baseline"]][row["status"]] += 1

    lines = [
        "# Backend Separation Audit",
        "",
        f"Result directory: `{result_dir}`",
        "",
        "## Verdict",
        "",
    ]
    if log_counts.get("fail", 0) or result_counts.get("fail", 0):
        lines.append("FAIL: at least one backend separation check failed.")
    else:
        lines.append("PASS: all non-timeout EGGPU, easygraph-cpp, and easygraph-cpu logs match the expected isolated modes.")
    lines.extend(
        [
            "",
            "## Expected Contract",
            "",
            "- EGGPU: `mode=gpu`, `EASYGRAPH_ENABLE_GPU=TRUE`, `EASYGRAPH_GPU_BACKEND=mine`, `easygraph_warmup=2`.",
            "- easygraph-cpp: `mode=cpp`, `EASYGRAPH_ENABLE_GPU=FALSE`, empty `EASYGRAPH_GPU_BACKEND`, `easygraph_warmup=0`.",
            "- easygraph-cpu: `mode=cpu`, `EASYGRAPH_ENABLE_GPU=FALSE`, empty `EASYGRAPH_GPU_BACKEND`, `easygraph_warmup=0`.",
            "- easygraph-cpp structural-hole rows are explicitly skipped because those bindings route to CUDA in this build; this preserves a CPU C++ baseline instead of mixing it with EGGPU.",
            "",
            "## Log Checks",
            "",
            "| Baseline | OK | Timeout before entry | Fail |",
            "|---|---:|---:|---:|",
        ]
    )
    for baseline in EXPECTED:
        counts = by_baseline[baseline]
        lines.append(
            f"| {baseline} | {counts.get('ok', 0)} | {counts.get('timeout', 0)} | {counts.get('fail', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Result Rows",
            "",
            "| Metric | Baseline | Rows | OK | Timeout | Skipped | Status |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in result_rows:
        lines.append(
            f"| {row['metric']} | {row['baseline']} | {row['rows']} | {row['ok']} | "
            f"{row['timeout']} | {row['skipped']} | {row['status']} |"
        )

    failed = [row for row in log_rows if row["status"] == "fail"] + [
        row for row in result_rows if row["status"] != "ok"
    ]
    timed_out = [row for row in log_rows if row["status"] == "timeout"]
    if timed_out:
        lines.extend(["", "## Timeout Before Backend Entry", ""])
        lines.append(
            "These baselines hit the per-function timeout before `library_baselines.py` printed the EasyGraph mode header. "
            "They are timeout rows, not successful timings, so they do not mix EGGPU and easygraph-cpp measurements."
        )
        for row in timed_out[:20]:
            lines.append(f"- {row['baseline']} / {row['dataset']} / {row['function']}")
        if len(timed_out) > 20:
            lines.append(f"- ... {len(timed_out) - 20} more")
    if failed:
        lines.extend(["", "## Failures", ""])
        for row in failed[:50]:
            lines.append(f"- `{row}`")
    lines.append("")
    (out_dir / "BACKEND_SEPARATION_AUDIT.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    out_dir = args.out_dir or (result_dir / "audit" / "backend_separation")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_rows, _ = audit_logs(result_dir)
    result_rows = audit_result_rows(result_dir)
    write_csv(out_dir / "backend_separation_logs.csv", log_rows)
    write_csv(out_dir / "backend_separation_result_rows.csv", result_rows)
    write_markdown(out_dir, result_dir, log_rows, result_rows)

    failures = sum(1 for row in log_rows if row["status"] == "fail") + sum(
        1 for row in result_rows if row["status"] != "ok"
    )
    print(f"Wrote backend separation audit to {out_dir}")
    print(f"failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
