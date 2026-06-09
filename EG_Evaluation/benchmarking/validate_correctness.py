#!/usr/bin/env python3
"""Post-process benchmark correctness summaries into explicit validation rows."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


REFERENCE_PRIORITY = (
    "networkx",
    "igraph",
    "easygraph-cpu",
    "easygraph-cpp",
    "nx-cugraph",
    "EGGPU",
)
STRUCTURAL_HOLE_FUNCTIONS = {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}
EASYGRAPH_STRUCTURAL_REFERENCE_PRIORITY = (
    "easygraph-cpu",
    "EGGPU",
    "networkx",
    "igraph",
    "easygraph-cpp",
    "nx-cugraph",
)

FIELDS = (
    "dataset_size",
    "graph_type",
    "dataset",
    "function",
    "baseline",
    "reference",
    "validation_status",
    "details",
    "correctness",
    "reference_correctness",
)


def _parse_correctness(text):
    out = {}
    if not text:
        return out
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,;]+)", str(text)):
        value = value.strip()
        try:
            if re.search(r"[.eE+-]", value):
                out[key] = float(value)
            else:
                out[key] = int(value)
        except Exception:
            out[key] = value
    return out


def _detail_path(parsed):
    path = parsed.get("detail")
    if not path:
        return None
    p = Path(str(path))
    return p if p.exists() else None


def _load_detail(parsed):
    p = _detail_path(parsed)
    if p is None:
        return None, "detail file missing"
    try:
        import numpy as np

        data = np.load(p, allow_pickle=False)
        kind = str(data["kind"].tolist()) if "kind" in data else str(parsed.get("detail_kind", ""))
        values = data["values"]
        sources = data["sources"] if "sources" in data else None
        return {"kind": kind, "values": values, "sources": sources, "path": str(p)}, ""
    except Exception as e:
        return None, f"detail load failed: {e}"


def _detail_tolerance(function):
    if function == "PageRank":
        return 1e-4, 1e-8
    if function in {"LCC", "BC", "Closeness"}:
        return 1e-6, 1e-8
    if function in {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}:
        return 1e-5, 1e-7
    if function in {"SSSP", "BFS", "Dijkstra", "BellmanFord"}:
        return 1e-6, 1e-5
    if function in {"KCore", "CC", "WCC", "SCC"}:
        return 0.0, 0.0
    return 1e-6, 1e-8


def _compare_details(function, parsed, ref_parsed):
    lhs, lhs_err = _load_detail(parsed)
    rhs, rhs_err = _load_detail(ref_parsed)
    if lhs is None or rhs is None:
        return None, lhs_err or rhs_err
    if lhs["kind"] != rhs["kind"]:
        return False, f"detail kind mismatch: {lhs['kind']} vs {rhs['kind']}"
    try:
        import numpy as np

        a = lhs["values"]
        b = rhs["values"]
        if tuple(a.shape) != tuple(b.shape):
            return False, f"detail shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
        if function in {"SSSP", "BFS", "Dijkstra", "BellmanFord"}:
            sa = lhs.get("sources")
            sb = rhs.get("sources")
            if sa is not None and sb is not None and not np.array_equal(sa, sb):
                return False, f"{function} source list mismatch"
            finite_a = np.isfinite(a)
            finite_b = np.isfinite(b)
            if not np.array_equal(finite_a, finite_b):
                diff = int(np.count_nonzero(finite_a != finite_b))
                return False, f"{function} reachable mask differs at {diff} entries"
            if not finite_a.any():
                return True, f"full {function} detail matches"
            rel, abs_ = _detail_tolerance(function)
            ok = np.allclose(a[finite_a], b[finite_b], rtol=rel, atol=abs_)
            if ok:
                return True, f"full {function} detail matches"
            maxdiff = float(np.max(np.abs(a[finite_a] - b[finite_b])))
            return False, f"{function} distance values differ, max_abs_diff={maxdiff:.6g}"
        if function in {"KCore", "CC", "WCC", "SCC"}:
            ok = np.array_equal(a, b)
            if ok:
                return True, f"full {lhs['kind']} detail matches"
            diff = int(np.count_nonzero(a != b))
            return False, f"detail differs at {diff} entries"
        rel, abs_ = _detail_tolerance(function)
        ok = np.allclose(a, b, rtol=rel, atol=abs_, equal_nan=True)
        if ok:
            return True, f"full {lhs['kind']} detail matches"
        maxdiff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        return False, f"detail values differ, max_abs_diff={maxdiff:.6g}"
    except Exception as e:
        return None, f"detail compare failed: {e}"


def _isclose(a, b, rel=1e-6, abs_=1e-8):
    try:
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
    except Exception:
        return False


def _exact(a, b):
    try:
        return int(a) == int(b)
    except Exception:
        return a == b


def _function_checks(function):
    if function == "PageRank":
        return (("sum", "float", 1e-6, 1e-8),), "weak_pass"
    if function == "MST":
        return (("weight", "float", 0.0, 0.0),), "pass"
    if function == "LCC":
        return (
            ("vertices", "int", 0.0, 0.0),
            ("mean", "float", 1e-6, 1e-8),
        ), "pass"
    if function in {"CC", "WCC", "SCC"}:
        return (("components", "int", 0.0, 0.0),), "pass"
    if function in {"SSSP", "BFS", "Dijkstra", "BellmanFord"}:
        return (
            ("sources", "int", 0.0, 0.0),
            ("reachable", "int", 0.0, 0.0),
            ("checksum", "float", 1e-6, 1e-5),
        ), "pass"
    if function == "KCore":
        return (
            ("nodes", "int", 0.0, 0.0),
            ("sum", "float", 1e-8, 1e-6),
            ("max", "int", 0.0, 0.0),
        ), "pass"
    if function == "BC":
        return (
            ("nodes", "int", 0.0, 0.0),
            ("sum", "float", 1e-5, 1e-4),
        ), "pass"
    if function == "Closeness":
        return (
            ("nodes", "int", 0.0, 0.0),
            ("sum", "float", 1e-5, 1e-6),
            ("mean", "float", 1e-6, 1e-8),
        ), "pass"
    if function in {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}:
        return (
            ("nodes", "int", 0.0, 0.0),
            ("sum", "float", 1e-5, 1e-5),
            ("mean", "float", 1e-5, 1e-7),
        ), "pass"
    return (), "pass"


def _reference_priority(function):
    if function in STRUCTURAL_HOLE_FUNCTIONS:
        return EASYGRAPH_STRUCTURAL_REFERENCE_PRIORITY
    return REFERENCE_PRIORITY


def _pick_reference(rows, function):
    by_backend = {r.get("baseline"): r for r in rows}
    for backend in _reference_priority(function):
        hit = by_backend.get(backend)
        if hit is not None:
            parsed = _parse_correctness(hit.get("correctness", ""))
            if parsed:
                return hit, parsed
    return None, {}


def _validate_one(row, ref_row, ref_parsed):
    function = row.get("function", "")
    baseline = row.get("baseline", "")
    notes = str(row.get("notes", ""))
    parsed = _parse_correctness(row.get("correctness", ""))
    if not parsed:
        return "inconclusive", "no comparable correctness fields"
    if ref_row is None:
        return "inconclusive", "no reference baseline with correctness fields"
    if baseline == ref_row.get("baseline"):
        if baseline == "EGGPU":
            return (
                "inconclusive_self_reference",
                "EGGPU is the only available reference; correctness is not externally established",
            )
        return "reference", "selected reference baseline"
    if baseline == "Gunrock" and function == "MST" and (
        "largest connected component" in notes or "connected-graph" in notes
    ):
        return "semantic_mismatch", "Gunrock MST row is component/connected-graph semantics, not full spanning forest"
    if baseline == "Gunrock" and "validation_errors" in parsed:
        try:
            if int(parsed["validation_errors"]) == 0:
                return "pass", "Gunrock internal validation reports zero errors"
        except Exception:
            pass

    detail_ok, detail_msg = _compare_details(function, parsed, ref_parsed)
    if detail_ok is True:
        return "pass", detail_msg
    if detail_ok is False:
        return "fail", detail_msg

    checks, pass_status = _function_checks(function)
    if not checks:
        return "inconclusive", f"no validation rule for function={function}"

    missing = [name for name, _, _, _ in checks if name not in parsed or name not in ref_parsed]
    if missing:
        return "inconclusive", "missing field(s): " + ",".join(missing)

    diffs = []
    for name, kind, rel, abs_ in checks:
        if kind == "int":
            ok = _exact(parsed[name], ref_parsed[name])
        else:
            if rel == 0.0 and abs_ == 0.0:
                ok = _exact(round(float(parsed[name])), round(float(ref_parsed[name])))
            else:
                ok = _isclose(parsed[name], ref_parsed[name], rel=rel, abs_=abs_)
        if not ok:
            diffs.append(f"{name}: got {parsed[name]} vs ref {ref_parsed[name]}")
    if diffs:
        return "fail", "; ".join(diffs)
    if pass_status == "weak_pass":
        return "weak_pass", f"only summary fields match reference ({detail_msg})"
    return pass_status, "summary fields match reference" + (f"; {detail_msg}" if detail_msg else "")


def validate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("metric") != "e2e":
            continue
        if row.get("status") != "ok":
            continue
        groups[(row.get("dataset"), row.get("function"))].append(row)

    out = []
    for (dataset, function), group in sorted(groups.items()):
        ref_row, ref_parsed = _pick_reference(group, function)
        for row in sorted(group, key=lambda r: r.get("baseline", "")):
            status, details = _validate_one(row, ref_row, ref_parsed)
            out.append(
                {
                    "dataset_size": row.get("dataset_size", ""),
                    "graph_type": row.get("graph_type", ""),
                    "dataset": dataset,
                    "function": function,
                    "baseline": row.get("baseline", ""),
                    "reference": "" if ref_row is None else ref_row.get("baseline", ""),
                    "validation_status": status,
                    "details": details,
                    "correctness": row.get("correctness", ""),
                    "reference_correctness": "" if ref_row is None else ref_row.get("correctness", ""),
                }
            )
    return out


def write_validation_outputs(out_dir, rows):
    out_dir = Path(out_dir)
    validation = validate_rows(rows)
    csv_path = out_dir / "correctness_validation.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(validation)

    counts = defaultdict(int)
    for row in validation:
        counts[row["validation_status"]] += 1
    failures = [r for r in validation if r["validation_status"] in {"fail", "semantic_mismatch", "inconclusive"}]
    md = ["# Correctness Validation", ""]
    if counts:
        md.append("Status counts:")
        for key in ("pass", "weak_pass", "reference", "semantic_mismatch", "fail", "inconclusive"):
            if counts.get(key):
                md.append(f"- {key}: {counts[key]}")
        md.append("")
    md.append("Validation compares full detail dumps when present. If a baseline lacks a detail dump, it falls back to the legacy summary fields.")
    md.append("Reference priority: networkx, igraph, easygraph-cpu, easygraph-cpp, nx-cugraph, EGGPU.")
    md.append("Structural-hole metrics use EasyGraph-compatible semantics: easygraph-cpu, EGGPU, networkx, igraph, easygraph-cpp, nx-cugraph.")
    md.append("PageRank is only marked `weak_pass` when no full vector detail is available.")
    md.append("EGGPU self-reference rows are marked `inconclusive_self_reference`; final audits require EGGPU rows to be externally validated as `pass`.")
    if failures:
        md.extend(["", "Non-passing rows:"])
        for row in failures[:200]:
            md.append(
                f"- {row['dataset']} / {row['function']} / {row['baseline']}: "
                f"{row['validation_status']} ({row['details']})"
            )
        if len(failures) > 200:
            md.append(f"- ... {len(failures) - 200} more rows omitted")
    (out_dir / "correctness_validation.md").write_text("\n".join(md) + "\n")
    return validation


def read_results_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", help="Benchmark result directory containing results_long.csv")
    args = ap.parse_args()
    result_dir = Path(args.result_dir)
    rows = read_results_csv(result_dir / "results_long.csv")
    write_validation_outputs(result_dir, rows)
    print(result_dir / "correctness_validation.csv")


if __name__ == "__main__":
    main()
