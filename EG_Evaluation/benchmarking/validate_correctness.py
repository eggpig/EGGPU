#!/usr/bin/env python3
"""Post-process benchmark correctness summaries into explicit validation rows."""

import argparse
import csv
import math
import os
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
UNSAFE_REFERENCE_BY_FUNCTION = {
    # nx-cugraph BC can use a different normalization/semantic path than
    # NetworkX/igraph/EasyGraph on large graphs.  It remains a measured
    # baseline, but it must not become the oracle when exact CPU references
    # time out on a slower reproduction machine.
    "BC": {"nx-cugraph"},
}
STRUCTURAL_HOLE_FUNCTIONS = {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATHS = {
    "ca-GrQc": "datasets/undirected/ca-GrQc.txt",
    "ca-HepTh": "datasets/undirected/ca-HepTh.txt",
    "LastFM": "datasets/undirected/LastFM.txt",
    "pgp": "datasets/undirected/pgp.txt",
    "ca-CondMat": "datasets/undirected/ca-CondMat.txt",
    "ca-HepPh": "datasets/undirected/ca-HepPh.txt",
    "email-Enron": "datasets/undirected/email-Enron.txt",
    "com-youtube": "datasets/undirected/com-youtube.ungraph.txt",
    "p2p-Gnutella04": "datasets/directed/p2p-Gnutella04.txt",
    "p2p-Gnutella08": "datasets/directed/p2p-Gnutella08.txt",
    "wiki-Vote": "datasets/directed/wiki-Vote.txt",
    "soc-Epinions1": "datasets/directed/soc-Epinions1.txt",
    "email-EuAll": "datasets/directed/email-EuAll.txt",
    "soc-Slashdot0811": "datasets/directed/soc-Slashdot0811.txt",
    "web-NotreDame": "datasets/directed/web-NotreDame.txt",
    "ER-100k": "datasets/directed/ER-100k.txt",
    "wiki-Talk": "datasets/directed/wiki-Talk.txt",
}
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
_STRUCTURAL_GRAPH_CACHE = {}


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


def _normalized_clean_edges(dataset):
    cached = _STRUCTURAL_GRAPH_CACHE.get(dataset)
    if cached is not None:
        return cached
    rel = DATASET_PATHS.get(dataset)
    if rel is None:
        raise KeyError(f"unknown dataset={dataset}")
    path = ROOT / rel
    raw_edges = []
    nodes = set()
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#%/c":
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            raw_edges.append((u, v))
            nodes.add(u)
            nodes.add(v)
    ordered = sorted(nodes)
    remap = {node: idx for idx, node in enumerate(ordered)}
    directed = sorted({(remap[u], remap[v]) for u, v in raw_edges})
    undirected = sorted({(min(remap[u], remap[v]), max(remap[u], remap[v])) for u, v in raw_edges})
    out = {"n": len(ordered), "directed": directed, "undirected": undirected}
    _STRUCTURAL_GRAPH_CACHE[dataset] = out
    return out


def _structural_graph(dataset, graph_type):
    cache_key = (dataset, graph_type)
    cached = _STRUCTURAL_GRAPH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    data = _normalized_clean_edges(dataset)
    n = data["n"]
    directed = graph_type == "directed"
    if directed:
        out_adj = [set() for _ in range(n)]
        in_adj = [set() for _ in range(n)]
        for u, v in data["directed"]:
            out_adj[u].add(v)
            in_adj[v].add(u)
    else:
        out_adj = [set() for _ in range(n)]
        in_adj = out_adj
        for u, v in data["undirected"]:
            out_adj[u].add(v)
            out_adj[v].add(u)
    degrees = [
        (len(out_adj[i]) + len(in_adj[i]) if directed else len(out_adj[i]))
        for i in range(n)
    ]
    if directed:
        sum_scale = degrees
        max_scale = [
            2 if any(v in out_adj[u] for u in out_adj[v]) else (1 if degrees[v] > 0 else 0)
            for v in range(n)
        ]
    else:
        sum_scale = degrees
        max_scale = [1 if degree > 0 else 0 for degree in degrees]
    top_degree_nodes = sorted(range(n), key=lambda idx: degrees[idx], reverse=True)[:32]
    graph = {
        "n": n,
        "directed": directed,
        "out": out_adj,
        "in": in_adj,
        "degrees": degrees,
        "sum_scale": sum_scale,
        "max_scale": max_scale,
        "top_degree_nodes": top_degree_nodes,
    }
    _STRUCTURAL_GRAPH_CACHE[cache_key] = graph
    return graph


def _all_neighbors(graph, node):
    if graph["directed"]:
        return graph["out"][node] | graph["in"][node]
    return graph["out"][node]


def _successor_count(graph, node):
    return len(graph["out"][node])


def _mutual_count(graph, u, v):
    if graph["directed"]:
        return int(v in graph["out"][u]) + int(u in graph["out"][v])
    return int(v in graph["out"][u])


def _normalized_mutual_weight(graph, u, v, norm):
    scale = graph["sum_scale"][u] if norm == "sum" else graph["max_scale"][u]
    return 0.0 if scale == 0 else float(_mutual_count(graph, u, v)) / float(scale)


def _structural_redundancy(graph, u, v):
    total = 0.0
    for w in _all_neighbors(graph, u):
        total += _normalized_mutual_weight(graph, u, w, "sum") * _normalized_mutual_weight(graph, v, w, "max")
    return 1.0 - total


def _structural_local_constraint(graph, u, v, cache):
    key = (u, v)
    hit = cache.get(key)
    if hit is not None:
        return hit
    direct = _normalized_mutual_weight(graph, u, v, "sum")
    indirect = 0.0
    for w in _all_neighbors(graph, u):
        indirect += _normalized_mutual_weight(graph, u, w, "sum") * _normalized_mutual_weight(graph, w, v, "sum")
    value = (direct + indirect) ** 2
    cache[key] = value
    return value


def _sampled_structural_value(function, graph, node, local_cache):
    neighbors = _all_neighbors(graph, node)
    if function == "EffectiveSize" and not graph["directed"]:
        if _successor_count(graph, node) == 0:
            return float("nan")
        degree = len(neighbors)
        internal_edges = 0
        ordered = list(neighbors)
        for i, u in enumerate(ordered):
            rest = ordered[i + 1 :]
            if rest:
                internal_edges += sum(1 for v in rest if v in graph["out"][u])
        return float(degree) if internal_edges == 0 else float(degree) - (2.0 * float(internal_edges)) / float(degree)

    if function == "EffectiveSize":
        if _successor_count(graph, node) == 0:
            return float("nan")
        return sum(_structural_redundancy(graph, node, u) for u in neighbors)

    if function == "Efficiency":
        effective = _sampled_structural_value("EffectiveSize", graph, node, local_cache)
        degree = graph["degrees"][node]
        return float("nan") if degree == 0 else effective / float(degree)

    if function == "Constraint":
        if _successor_count(graph, node) == 0:
            return float("nan")
        return sum(_structural_local_constraint(graph, node, u, local_cache) for u in neighbors)

    if function == "Hierarchy":
        c = {}
        total = 0.0
        for u in neighbors:
            value = _structural_local_constraint(graph, node, u, local_cache)
            c[u] = value
            total += value
        n = len(neighbors)
        if n <= 1 or total == 0.0:
            return 0.0
        denom = float(n) * math.log(float(n))
        return sum((value / total) * float(n) * math.log((value / total) * float(n)) / denom for value in c.values())

    raise ValueError(f"not a structural-hole function: {function}")


def _ordered_sample(values, graph, limit=24):
    import numpy as np

    n = int(graph["n"])
    max_degree = int(os.environ.get("EGGPU_STRUCTURAL_VALIDATION_MAX_SAMPLE_DEGREE", "2048"))
    out = []

    def add(node):
        node = int(node)
        if 0 <= node < n and node not in out and graph["degrees"][node] <= max_degree:
            out.append(node)

    for node in (0, n // 4, n // 2, (3 * n) // 4, n - 1):
        add(node)
    for node in graph["top_degree_nodes"]:
        add(node)
        if len(out) >= 8:
            break
    arr = np.asarray(values)
    finite = np.flatnonzero(np.isfinite(arr))
    if finite.size:
        take = min(8, int(finite.size))
        top_pos = finite[np.argpartition(np.abs(arr[finite]), -take)[-take:]]
        for node in top_pos[np.argsort(np.abs(arr[top_pos]))[::-1]]:
            add(node)
    nan_nodes = np.flatnonzero(np.isnan(arr))
    for node in nan_nodes[:3]:
        add(node)
    step = max(1, n // max(1, limit))
    for node in range(0, n, step):
        add(node)
        if len(out) >= limit:
            break
    if not out:
        out.append(min(range(n), key=lambda idx: graph["degrees"][idx]))
    return out[:limit]


def _sampled_structural_check(row, function, parsed):
    detail, detail_err = _load_detail(parsed)
    if detail is None:
        return None, detail_err
    if detail.get("kind") != "vector":
        return None, f"sampled structural validation needs vector detail, got {detail.get('kind')}"
    try:
        import numpy as np

        graph = _structural_graph(row.get("dataset", ""), row.get("graph_type", ""))
        values = np.asarray(detail["values"], dtype=float)
        if values.shape != (graph["n"],):
            return False, f"sampled structural detail shape mismatch: {values.shape} vs {(graph['n'],)}"
        sample = _ordered_sample(values, graph)
        local_cache = {}
        bad = []
        max_diff = 0.0
        for node in sample:
            expected = _sampled_structural_value(function, graph, node, local_cache)
            observed = float(values[node])
            if math.isnan(expected) and math.isnan(observed):
                continue
            diff = abs(expected - observed)
            max_diff = max(max_diff, diff)
            if not math.isclose(expected, observed, rel_tol=1e-5, abs_tol=1e-7):
                bad.append((node, observed, expected, diff))
        if bad:
            node, observed, expected, diff = bad[0]
            return (
                False,
                f"sampled exact CPU structural-hole validation failed at node={node}: "
                f"got {observed:.9g} vs expected {expected:.9g}, abs_diff={diff:.6g}, "
                f"sampled_nodes={len(sample)}, max_abs_diff={max_diff:.6g}",
            )
        return (
            True,
            f"sampled exact CPU structural-hole validation matches {len(sample)} nodes, "
            f"max_abs_diff={max_diff:.6g}",
        )
    except Exception as exc:
        return None, f"sampled structural validation unavailable: {exc}"


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
    unsafe = UNSAFE_REFERENCE_BY_FUNCTION.get(function, set())
    for backend in _reference_priority(function):
        if backend in unsafe:
            continue
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
            if function in STRUCTURAL_HOLE_FUNCTIONS:
                sampled_ok, sampled_msg = _sampled_structural_check(row, function, parsed)
                if sampled_ok is True:
                    return "sampled_pass", sampled_msg
                if sampled_ok is False:
                    return "fail", sampled_msg
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
        for key in ("pass", "sampled_pass", "weak_pass", "reference", "semantic_mismatch", "fail", "inconclusive", "inconclusive_self_reference"):
            if counts.get(key):
                md.append(f"- {key}: {counts[key]}")
        md.append("")
    md.append("Validation compares full detail dumps when present. If a baseline lacks a detail dump, it falls back to the legacy summary fields.")
    md.append("Reference priority: networkx, igraph, easygraph-cpu, easygraph-cpp, nx-cugraph, EGGPU.")
    md.append("Function-specific unsafe references are skipped when selecting an oracle; currently BC does not use nx-cugraph as an oracle because its large-graph values differ from NetworkX/igraph/EasyGraph semantics in existing validation rows.")
    md.append("Structural-hole metrics use EasyGraph-compatible semantics: easygraph-cpu, EGGPU, networkx, igraph, easygraph-cpp, nx-cugraph.")
    md.append("Large structural-hole EGGPU rows without a full external reference use deterministic sampled exact CPU validation and are marked `sampled_pass`.")
    md.append("PageRank is only marked `weak_pass` when no full vector detail is available.")
    md.append("EGGPU self-reference rows are marked `inconclusive_self_reference`; final audits require EGGPU rows to be externally validated as `pass` or `sampled_pass`.")
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
